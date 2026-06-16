"""ctvlm — user-facing Streamlit app.

Friendly frontend with three workflows:
  - Demo Studies: browse pre-cached features for the bundled demo SIDs (no GPU)
  - Upload DICOM: drag-drop a DICOM series → DICOM→NIfTI → encoder forward →
    full RATE-225 prediction (needs GPU + cached encoder weights)
  - All 220 Findings: taxonomy explorer + per-finding probability for the
    currently-loaded study

Run:
    streamlit run viewer/web_app.py
or via Docker:
    docker compose up viewer
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Bind demo-friendly defaults BEFORE any path-config import
os.environ.setdefault("CTVLM_WORK_ROOT", str(REPO_ROOT / "data" / "demo_features"))
os.environ.setdefault("CTVLM_CHECKPOINTS_DIR", str(REPO_ROOT / "checkpoints"))

from src.agent import pipeline, render                                    # noqa: E402
from src.agent.canonical import (CANONICAL_TO_CATEGORY,                   # noqa: E402
                                  CATEGORY_TO_CANONICALS, canonical)

DEMO_FEATURES_ROOT = REPO_ROOT / "data" / "demo_features"
UPLOAD_FEATURES_ROOT = Path(os.environ.get(
    "CTVLM_UPLOAD_WORK", str(REPO_ROOT / "data" / "uploaded")))
UPLOAD_FEATURES_ROOT.mkdir(parents=True, exist_ok=True)
(UPLOAD_FEATURES_ROOT / "merlin_global").mkdir(parents=True, exist_ok=True)
(UPLOAD_FEATURES_ROOT / "pillar0_emb").mkdir(parents=True, exist_ok=True)
(UPLOAD_FEATURES_ROOT / "ct_volumes").mkdir(parents=True, exist_ok=True)

REPORTS_CSV = REPO_ROOT / "data" / "reports_25k.csv"
RATE_CSV = REPO_ROOT / "data" / "finding_labels_rate.csv"


# ──────────────────────────────────────────────────────────────────────────
# Page config + styling
# ──────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ctvlm — radiology report generator",
    page_icon="🫁",
    layout="wide",
    menu_items={
        "Get Help": "https://github.com/<owner>/ctvlm/blob/main/docs/01_OVERVIEW.md",
        "Report a bug": "https://github.com/<owner>/ctvlm/issues",
        "About": "ctvlm — agentic CT abdomen-pelvis report generator. MIT license. NOT FDA-cleared.",
    },
)

st.markdown("""
<style>
    .stApp { background: linear-gradient(180deg, #fafbfc 0%, #f3f5f8 100%); }
    .ctvlm-hero {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white; padding: 2rem 2.5rem; border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    .ctvlm-hero h1 { color: white; margin: 0; font-size: 2rem; }
    .ctvlm-hero p  { color: #dbeafe; margin: 0.5rem 0 0 0; font-size: 1.05rem; }
    .ctvlm-badge {
        display: inline-block; padding: 2px 10px; border-radius: 10px;
        font-size: 0.78rem; font-weight: 600; margin-right: 4px;
    }
    .badge-A { background: #dcfce7; color: #166534; }
    .badge-B { background: #e0e7ff; color: #3730a3; }
    .badge-C { background: #f1f5f9; color: #475569; }
    .badge-prob-hi { background: #fee2e2; color: #991b1b; }
    .badge-prob-md { background: #fef3c7; color: #92400e; }
    .badge-prob-lo { background: #ecfdf5; color: #065f46; }
    .ctvlm-finding-card {
        background: white; border-radius: 8px; padding: 0.9rem 1.1rem;
        margin-bottom: 0.6rem; border-left: 3px solid #3b82f6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .ctvlm-finding-title { font-weight: 600; font-size: 1.0rem; margin: 0; color: #1e293b; }
    .ctvlm-finding-sentence { margin: 0.4rem 0 0 0; color: #334155; line-height: 1.45; }
    .ctvlm-footer { color: #64748b; font-size: 0.82rem; margin-top: 2rem;
                    padding-top: 1rem; border-top: 1px solid #e2e8f0; }
    [data-testid="stHeader"] { background: transparent; }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] { padding: 8px 16px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="ctvlm-hero">
    <h1>🫁 ctvlm</h1>
    <p>Agentic radiology report generator for CT abdomen-pelvis · 220 RATE findings · template-only, deterministic, auditable · MIT</p>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────
# Sidebar — operating point + dataset access
# ──────────────────────────────────────────────────────────────────────────

st.sidebar.markdown("### Operating point")
contrast_phase = st.sidebar.radio(
    "Contrast phase",
    options=["ce", "nc"],
    format_func=lambda x: "Contrast-enhanced" if x == "ce" else "Non-contrast",
    help="Routes through phase-specific Platt calibration. ~99% of training was "
         "CE; NC calibration covers 26 high-prevalence findings."
)
min_threshold = st.sidebar.slider(
    "Minimum probability floor", 0.05, 0.50, 0.20, 0.05,
    help="Universal lower bound on per-finding Youden-J thresholds. "
         "0.20 is the validated F1 knee on the 5,082-case val split."
)
max_findings = st.sidebar.slider("Max findings rendered", 5, 25, 12, 1)

st.sidebar.markdown("---")
st.sidebar.markdown("### Get the Merlin CT dataset")
st.sidebar.markdown(
    "The CT volumes themselves are hosted by Stanford / Azure (not bundled "
    "in this repo). Required if you want to run the encoders on the original "
    "25k cohort or pre-cache features for new studies."
)
st.sidebar.markdown(
    "🔗 [Stanford AIMI — Merlin (HF Datasets)](https://huggingface.co/datasets/stanfordmimi/Merlin)  \n"
    "🔗 [Stanford AIMI Shared Datasets](https://aimi.stanford.edu/shared-datasets)  \n"
    "🔗 [Azure Open Datasets — Medical Imaging](https://learn.microsoft.com/en-us/azure/open-datasets/dataset-medical-imaging)  \n"
    "🔗 [Merlin model card (encoder)](https://huggingface.co/stanfordmimi/Merlin)  \n"
    "🔗 [Pillar-0 model card (encoder)](https://huggingface.co/YalaLab/Pillar0-AbdomenCT)",
    unsafe_allow_html=False,
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Pipeline")
st.sidebar.markdown(
    "Linear probe over L2[Merlin‖Pillar-0] (3200-d) → 220 findings  \n"
    "Platt + Youden-J calibration, phase-aware  \n"
    "COREQUIRES + SUBSUMPTION + per-category cap  \n"
    "Tier A/B template render, anatomical-section ordered"
)


# ──────────────────────────────────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────────────────────────────────

@st.cache_data
def list_demo_studies() -> list[str]:
    merlin = {p.stem for p in (DEMO_FEATURES_ROOT / "merlin_global").glob("*.npy")}
    pillar0 = {p.stem for p in (DEMO_FEATURES_ROOT / "pillar0_emb").glob("*.npy")}
    return sorted(merlin & pillar0)


def list_uploaded_studies() -> list[str]:
    merlin = {p.stem for p in (UPLOAD_FEATURES_ROOT / "merlin_global").glob("*.npy")}
    pillar0 = {p.stem for p in (UPLOAD_FEATURES_ROOT / "pillar0_emb").glob("*.npy")}
    return sorted(merlin & pillar0)


@st.cache_data
def load_ground_truth_report(sid: str) -> str | None:
    if not REPORTS_CSV.exists():
        return None
    try:
        df = pd.read_csv(REPORTS_CSV, usecols=["Accession", "Report Text"])
    except Exception:
        return None
    row = df[df["Accession"] == sid]
    return row.iloc[0]["Report Text"] if not row.empty else None


@st.cache_data
def load_ground_truth_findings(sid: str) -> set[str]:
    if not RATE_CSV.exists():
        return set()
    try:
        df = pd.read_csv(RATE_CSV).set_index("study_id")
    except Exception:
        return set()
    if sid not in df.index:
        return set()
    row = df.loc[sid]
    positives = set()
    for col, val in row.items():
        if val == 1:
            c = canonical(col)
            if c:
                positives.add(c)
    return positives


def run_pipeline_for_sid(sid: str, features_root: Path):
    """Run the pipeline by temporarily redirecting feature paths."""
    # Re-import pipeline module so we can override its DEFAULT_PROBE and feature dirs
    import importlib
    from src.agent import pipeline as _p
    importlib.reload(_p)
    _p.MERLIN_FEATURES = features_root / "merlin_global"
    _p.PILLAR0_FEATURES = features_root / "pillar0_emb"
    return _p.generate_report(
        sid,
        skip_llm=True,
        contrast_phase=contrast_phase,
        min_threshold=min_threshold,
        max_findings=max_findings,
    )


# ──────────────────────────────────────────────────────────────────────────
# Shared report-render helpers (called from each tab)
# ──────────────────────────────────────────────────────────────────────────

def _render_report(report, sid: str, show_gt: bool = True):
    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        st.markdown(f"### Structured FINDINGS — `{sid}`")
        n_tierA = sum(1 for s in report.structured if s.recipe_tier == "A")
        st.caption(
            f"Encoder: `{report.encoder}` · "
            f"{len(report.positives)} positives ({n_tierA} Tier A) · "
            f"{report.latency_total_s*1000:.0f} ms"
        )

        if not report.structured:
            st.success(
                "No findings above threshold for this study. (See \"All 220 "
                "findings\" tab for per-finding probabilities.)"
            )

        for sf in report.structured:
            tier_badge = f'<span class="ctvlm-badge badge-{sf.recipe_tier}">Tier {sf.recipe_tier}</span>'
            if sf.probability >= 0.7:
                prob_badge = f'<span class="ctvlm-badge badge-prob-hi">p={sf.probability:.2f}</span>'
            elif sf.probability >= 0.4:
                prob_badge = f'<span class="ctvlm-badge badge-prob-md">p={sf.probability:.2f}</span>'
            else:
                prob_badge = f'<span class="ctvlm-badge badge-prob-lo">p={sf.probability:.2f}</span>'
            st.markdown(f"""
            <div class="ctvlm-finding-card">
                <p class="ctvlm-finding-title">{sf.finding.replace('_', ' ').title()} {tier_badge} {prob_badge}</p>
                <p class="ctvlm-finding-sentence">{sf.sentence}</p>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("📄 Full structured summary block", expanded=False):
            st.code(report.summary_text, language="text")

    with col_right:
        if show_gt:
            gt_findings = load_ground_truth_findings(sid)
            if gt_findings:
                pred = set(report.positives)
                tp = pred & gt_findings; fp = pred - gt_findings; fn = gt_findings - pred
                st.markdown("### Compared to ground truth (RATE labels)")
                a, b, c = st.columns(3)
                a.metric("TP", len(tp))
                b.metric("FP", len(fp))
                c.metric("FN", len(fn))
                if fp or fn:
                    with st.expander("Mismatches", expanded=False):
                        if fp: st.markdown(f"**FPs:** {', '.join(sorted(fp))}")
                        if fn: st.markdown(f"**FNs:** {', '.join(sorted(fn))}")
            gt_report = load_ground_truth_report(sid)
            if gt_report:
                with st.expander("📑 Source radiology report (GT)", expanded=False):
                    st.markdown(gt_report.replace("\n", "  \n"))

        st.markdown("### Top probabilities")
        probs_df = (
            pd.DataFrame(list(report.probabilities.items()),
                          columns=["finding", "probability"])
              .sort_values("probability", ascending=False)
              .head(15)
              .reset_index(drop=True)
        )
        st.dataframe(probs_df, use_container_width=True, hide_index=True,
                      column_config={
                          "probability": st.column_config.ProgressColumn(
                              "Probability", format="%.3f",
                              min_value=0.0, max_value=1.0)
                      })


# ──────────────────────────────────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────────────────────────────────

tab_demo, tab_upload, tab_taxonomy, tab_about = st.tabs([
    "🩻 Demo studies",
    "📤 Upload DICOM",
    "🧭 All 220 findings",
    "ℹ️ About + docs",
])


# ─── Tab: Demo studies ──────────────────────────────────────────────────── #

with tab_demo:
    demo_sids = list_demo_studies()
    if not demo_sids:
        st.warning(
            "No demo features cached. Place `.npy` files under "
            "`data/demo_features/merlin_global/` and `data/demo_features/pillar0_emb/`."
        )
    else:
        st.markdown(f"_{len(demo_sids)} bundled studies. Pre-cached encoder "
                     f"features → probe + Platt + Youden-J + filter + render < 100 ms._")
        sid = st.selectbox(
            "Study", demo_sids,
            index=demo_sids.index("AC421363f") if "AC421363f" in demo_sids else 0,
            key="demo_sid"
        )
        with st.spinner(f"Running pipeline on {sid} ({contrast_phase})..."):
            try:
                report = run_pipeline_for_sid(sid, DEMO_FEATURES_ROOT)
                st.session_state["last_report"] = report
                st.session_state["last_sid"] = sid
            except FileNotFoundError as e:
                st.error(f"Missing feature for {sid}: {e}")
                st.stop()
        _render_report(report, sid, show_gt=True)


# ─── Tab: Upload DICOM ──────────────────────────────────────────────────── #

with tab_upload:
    st.markdown("### Upload a CT abdomen-pelvis DICOM series")
    st.markdown(
        "Drop the **entire series** of `.dcm` files for one study (a typical "
        "thin-slice abdomen-pelvis CT is 200-800 files, ~200-800 MB). The app "
        "will:\n\n"
        "1. Stage the files into a temp dir.\n"
        "2. Convert DICOM → canonical-RAS NIfTI with `dcm2niix`.\n"
        "3. Run **Merlin** and **Pillar-0** encoders (~5 s combined on GPU, "
        "first time each study). Features cache to disk.\n"
        "4. Forward through the **concat probe → Platt → Youden-J → filter → "
        "render**.\n"
        "5. Display the structured FINDINGS block + per-finding probabilities."
    )

    uploaded = st.file_uploader(
        "Drop DICOM files here",
        type=["dcm"],
        accept_multiple_files=True,
        key="dicom_upload",
        help="Tip: zip a series first and `unzip` outside Streamlit if many "
             "hundreds of files exceed the upload widget's limit."
    )

    study_id_input = st.text_input(
        "Study ID (any identifier — used as the cache key)",
        value="upload_001",
        help="Anything alphanumeric. The encoder features will be cached "
             "under this key so re-running is instant."
    )

    if uploaded and st.button("▶️ Convert + analyse", type="primary",
                                disabled=not study_id_input.strip()):
        sid = study_id_input.strip()

        # Stage files
        with tempfile.TemporaryDirectory(prefix=f"ctvlm_{sid}_") as tmp:
            tmp = Path(tmp)
            for f in uploaded:
                (tmp / f.name).write_bytes(f.getbuffer())

            try:
                from deploy.example_dicom_to_nifti import convert
            except ImportError:
                st.error("`deploy/example_dicom_to_nifti` not on path. "
                          "Make sure the repo is installed: `pip install -e .`")
                st.stop()

            # 1. DICOM → NIfTI
            with st.status(f"Converting {len(uploaded)} DICOM files → NIfTI ...",
                            expanded=False) as status:
                try:
                    out = convert(tmp, UPLOAD_FEATURES_ROOT / "ct_volumes", sid)
                    status.update(label=f"✅ NIfTI ready: {out.name}", state="complete")
                except Exception as e:
                    status.update(label=f"❌ dcm2niix failed: {e}", state="error")
                    st.stop()

            # 2. Force encoder forward (writes features to UPLOAD_FEATURES_ROOT)
            with st.status("Running Merlin + Pillar-0 encoders (~5 s on GPU) ...",
                            expanded=False) as status:
                try:
                    # Override the source CT path so encoders find this study
                    from src.config import load_paths
                    load_paths.cache_clear()
                    os.environ["CTVLM_MERLIN_ROOT"] = str(
                        UPLOAD_FEATURES_ROOT / "ct_volumes")
                    os.environ["CTVLM_WORK_ROOT"] = str(UPLOAD_FEATURES_ROOT)
                    # The pipeline auto-extracts features if not cached; just call predict
                    import importlib
                    from src.agent import pipeline as _p
                    importlib.reload(_p)
                    _p.MERLIN_FEATURES = UPLOAD_FEATURES_ROOT / "merlin_global"
                    _p.PILLAR0_FEATURES = UPLOAD_FEATURES_ROOT / "pillar0_emb"
                    report = _p.generate_report(
                        sid, skip_llm=True, contrast_phase=contrast_phase,
                        min_threshold=min_threshold, max_findings=max_findings,
                    )
                    status.update(label="✅ Inference complete", state="complete")
                except Exception as e:
                    status.update(label=f"❌ Pipeline failed: {e}", state="error")
                    st.exception(e)
                    st.stop()

        st.session_state["last_report"] = report
        st.session_state["last_sid"] = sid
        st.success(f"Analysis complete for `{sid}` — see results below.")
        _render_report(report, sid, show_gt=False)
    elif uploaded:
        st.info(f"📁 {len(uploaded)} files staged. Click **Convert + analyse** "
                 f"to process.")

    # Show previously uploaded
    uploaded_sids = list_uploaded_studies()
    if uploaded_sids:
        st.markdown("---")
        st.markdown(f"### Previously uploaded ({len(uploaded_sids)})")
        cached_sid = st.selectbox("Re-render a cached upload", uploaded_sids,
                                   key="cached_upload_sid")
        if st.button("▶️ Re-render (cached features)"):
            report = run_pipeline_for_sid(cached_sid, UPLOAD_FEATURES_ROOT)
            st.session_state["last_report"] = report
            st.session_state["last_sid"] = cached_sid
            _render_report(report, cached_sid, show_gt=False)


# ─── Tab: All 220 findings ──────────────────────────────────────────────── #

with tab_taxonomy:
    st.markdown("### Browse the full RATE-225 taxonomy")
    st.markdown(
        "The probe outputs **220 unique canonical findings** organised across "
        "**17 RATE categories**. Drill in by category, search by name, or see "
        "the per-finding probability + tier from the most-recently-loaded study."
    )

    last_report = st.session_state.get("last_report")
    last_sid = st.session_state.get("last_sid")

    if last_report:
        st.caption(f"Showing probabilities from study `{last_sid}` "
                    f"({last_report.encoder})")

    search = st.text_input("🔍 Search findings", placeholder="e.g. hepatic, pancreas, ...",
                            key="finding_search")

    col_cats = st.columns(3)
    sel_categories = []
    all_cats = sorted(CATEGORY_TO_CANONICALS.keys())
    for i, cat in enumerate(all_cats):
        with col_cats[i % 3]:
            if st.checkbox(f"{cat} ({len(CATEGORY_TO_CANONICALS[cat])})",
                            value=True, key=f"cat_{cat}"):
                sel_categories.append(cat)

    rows = []
    from src.agent.recipes import HAND_WRITTEN as TIER_A_RECIPES
    for cat in sel_categories:
        for f in sorted(CATEGORY_TO_CANONICALS[cat]):
            if search and search.lower() not in f.lower():
                continue
            tier = "A" if f in TIER_A_RECIPES else "B"
            prob = (last_report.probabilities.get(f, float("nan"))
                     if last_report else float("nan"))
            rows.append({"category": cat, "finding": f, "tier": tier,
                          "probability": prob})

    df = pd.DataFrame(rows)
    if df.empty:
        st.info("No findings match the current filters.")
    else:
        df = df.sort_values(["category", "finding"]).reset_index(drop=True)
        col_config = {
            "category": "Category",
            "finding": "Canonical name",
            "tier": st.column_config.TextColumn("Tier", width="small"),
        }
        if last_report:
            col_config["probability"] = st.column_config.ProgressColumn(
                "Probability", format="%.3f", min_value=0.0, max_value=1.0)
        st.dataframe(df, use_container_width=True, hide_index=True,
                      column_config=col_config, height=600)


# ─── Tab: About ─────────────────────────────────────────────────────────── #

with tab_about:
    st.markdown("### About ctvlm")
    st.markdown(
        "Deterministic agentic radiology report generator for CT abdomen-pelvis. "
        "Linear probe over a 3200-d concatenated dual-encoder feature "
        "(Merlin 2048-d + Pillar-0 1152-d, L2-normed), with Platt scaling, "
        "Youden-J thresholds, contrast-phase-aware calibration, and a "
        "tool-router + template renderer that produces auditable structured "
        "FINDINGS sentences."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Findings", "220", help="Unique canonical findings (out of 225 RATE questions)")
    c2.metric("Val micro F1", "0.529", help="On 5,082-case held-out val split (CE)")
    c3.metric("Cold latency", "~5 s", help="Per case on a 24 GB GPU")

    st.markdown("### Documentation")
    docs = [
        ("01 — Overview", "docs/01_OVERVIEW.md", "Architecture + component map"),
        ("02 — Installation", "docs/02_INSTALLATION.md", "Hardware, env vars, FS layout"),
        ("03 — Python API", "docs/03_PYTHON_API.md", "generate_report() contract"),
        ("04 — Data pipeline", "docs/04_DATA_PIPELINE.md", "DICOM → report stages"),
        ("05 — Orthanc integration", "docs/05_ORTHANC_INTEGRATION.md", "Receiver hook + DICOM tags"),
        ("06 — GPU broker", "docs/06_GPU_BROKER_INTEGRATION.md", "Worker pattern + memory"),
        ("07 — Calibration", "docs/07_CALIBRATION.md", "Platt + Youden-J + contrast phase"),
        ("08 — Findings taxonomy", "docs/08_FINDINGS_TAXONOMY.md", "All 220 canonicals + recipes"),
        ("09 — Grad-CAM", "docs/09_GRAD_CAM.md", "Heatmap generation + NiiVue"),
        ("10 — Performance", "docs/10_PERFORMANCE.md", "Full eval numbers"),
        ("11 — Limitations", "docs/11_LIMITATIONS.md", "Known FP/FN + regulatory"),
        ("12 — Operations", "docs/12_OPERATIONS.md", "Logs, metrics, runbook"),
        ("13 — Model weights", "docs/13_MODEL_WEIGHTS.md", "Where weights live + offline load"),
    ]
    for title, path, desc in docs:
        st.markdown(f"- **[{title}](https://github.com/<owner>/ctvlm/blob/main/{path})** — {desc}")

    st.markdown("### Encoders")
    st.markdown(
        "- **Merlin** ([stanfordmimi/Merlin](https://huggingface.co/stanfordmimi/Merlin)) — "
        "3D I3D-inflated ResNet-152 abdomen-CT image-text encoder, MIT license  \n"
        "- **Pillar-0** ([YalaLab/Pillar0-AbdomenCT](https://huggingface.co/YalaLab/Pillar0-AbdomenCT)) — "
        "Atlas abdomen-CT encoder, permissive license"
    )

    st.warning(
        "⚠️ **Research use only — NOT FDA-cleared / NOT CE-marked.** "
        "Do not use as a sole basis for clinical decisions. Always reviewed by "
        "a qualified radiologist."
    )


# ──────────────────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="ctvlm-footer">
    Built on Merlin + Pillar-0 CT encoders, calibrated against RATE-225 labels.
    MIT license. Research use only — NOT FDA-cleared.
    <a href="https://github.com/&lt;owner&gt;/ctvlm">GitHub</a> ·
    <a href="https://github.com/&lt;owner&gt;/ctvlm/blob/main/docs/11_LIMITATIONS.md">Limitations</a> ·
    <a href="https://github.com/&lt;owner&gt;/ctvlm/blob/main/LICENSE">License</a>
</div>
""", unsafe_allow_html=True)
