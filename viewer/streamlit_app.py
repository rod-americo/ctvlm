"""Streamlit QA dashboard with NiiVue viewer for CT cases, masks, labels, R-GAT preds.

Eye-on-the-prize: validate the upstream pipeline (labels + masks + organ pool) by
actually looking at the cases. The CT viewer is a NiiVue WebGL iframe served from a
sidecar process on :8502 (scripts/ct_files_server.py), so slice scrolling is instant
client-side.

Components:
  - This file: case browser, finding filter, mask multi-select, report panel,
    label/prediction table, label-override capture (data/finding_overrides.csv).
  - scripts/ct_files_server.py: NiiVue viewer page + CT/mask file streams with CORS.
  - First-run helper converts reports_final.xlsx → reports_final.parquet (~10x faster
    on cold start).

Run BOTH (in two shells or backgrounded):
    python scripts/ct_files_server.py &
    streamlit run scripts/28_validation_dashboard.py \\
        --server.address 0.0.0.0 --server.port 8501

Then open http://localhost:8501.
"""
from __future__ import annotations

import csv
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import paths                           # noqa: E402
from src.data import merlinplus as mp                  # noqa: E402
from src.graph.anatomy_prior import NODE_ORGANS        # noqa: E402
from src.ontology.finding_extractor import SYNONYMS, FINDING_NAMES  # noqa: E402

LABELS_CSV = REPO_ROOT / "data" / "finding_labels.csv"
OVERRIDES_CSV = REPO_ROOT / "data" / "finding_overrides.csv"
REPORTS_XLSX = paths.merlin_root / "merlinabdominalctdataset" / "reports_final.xlsx"
REPORTS_PARQUET = REPO_ROOT / "data" / "reports_final.parquet"
CHECKPOINTS_DIR = paths.checkpoints_dir
PILLAR0_NODES = paths.work_root / "pillar0_nodes_attn"
SUBSET_DIRS = {
    "all MerlinPlus cases": None,
    "2k benchmark subset": paths.work_root / "pillar0_nodes_attn",
}

FILE_SERVER_PORT = 8502

# Mirrored from scripts/ct_files_server.py — kept here so the dashboard can label the
# mask chips with the same colors NiiVue assigns inside the iframe.
MASK_COLORMAPS = ["red", "green", "blue", "yellow", "cyan", "violet",
                  "warm", "winter", "spring", "summer", "autumn", "cool"]


# --------------------------------------------------------------------------- #
# Data loaders
# --------------------------------------------------------------------------- #

@st.cache_data(show_spinner=False)
def load_labels() -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(LABELS_CSV)
    findings = [c for c in df.columns if c != "study_id"]
    df = df.set_index("study_id")
    return df, findings


def _ensure_reports_parquet() -> Path:
    """Cache the 25k-row reports xlsx as parquet (~50ms vs ~15s for xlsx)."""
    xlsx_mt = REPORTS_XLSX.stat().st_mtime
    if REPORTS_PARQUET.exists() and REPORTS_PARQUET.stat().st_mtime >= xlsx_mt:
        return REPORTS_PARQUET
    REPORTS_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_excel(REPORTS_XLSX)
    df.to_parquet(REPORTS_PARQUET, index=False)
    return REPORTS_PARQUET


@st.cache_data(show_spinner=True)
def load_reports() -> pd.DataFrame:
    p = _ensure_reports_parquet()
    df = pd.read_parquet(p)
    sid_col = next(c for c in df.columns if c.lower().replace(" ", "") in ("studyid", "study_id"))
    df = df.rename(columns={sid_col: "study_id"}).set_index("study_id")
    return df


@st.cache_data(show_spinner=False)
def load_subset_ids(subset_key: str) -> list[str]:
    if subset_key == "all MerlinPlus cases":
        return sorted(mp.list_cases())
    p = SUBSET_DIRS[subset_key]
    if p is None or not p.is_dir():
        return sorted(mp.list_cases())
    return sorted(s.stem for s in p.glob("*.npz"))


@st.cache_data(show_spinner=False, max_entries=64)
def available_masks(study_id: str) -> list[str]:
    return mp.available_classes(study_id)


@st.cache_resource(show_spinner=False)
def find_checkpoints() -> dict[str, Path]:
    return {p.name: p for p in CHECKPOINTS_DIR.glob("*.pt")} if CHECKPOINTS_DIR.is_dir() else {}


@st.cache_resource(show_spinner=True)
def load_rgat_model(ckpt_path_str: str):
    import torch
    ck = torch.load(ckpt_path_str, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    findings = ck.get("findings", FINDING_NAMES)
    is_attn = any(k.startswith("pool.") for k in sd)
    in_dim = None
    for k, v in sd.items():
        if not is_attn and k.endswith("weight") and v.ndim == 2:
            in_dim = v.shape[1]; break
        if is_attn and "gnn." in k and k.endswith("weight") and v.ndim == 2:
            in_dim = v.shape[1]; break
    from src.graph.gnn import OrganRGAT
    if is_attn:
        import torch.nn as nn
        from src.embeddings.attention_pool import OrganAttentionPool
        n_organs = sd["pool.queries"].shape[0]
        pool_dim = sd["pool.queries"].shape[1]

        class AttnPoolRGAT(nn.Module):
            def __init__(self):
                super().__init__()
                heads = 4 if pool_dim % 4 == 0 else 8
                self.pool = OrganAttentionPool(n_organs, dim=pool_dim, num_heads=heads)
                self.gnn = OrganRGAT(in_dim=in_dim or (pool_dim + 768),
                                     num_relations=3, num_classes=len(findings))

            def forward(self, x_tokens, organ_idx, mask, x_pool, edge_index, edge_type, batch):
                a = self.pool(x_tokens, organ_idx, key_padding_mask=~mask)
                x = torch.cat([x_pool, a], dim=-1)
                return self.gnn(x, edge_index, edge_type, batch)

        m = AttnPoolRGAT()
    else:
        m = OrganRGAT(in_dim=in_dim, num_relations=3, num_classes=len(findings))
    m.load_state_dict(sd, strict=False)
    m.eval()
    return m, findings, in_dim, ("AttnPoolRGAT" if is_attn else "OrganRGAT")


def _try_predict(model, arch: str, study_id: str, findings: list[str]):
    import torch
    from src.graph.study_graph import build_study_graph
    if arch == "OrganRGAT":
        for nd in (PILLAR0_NODES, paths.work_root / "pillar0_nodes"):
            if (nd / f"{study_id}.npz").exists():
                sg = build_study_graph(study_id, nd, ops=("mean", "max"))
                break
        else:
            return None
        if sg is None or sg.x.shape[0] == 0: return None
        with torch.no_grad():
            p = torch.sigmoid(model(
                torch.from_numpy(sg.x).float(),
                torch.from_numpy(sg.edge_index).long(),
                torch.from_numpy(sg.edge_type).long(),
                torch.zeros(sg.x.shape[0], dtype=torch.long),
            )).cpu().numpy().ravel()
        return {f: float(p[i]) for i, f in enumerate(findings)}
    if arch == "AttnPoolRGAT":
        npz = PILLAR0_NODES / f"{study_id}.npz"
        if not npz.exists(): return None
        z = np.load(npz, allow_pickle=True)
        if "organ_tokens" not in z.files: return None
        organ_pool = z["organ"].item(); organ_toks = z["organ_tokens"].item()
        organ_sizes = z["organ_sizes"].item() if "organ_sizes" in z.files else {}
        nodes = [o for o in NODE_ORGANS if o in organ_pool and o in organ_toks
                 and "mean" in organ_pool[o] and "max" in organ_pool[o]]
        if not nodes: return None
        K = next(iter(organ_toks.values())).shape[0]
        x_pool = np.stack([np.concatenate([organ_pool[o]["mean"].astype(np.float32),
                                           organ_pool[o]["max"].astype(np.float32)]) for o in nodes])
        x_tokens = np.stack([organ_toks[o].astype(np.float32) for o in nodes])
        organ_idx = np.array([NODE_ORGANS.index(o) for o in nodes], np.int64)
        mask = np.zeros((len(nodes), K), bool)
        for i, o in enumerate(nodes):
            n_real = min(int(organ_sizes.get(o, K)), K); mask[i, :n_real] = True
        from src.graph.anatomy_prior import EDGE_TYPES, prior_edges
        idx = {o: i for i, o in enumerate(nodes)}
        ei = []; et = []
        for a, b, t in prior_edges():
            if a in idx and b in idx:
                ei.append((idx[a], idx[b])); et.append(EDGE_TYPES.index(t))
        edge_index = np.array(ei, np.int64).T if ei else np.zeros((2, 0), np.int64)
        edge_type = np.array(et, np.int64)
        with torch.no_grad():
            p = torch.sigmoid(model(
                torch.from_numpy(x_tokens).float(), torch.from_numpy(organ_idx).long(),
                torch.from_numpy(mask), torch.from_numpy(x_pool).float(),
                torch.from_numpy(edge_index).long(), torch.from_numpy(edge_type).long(),
                torch.zeros(len(nodes), dtype=torch.long),
            )).cpu().numpy().ravel()
        return {f: float(p[i]) for i, f in enumerate(findings)}
    return None


# --------------------------------------------------------------------------- #
# Overrides
# --------------------------------------------------------------------------- #

OVERRIDE_FIELDS = ["study_id", "finding", "original_label", "corrected_label",
                   "timestamp", "reviewer", "note"]


@st.cache_data(show_spinner=False, ttl=5)
def load_overrides() -> dict[tuple[str, str], dict]:
    if not OVERRIDES_CSV.exists():
        return {}
    out: dict[tuple[str, str], dict] = {}
    with OVERRIDES_CSV.open() as f:
        for row in csv.DictReader(f):
            key = (row["study_id"], row["finding"])
            prev = out.get(key)
            if prev is None or row["timestamp"] > prev["timestamp"]:
                out[key] = row
    return out


def append_override(study_id, finding, original, corrected, reviewer, note=""):
    OVERRIDES_CSV.parent.mkdir(parents=True, exist_ok=True)
    new = not OVERRIDES_CSV.exists()
    with OVERRIDES_CSV.open("a") as f:
        w = csv.DictWriter(f, fieldnames=OVERRIDE_FIELDS)
        if new: w.writeheader()
        w.writerow({"study_id": study_id, "finding": finding,
                    "original_label": int(original), "corrected_label": int(corrected),
                    "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
                    "reviewer": reviewer, "note": note})
    load_overrides.clear()


# --------------------------------------------------------------------------- #
# Report text highlighting (regex + simple negation heuristic)
# --------------------------------------------------------------------------- #

NEG_RE = re.compile(r"\b(no|without|negative for|no evidence of|free of|resolved|"
                    r"ruled out|absence of|none)\b", re.I)


@st.cache_data(show_spinner=False)
def compile_synonym_pattern() -> re.Pattern:
    parts = sorted({s for forms in SYNONYMS.values() for s in forms}, key=len, reverse=True)
    return re.compile(r"\b(" + "|".join(re.escape(p) for p in parts) + r")\b", re.I)


def highlight_report(text: str) -> str:
    pat = compile_synonym_pattern()
    out, last = [], 0
    for m in pat.finditer(text):
        out.append(text[last:m.start()])
        negated = bool(NEG_RE.search(text[max(0, m.start() - 40):m.start()]))
        bg = "#fde68a" if negated else "#bbf7d0"
        out.append(f"<mark style='background:{bg};padding:0 2px;border-radius:2px'>"
                   f"{m.group(0)}</mark>")
        last = m.end()
    out.append(text[last:])
    return "".join(out).replace("\n", "<br>")


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #

def init_state():
    ss = st.session_state
    ss.setdefault("study_idx", 0)
    ss.setdefault("subset_key", "2k benchmark subset")
    ss.setdefault("filter_finding", "(none)")
    ss.setdefault("view_mode", "MPR")
    ss.setdefault("mask_choices", ["liver", "spleen", "pancreas",
                                    "kidney_left", "kidney_right", "gall_bladder"])
    ss.setdefault("ckpt", "")
    ss.setdefault("reviewer", "")
    ss.setdefault("pending_overrides", {})
    ss.setdefault("heat_finding", "(none)")
    ss.setdefault("heat_encoders", ["merlin"])
    ss.setdefault("heat_opacity", 0.55)


def viewer_iframe_url(study_id: str, masks: list[str], view_mode: str,
                       heat_finding: str = "(none)",
                       heat_encoders: list[str] | None = None,
                       heat_opacity: float = 0.55) -> str:
    slice_id = {"MPR": 3, "axial": 0, "sagittal": 1, "coronal": 2}[view_mode]
    mask_qs = ",".join(masks)
    heat_qs = ""
    if heat_finding and heat_finding != "(none)" and heat_encoders:
        heat_qs = "&heatmap=" + ",".join(f"{e}:{heat_finding}" for e in heat_encoders)
        heat_qs += f"&heatopacity={heat_opacity:.2f}"
    # `localhost` matches the host the user typed to reach the streamlit page, so the
    # iframe resolves through the same WSL2 port-forward chain (8501 -> 8502).
    return (f"http://localhost:{FILE_SERVER_PORT}/viewer.html?"
            f"sid={study_id}&masks={mask_qs}&slice={slice_id}{heat_qs}")


def app():
    st.set_page_config(page_title="CT-VLM QA", layout="wide")
    st.markdown("""<style>
        .block-container {padding-top: 0.5rem; padding-bottom: 0.5rem; max-width: none;}
        .stRadio > label, .stCheckbox > label {font-size: 0.85rem;}
        iframe {border-radius: 6px;}
    </style>""", unsafe_allow_html=True)
    init_state()

    labels_df, findings = load_labels()
    reports_df = load_reports()
    overrides = load_overrides()
    checkpoints = find_checkpoints()

    # ===== Sidebar =====
    with st.sidebar:
        st.markdown("### Case")
        subset_key = st.selectbox("Subset", list(SUBSET_DIRS.keys()),
                                  index=list(SUBSET_DIRS.keys()).index(st.session_state.subset_key))
        if subset_key != st.session_state.subset_key:
            st.session_state.subset_key = subset_key
            st.session_state.study_idx = 0
        ids = [s for s in load_subset_ids(subset_key) if s in labels_df.index]
        ff = st.selectbox("Filter: cases with finding present",
                          ["(none)"] + findings,
                          index=(["(none)"] + findings).index(st.session_state.filter_finding))
        st.session_state.filter_finding = ff
        if ff != "(none)":
            ids = [s for s in ids if int(labels_df.loc[s, ff]) == 1]
        if not ids:
            st.error("No cases match the current filter."); return

        cp, cn = st.columns(2)
        if cp.button("◀ prev"):
            st.session_state.study_idx = (st.session_state.study_idx - 1) % len(ids)
        if cn.button("next ▶"):
            st.session_state.study_idx = (st.session_state.study_idx + 1) % len(ids)
        st.session_state.study_idx = min(st.session_state.study_idx, len(ids) - 1)

        sid_input = st.text_input(f"Study ID ({st.session_state.study_idx + 1} / {len(ids)})",
                                  value=ids[st.session_state.study_idx])
        if sid_input != ids[st.session_state.study_idx] and sid_input in ids:
            st.session_state.study_idx = ids.index(sid_input)
        study_id = ids[st.session_state.study_idx]

        st.markdown("---")
        st.markdown("### Viewer")
        st.session_state.view_mode = st.radio("Layout",
                                              ["MPR", "axial", "sagittal", "coronal"],
                                              index=["MPR", "axial", "sagittal",
                                                     "coronal"].index(st.session_state.view_mode),
                                              horizontal=True,
                                              help=("MPR = 3 planes at once. "
                                                    "Use mouse: scroll = slice, "
                                                    "drag = window/level."))
        avail = available_masks(study_id)
        defaults = [o for o in st.session_state.mask_choices if o in avail]
        # if user picks an organ that doesn't exist for this case, just drop it silently
        st.session_state.mask_choices = st.multiselect("Mask overlays", avail, default=defaults,
                                                       help=f"{len(avail)} MerlinPlus classes available")

        st.markdown("---")
        st.markdown("### CAM heatmap")
        # Findings the CAM probes are trained on (= negspaCy 50-finding taxonomy)
        heat_choice = st.selectbox("Heatmap finding", ["(none)"] + findings,
                                    index=(["(none)"] + findings).index(st.session_state.heat_finding)
                                          if st.session_state.heat_finding in (["(none)"] + findings) else 0,
                                    help=("Per-finding CAM (linear probe × pre-pool feature map). "
                                          "Cold cache: ~25s; warm cache: instant. Heatmaps cached at "
                                          "/mnt/e/ctvlm/heatmaps/<encoder>/<sid>/<finding>.nii.gz."))
        st.session_state.heat_finding = heat_choice
        st.session_state.heat_encoders = st.multiselect(
            "Encoders", ["merlin", "pillar0"],
            default=st.session_state.heat_encoders,
            help="Side-by-side overlay: hot=merlin (avgpool, exact), winter=pillar0 (max-pool, approximate).")
        st.session_state.heat_opacity = st.slider("Heatmap opacity", 0.10, 0.95,
                                                   st.session_state.heat_opacity, 0.05)

        st.markdown("---")
        st.markdown("### Predictions")
        names = list(checkpoints.keys())
        choice = st.selectbox("Checkpoint", ["(none)"] + names,
                              index=(0 if st.session_state.ckpt not in names
                                     else names.index(st.session_state.ckpt) + 1))
        st.session_state.ckpt = "" if choice == "(none)" else choice
        st.session_state.reviewer = st.text_input("Reviewer (for overrides)",
                                                  value=st.session_state.reviewer)

    # ===== Layout: viewer (60%) | side panel (40%) =====
    col_view, col_panel = st.columns([3, 2])

    with col_view:
        st.markdown(f"#### `{study_id}`")
        url = viewer_iframe_url(study_id,
                                st.session_state.mask_choices,
                                st.session_state.view_mode,
                                heat_finding=st.session_state.heat_finding,
                                heat_encoders=st.session_state.heat_encoders,
                                heat_opacity=st.session_state.heat_opacity)
        st.components.v1.iframe(url, height=720, scrolling=False)
        # mask legend
        if st.session_state.mask_choices:
            chips = []
            for i, o in enumerate(st.session_state.mask_choices):
                c = MASK_COLORMAPS[i % len(MASK_COLORMAPS)]
                chips.append(f"<span style='display:inline-block;background:#374151;"
                             f"color:#e5e7eb;border-radius:6px;padding:1px 8px;margin:2px;"
                             f"font-size:0.78rem'><b style='color:{c}'>●</b> {o}</span>")
            st.markdown(" ".join(chips), unsafe_allow_html=True)
        if (st.session_state.heat_finding and st.session_state.heat_finding != "(none)"
                and st.session_state.heat_encoders):
            heat_chips = []
            heat_palette = {"merlin": "#ef4444", "pillar0": "#3b82f6"}
            for e in st.session_state.heat_encoders:
                heat_chips.append(f"<span style='display:inline-block;background:#1f2937;"
                                  f"color:#e5e7eb;border-radius:6px;padding:1px 8px;margin:2px;"
                                  f"font-size:0.78rem;border:1px solid {heat_palette[e]}'>"
                                  f"<b style='color:{heat_palette[e]}'>▮</b> {e} CAM · "
                                  f"{st.session_state.heat_finding}</span>")
            st.markdown(" ".join(heat_chips), unsafe_allow_html=True)
        st.caption("NiiVue WebGL viewer. Scroll = slice; right-drag = window/level; "
                   "left-drag = pan in MPR. CAM heatmap first-hit ~25s, then instant.")

    with col_panel:
        # report
        st.markdown("#### Report — Findings")
        if study_id in reports_df.index:
            row = reports_df.loc[study_id]
            text = str(row["Findings"])
            split = row["Split"] if "Split" in reports_df.columns else "?"
            st.caption(f"Merlin split: **{split}**")
            st.markdown(f"<div style='max-height:220px;overflow-y:auto;font-size:0.84rem;"
                        f"padding:8px;border:1px solid #374151;border-radius:4px;"
                        f"background:#1f2937;color:#e5e7eb;'>{highlight_report(text)}</div>",
                        unsafe_allow_html=True)
        else:
            st.caption("(report not found)")

        preds = None
        if st.session_state.ckpt:
            try:
                model, ckpt_findings, _, arch = load_rgat_model(
                    str(checkpoints[st.session_state.ckpt]))
                preds = _try_predict(model, arch, study_id, ckpt_findings)
                if preds is None:
                    st.caption("(no cached node features for this case → no predictions)")
            except Exception as e:
                st.caption(f"prediction failed: {e}")

        st.markdown("#### Findings table")
        labs = labels_df.loc[study_id].to_dict()
        edits_now = st.session_state.pending_overrides
        show_all = st.checkbox("show all 50 findings", value=False)
        rows = []
        for f in findings:
            lab = int(labs.get(f, 0))
            ov = overrides.get((study_id, f))
            eff_lab = int(ov["corrected_label"]) if ov else lab
            p = preds.get(f) if preds else None
            if not show_all and eff_lab == 0 and (p is None or p < 0.2):
                continue
            rows.append((f, lab, eff_lab, p, ov is not None))

        if not rows:
            st.caption("All findings 0 (negspaCy) and no high-prob predictions. "
                       "Tick *show all 50 findings* for the full list.")
        for f, lab, eff_lab, p, has_ov in rows:
            c1, c2, c3, c4 = st.columns([3, 1, 1, 2])
            with c1:
                tag = " ✏️" if has_ov else ""
                st.markdown(f"**{f}**{tag}")
            with c2:
                color = "#16a34a" if eff_lab == 1 else "#9ca3af"
                st.markdown(f"<div style='color:{color};font-weight:700'>{eff_lab}</div>",
                            unsafe_allow_html=True)
            with c3:
                if p is None: st.markdown("—")
                else:
                    bg = "#16a34a" if p > 0.5 else "#9ca3af"
                    st.markdown(f"<span style='background:{bg};color:white;border-radius:3px;"
                                f"padding:1px 5px'>{p:.2f}</span>", unsafe_allow_html=True)
            with c4:
                key = (study_id, f); cur = edits_now.get(key, "keep")
                opt = st.selectbox(" ", ["keep", "force 0", "force 1"],
                                   index=["keep", "force 0", "force 1"].index(cur),
                                   key=f"ov_{f}", label_visibility="collapsed")
                edits_now[key] = opt
        st.session_state.pending_overrides = edits_now
        pend = {k: v for k, v in edits_now.items() if k[0] == study_id and v != "keep"}
        cols = st.columns([2, 1])
        with cols[0]:
            note = st.text_input("Note (optional)", value="", key="ov_note")
        with cols[1]:
            if st.button(f"💾 save {len(pend)} overrides", disabled=not pend):
                reviewer = st.session_state.reviewer or "(anon)"
                for (sid, f), choice in pend.items():
                    new_lab = 0 if choice == "force 0" else 1
                    append_override(sid, f, int(labs.get(f, 0)), new_lab, reviewer, note)
                for k in list(pend.keys()): edits_now[k] = "keep"
                st.session_state.pending_overrides = edits_now
                st.success(f"Saved {len(pend)} → {OVERRIDES_CSV.name}")
                st.rerun()
        if OVERRIDES_CSV.exists():
            n_tot = len(load_overrides())
            n_case = sum(1 for k in overrides if k[0] == study_id)
            st.caption(f"Overrides on file: **{n_tot}** total, **{n_case}** on this case.")


# Streamlit execs this file on every rerun; `app()` builds the page.
app()
