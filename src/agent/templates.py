"""Per-finding sentence templates.

Each template_key maps to a format string that consumes a StructuredFinding's
`.fields` dict. The renderer also has access to `.probability`, `.organ`, and
`.finding` (canonical name) via the wrapper below.

Three tiers:
  A — hand-written, finding-specific phrasing with measurements
  B — generic organ-anchored sentence ("{organ}: {phrase} (axial slice X, p=Y)")
  C — generic fallback ("{phrase} (p=Y)")

Sentence strings deliberately mirror the prose patterns the prior project's
MedGemma-4B LoRA was fine-tuned on (see `reports/samples_medgemma-4b-it.md`).
"""
from __future__ import annotations

import re

from src.agent.schema import StructuredFinding


# --------------------------------------------------------------------------- #
# Tier A — hand-written per-finding templates (Phase 1 ships ~11 of these)
# --------------------------------------------------------------------------- #

SENTENCES: dict[str, str] = {
    # --- liver ---
    "hepatic_steatosis":
        "Liver: diffusely decreased attenuation "
        "(mean {liver_mean_hu:.0f} HU, liver-spleen {liver_minus_spleen_hu:+.0f}) "
        "compatible with hepatic steatosis.",
    "hepatic_cyst":
        "Liver: well-defined hypodense lesion"
        "{segment_clause}{size_clause}{slice_clause}{hu_clause}"
        " consistent with simple hepatic cyst.",
    "hepatic_lesion":
        "Liver: focal hypodense lesion{segment_clause}{size_clause}{slice_clause}{hu_clause}.",

    # --- spleen ---
    "splenomegaly":
        "Spleen: enlarged "
        "(volume {volume_ml:.0f} mL, craniocaudal {extent_cc_mm:.0f} mm) "
        "compatible with splenomegaly.",

    # --- kidney ---
    "renal_cyst":
        "Kidneys: simple cyst{side_clause}{size_clause}{slice_clause} "
        "(mean HU {mean_hu:.0f}).",

    # --- vascular ---
    "aortic_atherosclerosis":
        "Abdominal aorta: mural calcifications throughout, "
        "compatible with aortic atherosclerosis.",

    # --- peritoneum ---
    "ascites":
        "Peritoneum: free intraperitoneal fluid {slice_clause}, "
        "compatible with ascites.",

    # --- thoracic ---
    "pleural_effusion":
        "Thorax (visible): {side} pleural effusion at the lung base"
        "{slice_clause}.",

    # --- bowel ---
    "appendicitis":
        "Right lower quadrant: enlarged tubular blind-ending structure"
        "{slice_clause} concerning for acute appendicitis.",

    # --- nodal ---
    "lymphadenopathy":
        "Retroperitoneum: enlarged lymph nodes{slice_clause}.",

    # --- pancreas ---
    "pancreatitis":
        "Pancreas: peripancreatic stranding{slice_clause}, "
        "compatible with acute pancreatitis.",

    # --- cardiac ---
    "cardiomegaly":
        "Heart: enlarged cardiac silhouette{slice_clause}, "
        "compatible with cardiomegaly.",

    # --- thoracic atelectasis ---
    "atelectasis":
        "Lower thorax: dependent atelectasis at the lung bases{slice_clause}.",

    # --- skeletal ---
    "osteopenia":
        "Musculoskeletal: generalized osteopenia of the visualised skeleton.",

    # --- gallbladder ---
    "gallbladder_stones":
        "Gallbladder: discrete radiopaque calculi{slice_clause}{size_clause}.",

    # --- renal collecting system ---
    "hydronephrosis":
        "Kidneys: hydronephrosis{side_clause}{slice_clause}.",

    # Tier B & C generic templates. Note: probability annotations are NOT in the
    # LLM input (the MedGemma LoRA wasn't trained on `(p=0.88)`-style markup).
    # The probability is preserved in StructuredFinding.probability for the
    # dashboard Trace tab.
    "_generic_organ":
        "{organ_label}: {finding_humanised}{slice_clause}.",
    "_generic_organ_no_slice":
        "{organ_label}: {finding_humanised}.",
    "_generic_fallback":
        "{finding_humanised}{slice_clause}.",
}


# --------------------------------------------------------------------------- #
# Organ-label humanisation (mirrors src/llm/structured_summary.py:_ORGAN_LABEL)
# --------------------------------------------------------------------------- #

ORGAN_LABEL: dict[str, str] = {
    "liver":                "Liver",
    "spleen":               "Spleen",
    "kidney_left":          "Left kidney",
    "kidney_right":         "Right kidney",
    "pancreas":             "Pancreas",
    "gall_bladder":         "Gallbladder",
    "adrenal_gland_left":   "Left adrenal",
    "adrenal_gland_right":  "Right adrenal",
    "bladder":              "Bladder",
    "prostate":             "Prostate",
    "stomach":              "Stomach",
    "esophagus":            "Esophagus",
    "duodenum":             "Duodenum",
    "colon":                "Colon",
    "aorta":                "Aorta",
    "postcava":             "IVC",
    # category-from-RATE keys (broader)
    "Liver":                "Liver",
    "Spleen":               "Spleen",
    "Pancreas":             "Pancreas",
    "Gallbladder":          "Gallbladder",
    "Biliary Tree":         "Biliary tree",
    "Genitourinary":        "Genitourinary",
    "Adrenal gland":        "Adrenal",
    "Gastrointestinal":     "Gastrointestinal",
    "Peritoneum":           "Peritoneum",
    "Great Vessel":         "Great vessels",
    "Retroperitoneum":      "Retroperitoneum",
    "Multi Organs":         "Multi-organ",
    "Male Pelvis":          "Male pelvis",
    "Female pelvis":        "Female pelvis",
    "Musculoskeletal":      "Musculoskeletal",
    "Visible Thoracic":     "Thorax (visible)",
    "Device":               "Devices",
}


def organ_label_for(organ_or_category: str | None) -> str:
    if not organ_or_category:
        return ""
    return ORGAN_LABEL.get(organ_or_category, organ_or_category.replace("_", " ").title())


def humanise_finding(canonical: str) -> str:
    """`hepatic_cyst` → `hepatic cyst`; `prostate_cancer` → `prostate cancer`."""
    return canonical.replace("_", " ")


# --------------------------------------------------------------------------- #
# The renderer
# --------------------------------------------------------------------------- #

def _safe_fmt(template: str, ctx: dict) -> str:
    """str.format that tolerates missing keys (substitutes "")."""
    class _Default(dict):
        def __missing__(self, key):
            return ""
    # Pre-strip optional clauses where the supporting field is missing
    return template.format_map(_Default(ctx))


def render_finding(sf: StructuredFinding) -> str:
    """Pick a template by tier/key, fill from sf.fields, normalise whitespace."""
    key = sf.fields.get("template_key", "_generic_fallback")
    template = SENTENCES.get(key)
    if template is None:
        template = SENTENCES["_generic_fallback"]

    ctx = dict(sf.fields)
    ctx["probability"] = sf.probability
    ctx["finding"] = sf.finding
    ctx["finding_humanised"] = humanise_finding(sf.finding)
    ctx["organ_label"] = organ_label_for(sf.fields.get("organ_label_key", sf.organ))

    # Optional clauses — only appear if their inputs are present
    seg = sf.fields.get("segment_roman")
    ctx["segment_clause"] = f" in segment {seg}" if seg else ""
    sz = sf.fields.get("extent_mm")
    if sz and isinstance(sz, list) and len(sz) >= 2:
        # express in cm if > 1 cm, else mm
        a, b = sorted(sz)[:2]
        if max(a, b) >= 10:
            ctx["size_clause"] = f" measuring {a/10:.1f} × {b/10:.1f} cm"
        else:
            ctx["size_clause"] = f" measuring {a:.0f} × {b:.0f} mm"
    else:
        ctx["size_clause"] = ""
    ax = sf.fields.get("axial_slice")
    ctx["slice_clause"] = f" (axial slice {ax})" if ax else ""
    hu = sf.fields.get("mean_hu")
    ctx["hu_clause"] = f", mean HU {hu:.0f}" if hu is not None else ""
    ctx["side"] = sf.fields.get("side", "")
    ctx["side_clause"] = (
        f" in the {sf.fields['side']} kidney" if sf.fields.get("side") else ""
    )

    out = _safe_fmt(template, ctx)
    # tidy: collapse double spaces, fix " ." or " ,"
    out = re.sub(r"\s+", " ", out)
    out = re.sub(r"\s+([.,;])", r"\1", out)
    return out.strip()
