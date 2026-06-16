"""LLM call site — MedGemma-4B + existing text-only LoRA.

Extracted recipe from `scripts/22_train_hybrid.py:340` (the proven 0.83-recall
pattern). Greedy decoding, max_new_tokens=160, no chat template.

The model + tokenizer are cached in a module-level singleton so the dashboard
doesn't reload them on every render.

Layout of the prompt the LLM sees:

    Findings: <organ>: <sentence>; <sentence>; ...
              <organ>: <sentence>.
              ...

    Write the FINDINGS section of the abdominal CT report.

The structured Findings: section comes from `assemble_summary()`. The LLM's job
is purely to rewrite it as natural prose — it doesn't invent findings.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import torch

from src.agent.schema import StructuredFinding
from src.config import paths


DEFAULT_BASE = "google/medgemma-4b-it"
DEFAULT_LORA = paths.checkpoints_dir / "hybrid_medgemma-4b-it_text" / "lora"
INSTRUCTION = "\nWrite the FINDINGS section of the abdominal CT report.\n"


_model_lock = threading.Lock()
_state: dict = {}


def _load(base: str = DEFAULT_BASE, lora_dir: Path = DEFAULT_LORA, device: str = "cuda"):
    """Load (or return cached) MedGemma + LoRA adapter."""
    with _model_lock:
        if "model" in _state:
            return _state["model"], _state["tok"]
        os.environ.setdefault("HF_HOME", "/mnt/e/ctvlm/hf_cache")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(base)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        # The prior project loaded with 4-bit quantisation; we mirror that to
        # fit comfortably alongside Pillar-0/Merlin extraction memory pressure.
        try:
            from transformers import BitsAndBytesConfig
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                base, quantization_config=bnb, device_map={"": device})
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(
                base, torch_dtype=torch.bfloat16).to(device)
        # Apply LoRA if present. NOTE: do NOT merge_and_unload — with the
        # transformers 5.x + 4-bit base combination, merging silently breaks the
        # adapter weights and the model regresses to base. Keep PeftModel live.
        if lora_dir and Path(lora_dir).exists():
            try:
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, str(lora_dir))
                print(f"  [render] LoRA adapter loaded: {lora_dir}")
            except Exception as e:
                print(f"  [render] LoRA load failed; using base only: {e}")
        model.eval()
        _state["model"] = model
        _state["tok"] = tok
        return model, tok


# Per-category positive cap. The probe over-fires on co-occurring intra-organ
# findings (pancreatic cluster: IPMN + atrophy + duct dilation; vascular cluster:
# aortic + coronary atherosclerosis + valve calc + coronary artery calc, etc.).
# Cap at K → keep the K highest-probability findings per category, drop the rest.
# Lifted only when a category genuinely supports multiple findings per study
# (Liver/GI/MSK/Device); kept low (1-2) on tight-cluster organs.
#
# Default cap if a category isn't listed = 3.
CATEGORY_CAPS: dict[str, int] = {
    "Visible Thoracic": 3,        # pleural / cardiac / lower-lung cluster
    "Pancreas":         2,        # IPMN / atrophy / duct cluster
    "Spleen":           2,
    "Gallbladder":      2,
    "Biliary Tree":     2,
    "Adrenal gland":    2,
    "Great Vessel":     2,        # aortic / coronary cluster
    "Peritoneum":       2,
    "Retroperitoneum":  2,
    "Multi Organs":     1,        # the catch-all bucket
    "Liver":            4,        # legitimately can have many (cysts + steatosis + ...)
    "Genitourinary":    4,        # kidneys + bladder + ureters
    "Gastrointestinal": 4,
    "Musculoskeletal":  4,
    "Female pelvis":    3,
    "Male Pelvis":      3,
    "Device":           5,        # multiple devices common in ICU studies
}
DEFAULT_CATEGORY_CAP = 3


# Co-requirements: catch-all findings that need at least one supporting specific
# positive in order to be plausibly real. If `corequires & positives == empty`,
# suppress the catch-all. Different from SUBSUMPTION — that one trims a generic
# when a specific is also positive (both true). This one suppresses a generic
# when NO specific corroborates it (likely model hallucination).
COREQUIRES: dict[str, set[str]] = {
    # `metastatic_disease` is a Multi-Organs catch-all the probe over-fires when
    # any single high-prob lesion sits in an unusual organ. Require a concrete
    # malignancy or carcinomatosis pattern to back it up.
    "metastatic_disease": {
        "hepatic_metastases", "peritoneal_carcinomatosis",
        "hepatocellular_carcinoma", "cholangiocarcinoma",
        "renal_cell_carcinoma", "wilms_tumor",
        "colonic_carcinoma", "rectal_carcinoma", "gastric_carcinoma",
        "duodenal_carcinoma", "esophageal_carcinoma",
        "ductal_pancreatic_carcinoma", "neuroendocrine_tumor",
        "ovarian_cancer", "cervical_mass", "vaginal_cancer",
        "prostate_cancer", "bladder_transitional_cell_carcinoma",
        "gallbladder_cancer", "gastrointestinal_lymphoma",
        "hepatic_lymphoma", "splenic_lymphoma", "renal_lymphoma",
        # `lung_mass`, `osteolytic_lesion`, `osteosclerotic_lesion` removed —
        # they're benign-or-malignant on their own and the probe surfaces them
        # too readily; letting them support metastatic_disease compounds FPs.
    },
}


# Subsumption: when both a specific finding AND a generic one are positive,
# suppress the generic. Map = generic_finding -> set of specific finding names
# that "cover" it semantically.
SUBSUMPTION_RULES: dict[str, set[str]] = {
    "fracture": {"spinal_fracture", "rib_fracture",
                 "femoral_fracture", "pelvic_girdle_fracture"},
    # `renal_hypodensity` removed: it's NOT a generic version of "cyst" — it's a
    # separate finding that GT labels independently (could be cyst, mass, scar,
    # infarct). Subsuming it costs ~900+ FNs on the 5k val split.
    "hepatic_lesion": {"hepatic_cyst", "hepatic_hemangioma", "hepatic_adenoma",
                       "hepatic_metastases", "hepatocellular_carcinoma",
                       "hepatic_focal_nodular_hyperplasia"},
    "hepatic_mass": {"hepatocellular_carcinoma", "hepatic_metastases",
                     "fibrolamellar_carcinoma", "cholangiocarcinoma",
                     "hepatic_adenoma", "hepatic_lymphoma"},
    "adrenal_mass": {"adrenal_adenoma", "adrenal_myelolipoma",
                     "pheochromocytoma", "adrenal_hyperplasia"},
    "ovarian_tumor": {"ovarian_cancer", "ovarian_teratoma"},
    "pancreatic_tumor": {"ductal_pancreatic_carcinoma", "neuroendocrine_tumor",
                         "intraductal_papillary_mucinous_neoplasm",
                         "mucinous_cystic_neoplasm", "serous_cystic_neoplasm",
                         "solid_pseudopapillary_tumor"},
    "testicular_mass": {"testicular_infarct", "testicular_torsion"},
    "bowel_obstruction": {"small_bowel_obstruction", "large_bowel_obstruction"},
}


# Per-category "everything below threshold → say it's normal" templates.
# Keys are RATE organ categories (from CANONICAL_TO_CATEGORY).
NEGATIVE_STATEMENTS: dict[str, str] = {
    "Liver":            "Liver: normal in size and attenuation, no focal lesion.",
    "Spleen":           "Spleen: normal in size and attenuation.",
    "Pancreas":         "Pancreas: unremarkable, no ductal dilation.",
    "Gallbladder":      "Gallbladder: unremarkable, no wall thickening or stones.",
    "Biliary Tree":     "Biliary tree: no intrahepatic or extrahepatic ductal dilation.",
    "Adrenal gland":    "Adrenal glands: normal in size and contour.",
    "Genitourinary":    "Kidneys and bladder: unremarkable, no hydronephrosis or stones.",
    "Peritoneum":       "Peritoneum: no free fluid or free air.",
    "Great Vessel":     "Abdominal aorta: normal in caliber, no aneurysm or dissection.",
    "Retroperitoneum":  "Retroperitoneum: no lymphadenopathy.",
    "Visible Thoracic": "Visible thorax: lung bases clear, no pleural effusion.",
    "Gastrointestinal": "Bowel: no obstruction, wall thickening, or pneumatosis.",
}


# Stable anatomical order (top of chest → pelvis → musculoskeletal). Both
# RATE-category labels and per-Tier-A `recipe.organ` keys appear here.
ORGAN_ORDER: list[str] = [
    "Visible Thoracic",
    "liver", "Liver",
    "spleen", "Spleen",
    "pancreas", "Pancreas",
    "Biliary Tree",
    "gall_bladder", "Gallbladder",
    "Adrenal gland", "adrenal_gland_left", "adrenal_gland_right",
    "Genitourinary", "kidney_left", "kidney_right",
    "Gastrointestinal", "stomach", "duodenum", "colon",
    "Peritoneum",
    "Great Vessel", "aorta", "postcava",
    "Retroperitoneum",
    "Female pelvis", "Male Pelvis",
    "Musculoskeletal",
    "Multi Organs",
    "Device",
]


# Maps Tier A recipe `.organ` values (anatomical organ name) → RATE category
# (the higher-level grouping that NEGATIVE_STATEMENTS / CATEGORY_TO_CANONICALS
# use). Lets a Tier A sentence on `gall_bladder` suppress the
# Gallbladder-category negative statement.
ORGAN_TO_CATEGORY: dict[str, str] = {
    "liver":              "Liver",
    "spleen":             "Spleen",
    "pancreas":           "Pancreas",
    "gall_bladder":       "Gallbladder",
    "aorta":              "Great Vessel",
    "postcava":           "Great Vessel",
    "kidney_left":        "Genitourinary",
    "kidney_right":       "Genitourinary",
    "adrenal_gland_left": "Adrenal gland",
    "adrenal_gland_right":"Adrenal gland",
    "stomach":            "Gastrointestinal",
    "duodenum":           "Gastrointestinal",
    "colon":              "Gastrointestinal",
    "bladder":            "Genitourinary",
    "prostate":           "Male Pelvis",
}


def filter_positives(names: list[str],
                     probs: dict[str, float] | None = None) -> list[str]:
    """Apply COREQUIRES + SUBSUMPTION + per-category cap on a positive set.

    Order-preserving. Reusable by `assemble_summary` (rendering) and by the
    pipeline eval (counting TPs/FPs/FNs against ground-truth labels) so both
    paths see the same final set.

    Per-category cap is only applied if `probs` is provided (we need
    probabilities to pick top-K within a category).
    """
    name_set = set(names)
    keep: list[str] = []
    for n in names:
        # COREQUIRES: catch-all needs at least one specific supporter
        support = COREQUIRES.get(n)
        if support is not None and not (support & name_set):
            continue
        # SUBSUMPTION: generic dropped when a specific is also positive
        suppressors = SUBSUMPTION_RULES.get(n)
        if suppressors and (suppressors & name_set):
            continue
        keep.append(n)

    if probs is None:
        return keep

    # Per-category cap. Group surviving names by RATE category, sort by prob
    # desc, keep top CATEGORY_CAPS[cat]; drop the tail.
    from src.agent.canonical import CANONICAL_TO_CATEGORY
    by_cat: dict[str, list[str]] = {}
    no_cat: list[str] = []
    for n in keep:
        cat = CANONICAL_TO_CATEGORY.get(n)
        if cat is None:
            no_cat.append(n)
        else:
            by_cat.setdefault(cat, []).append(n)
    capped: set[str] = set(no_cat)
    for cat, items in by_cat.items():
        cap = CATEGORY_CAPS.get(cat, DEFAULT_CATEGORY_CAP)
        items_sorted = sorted(items, key=lambda x: -probs.get(x, 0.0))
        capped.update(items_sorted[:cap])
    return [n for n in keep if n in capped]


def _strip_redundant_organ_labels(sentences: list[str]) -> str:
    """If 2+ sentences in a group share the same 'Organ: ' prefix, drop it from #2+."""
    if not sentences:
        return ""
    first = sentences[0]
    if ":" not in first:
        return "; ".join(sentences)
    prefix = first.split(":", 1)[0] + ":"
    cleaned = [first]
    for s in sentences[1:]:
        cleaned.append(s[len(prefix):].strip() if s.startswith(prefix) else s)
    return "; ".join(cleaned)


def assemble_summary(structured: list[StructuredFinding],
                     probs: dict[str, float] | None = None,
                     *,
                     negative_threshold: float = 0.40) -> str:
    """Group sentences by organ in a stable anatomical order.

    - Subsumes generic findings when specific ones are positive (rules above).
    - Emits per-organ "unremarkable" statements when ALL canonicals in that
      category are below `negative_threshold` and the category has no positive
      sentences. Requires `probs` (the full 220-d probability dict).
    - Strips duplicate "Organ: " prefixes in multi-sentence groups.
    """
    # Step 1 — subsumption + corequires (shared with eval / pipeline)
    keep = set(filter_positives([sf.finding for sf in structured]))
    filtered = [sf for sf in structured if sf.finding in keep]

    # Step 2 — group by organ key, canonicalised to the RATE category so Tier A
    # recipes ("gall_bladder") and Tier B fallbacks ("Gallbladder" category)
    # share a bucket.
    by_organ: dict[str, list[str]] = {}
    for sf in filtered:
        raw_key = (sf.organ
                   or (sf.fields.get("organ_label_key") if sf.fields else None)
                   or "general")
        key = ORGAN_TO_CATEGORY.get(raw_key, raw_key)
        by_organ.setdefault(key, []).append(sf.sentence.rstrip(".").strip())

    # Step 3 — negative statements per category when no positive sentence and
    # every canonical in that category is below the threshold
    if probs is not None:
        from src.agent.canonical import CATEGORY_TO_CANONICALS
        positive_categories = {
            ORGAN_TO_CATEGORY.get(k, k) for k in by_organ
        }
        for category, neg_stmt in NEGATIVE_STATEMENTS.items():
            if category in positive_categories:
                continue
            canonicals = CATEGORY_TO_CANONICALS.get(category, set())
            if not canonicals:
                continue
            max_prob = max((probs.get(c, 0.0) for c in canonicals), default=0.0)
            if max_prob < negative_threshold:
                by_organ.setdefault(category, []).append(
                    neg_stmt.rstrip(".").strip())

    # Step 4 — assemble in canonical anatomical order
    parts = ["Findings:"]
    seen: set[str] = set()
    for key in ORGAN_ORDER:
        if key in by_organ and key not in seen:
            joined = _strip_redundant_organ_labels(by_organ[key])
            if joined:
                parts.append("  " + (joined if joined.endswith(".") else joined + "."))
            seen.add(key)
    for key, sentences in by_organ.items():
        if key in seen:
            continue
        joined = _strip_redundant_organ_labels(sentences)
        if joined:
            parts.append("  " + (joined if joined.endswith(".") else joined + "."))
    return "\n".join(parts)


def render_prose(summary_text: str, *, max_new_tokens: int = 160) -> str:
    """The one LLM call. Greedy decoding, no chat template, deterministic."""
    model, tok = _load()
    prompt = summary_text + INSTRUCTION
    input_ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=max_new_tokens,
                              do_sample=False, pad_token_id=tok.eos_token_id)
    full = tok.decode(out[0], skip_special_tokens=True)
    # Strip the prompt from the model's echo
    if full.startswith(prompt):
        return full[len(prompt):].strip()
    return full.strip()
