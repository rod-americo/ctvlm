"""Per-finding recipes — Tier A hand-written plus a generic-organ fallback that
auto-generates a Tier B/C recipe from the RATE canonical map's organ category.

Looking up a recipe for a canonical name:
    Recipe = lookup(canonical) returns either a hand-written entry from
    HAND_WRITTEN or an organ-anchored generic built from the category map.
"""
from __future__ import annotations

from src.agent.canonical import CANONICAL_TO_CATEGORY
from src.agent.schema import Recipe, ToolCall


# --------------------------------------------------------------------------- #
# Tier A — hand-written recipes
# --------------------------------------------------------------------------- #

HAND_WRITTEN: dict[str, Recipe] = {
    "hepatic_steatosis": Recipe(
        organ="liver",
        tools=[
            ToolCall("liver_to_spleen_hu_ratio"),
            ToolCall("organ_morphometrics", {"organ_class": "liver"}),
        ],
        template_key="hepatic_steatosis",
        tier="A",
    ),

    "hepatic_cyst": Recipe(
        organ="liver",
        tools=[
            ToolCall("cam_peak", {"encoder": "merlin"}),
            ToolCall("cam_connected_components", {"encoder": "merlin", "threshold": 0.7}),
            ToolCall("liver_segment_at", {"voxel_idx": "$cam_peak.voxel_idx"}),
            ToolCall("sample_hu_at", {"voxel_idx": "$cam_peak.voxel_idx"}),
            ToolCall("lesion_in_organ", {"voxel_idx": "$cam_peak.voxel_idx",
                                          "organ_class": "liver"}),
        ],
        template_key="hepatic_cyst",
        tier="A",
    ),

    "hepatic_lesion": Recipe(
        organ="liver",
        tools=[
            ToolCall("cam_peak", {"encoder": "merlin"}),
            ToolCall("liver_segment_at", {"voxel_idx": "$cam_peak.voxel_idx"}),
            ToolCall("sample_hu_at", {"voxel_idx": "$cam_peak.voxel_idx"}),
            ToolCall("cam_connected_components", {"encoder": "merlin", "threshold": 0.7}),
        ],
        template_key="hepatic_lesion",
        tier="A",
    ),

    "splenomegaly": Recipe(
        organ="spleen",
        tools=[
            ToolCall("organ_morphometrics", {"organ_class": "spleen"}),
        ],
        template_key="splenomegaly",
        tier="A",
    ),

    "renal_cyst": Recipe(
        organ=None,                                # could be left or right
        tools=[
            ToolCall("cam_peak", {"encoder": "merlin"}),
            ToolCall("kidney_side_at", {"voxel_idx": "$cam_peak.voxel_idx"}),
            ToolCall("sample_hu_at", {"voxel_idx": "$cam_peak.voxel_idx"}),
            ToolCall("cam_connected_components", {"encoder": "merlin", "threshold": 0.7}),
        ],
        template_key="renal_cyst",
        tier="A",
    ),

    "aortic_atherosclerosis": Recipe(
        organ="aorta",
        tools=[],                                  # no enrichment needed
        template_key="aortic_atherosclerosis",
        tier="A",
    ),

    "ascites": Recipe(
        organ=None,
        tools=[
            ToolCall("cam_peak", {"encoder": "merlin"}),
        ],
        template_key="ascites",
        tier="A",
    ),

    "pleural_effusion": Recipe(
        organ=None,
        tools=[
            ToolCall("cam_peak", {"encoder": "merlin"}),
        ],
        template_key="pleural_effusion",
        tier="A",
    ),

    "appendicitis": Recipe(
        organ=None,
        tools=[
            ToolCall("cam_peak", {"encoder": "merlin"}),
        ],
        template_key="appendicitis",
        tier="A",
    ),

    "lymphadenopathy": Recipe(
        organ=None,
        tools=[
            ToolCall("cam_peak", {"encoder": "merlin"}),
        ],
        template_key="lymphadenopathy",
        tier="A",
    ),

    "pancreatitis": Recipe(
        organ="pancreas",
        tools=[
            ToolCall("cam_peak", {"encoder": "merlin"}),
        ],
        template_key="pancreatitis",
        tier="A",
    ),

    "cardiomegaly": Recipe(
        organ="Visible Thoracic",
        tools=[ToolCall("cam_peak", {"encoder": "merlin"})],
        template_key="cardiomegaly",
        tier="A",
    ),

    "atelectasis": Recipe(
        organ="Visible Thoracic",
        tools=[ToolCall("cam_peak", {"encoder": "merlin"})],
        template_key="atelectasis",
        tier="A",
    ),

    "osteopenia": Recipe(
        organ="Musculoskeletal",
        tools=[],
        template_key="osteopenia",
        tier="A",
    ),

    "gallbladder_stones": Recipe(
        organ="gall_bladder",
        tools=[
            ToolCall("cam_peak", {"encoder": "merlin"}),
            ToolCall("cam_connected_components", {"encoder": "merlin", "threshold": 0.7}),
        ],
        template_key="gallbladder_stones",
        tier="A",
    ),

    "hydronephrosis": Recipe(
        organ=None,
        tools=[
            ToolCall("cam_peak", {"encoder": "merlin"}),
            ToolCall("kidney_side_at", {"voxel_idx": "$cam_peak.voxel_idx"}),
        ],
        template_key="hydronephrosis",
        tier="A",
    ),
}


# --------------------------------------------------------------------------- #
# Lookup with fallback to organ-category generic
# --------------------------------------------------------------------------- #

def lookup(canonical: str) -> Recipe:
    """Return the recipe for `canonical`. Falls back to Tier B/C.

    Tier B — finding has an entry in CANONICAL_TO_CATEGORY (i.e. RATE-derived):
            organ-anchored generic recipe with CAM peak + axial slice.
    Tier C — no organ category: generic-fallback recipe with just the
            probability annotation.
    """
    if canonical in HAND_WRITTEN:
        return HAND_WRITTEN[canonical]
    category = CANONICAL_TO_CATEGORY.get(canonical)
    if category:
        return Recipe(
            organ=None,
            tools=[ToolCall("cam_peak", {"encoder": "merlin"})],
            template_key="_generic_organ",
            tier="B",
        )
    return Recipe(
        organ=None,
        tools=[],
        template_key="_generic_fallback",
        tier="C",
    )


def covered_canonicals() -> dict[str, int]:
    """Diagnostic: how many findings fall into each tier?"""
    counts = {"A": 0, "B": 0, "C": 0}
    counts["A"] = len(HAND_WRITTEN)
    for canonical in CANONICAL_TO_CATEGORY:
        if canonical in HAND_WRITTEN:
            continue
        counts["B"] += 1
    # Tier C is hard to count without the full finding list; in practice the
    # negspaCy probe's 50 findings vs the RATE 225 are the universes we
    # care about. Caller can introspect HAND_WRITTEN / CATEGORY_TO_CANONICALS.
    return counts
