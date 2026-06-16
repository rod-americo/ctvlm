"""Router — runs a finding's recipe and returns a StructuredFinding.

For each positive finding, the pipeline calls Router.route(sid, finding, prob).
The router looks up the recipe, executes each tool through the cache, resolves
`$ref` arguments against earlier tool results, and assembles the
StructuredFinding ready for the templating step.
"""
from __future__ import annotations

import time
from typing import Any

from src.agent import recipes, templates
from src.agent.cache import cached
from src.agent.schema import Recipe, StructuredFinding, ToolCall, ToolResult
from src.agent.tools import REGISTRY


def _resolve_refs(args: dict, prior: dict[str, ToolResult]) -> dict:
    """Replace any string-valued `$tool.field` arg with the resolved value."""
    out = {}
    for k, v in args.items():
        if isinstance(v, str) and v.startswith("$"):
            ref = v[1:]
            if "." in ref:
                tool, field = ref.split(".", 1)
                tr = prior.get(tool)
                if tr is None or not isinstance(tr.result, dict):
                    out[k] = None
                else:
                    out[k] = tr.result.get(field)
            else:
                tr = prior.get(ref)
                out[k] = tr.result if tr else None
        else:
            out[k] = v
    return out


def _run_tool(sid: str, tc: ToolCall, prior: dict[str, ToolResult],
              *, finding: str) -> ToolResult:
    """Resolve refs → look up callable → call (through cache) → wrap.

    `finding` is auto-injected for tools whose signature needs it (cam_*); the
    recipe never has to know about it.
    """
    args = _resolve_refs(tc.args, prior)
    fn = REGISTRY.get(tc.name)
    if fn is None:
        return ToolResult(tc.name, args, {"error": f"unknown tool {tc.name!r}"}, 0.0)
    # Tools that need the finding canonical (CAM read paths). The kwarg goes
    # into args before cache-keying so different findings get distinct cache
    # entries.
    if tc.name in ("cam_peak", "cam_connected_components"):
        args = {**args, "finding": finding}
    # `axial_slice_of` doesn't take sid; handle the small-arity case.
    if tc.name == "axial_slice_of":
        t0 = time.time()
        try:
            res = fn(args["voxel_idx"])
        except Exception as e:
            res = {"error": f"{type(e).__name__}: {e}"}
        return ToolResult(tc.name, args, res, time.time() - t0)
    t0 = time.time()
    try:
        result, hit = cached(sid, tc.name, args, lambda: fn(sid, **args))
    except Exception as e:
        result, hit = {"error": f"{type(e).__name__}: {e}"}, False
    return ToolResult(tc.name, args, result, time.time() - t0, cache_hit=hit)


def _gather_fields(recipe: Recipe, results: dict[str, ToolResult],
                   finding: str) -> dict[str, Any]:
    """Pluck the values the template needs out of the tool results."""
    fields: dict[str, Any] = {"template_key": recipe.template_key}

    # Liver-to-spleen HU bundle (hepatic_steatosis)
    if "liver_to_spleen_hu_ratio" in results:
        r = results["liver_to_spleen_hu_ratio"].result
        if r.get("valid"):
            fields["liver_mean_hu"] = r["liver_mean_hu"]
            fields["spleen_mean_hu"] = r["spleen_mean_hu"]
            fields["liver_minus_spleen_hu"] = r["liver_minus_spleen_hu"]

    # Organ morphometrics (splenomegaly etc.)
    if "organ_morphometrics" in results:
        r = results["organ_morphometrics"].result
        if r.get("present"):
            if "volume_ml" in r and r["volume_ml"] is not None:
                fields["volume_ml"] = r["volume_ml"]
            ext = r.get("extent_mm")
            if isinstance(ext, list) and len(ext) >= 3:
                # craniocaudal extent ≈ extent along the S axis (index 2 in RAS)
                fields["extent_cc_mm"] = ext[2]

    # CAM peak
    if "cam_peak" in results:
        r = results["cam_peak"].result
        if r.get("valid"):
            fields["voxel_idx"] = r["voxel_idx"]
            fields["world_mm"] = r["world_mm"]
            fields["axial_slice"] = r["axial_slice"]

    # CAM connected components — pick largest
    if "cam_connected_components" in results:
        r = results["cam_connected_components"].result
        if r.get("valid") and r.get("components"):
            top = r["components"][0]
            fields["extent_mm"] = top["extent_mm"]
            fields["n_vox_largest"] = top["n_vox"]

    # Liver segment
    if "liver_segment_at" in results:
        r = results["liver_segment_at"].result
        if r.get("valid"):
            fields["segment"] = r["segment"]
            fields["segment_roman"] = r["roman"]

    # Pancreas region
    if "pancreas_region_at" in results:
        r = results["pancreas_region_at"].result
        if r.get("valid"):
            fields["pancreas_region"] = r["region"]

    # Kidney side
    if "kidney_side_at" in results:
        r = results["kidney_side_at"].result
        if r.get("valid"):
            fields["side"] = r["side"]

    # Per-voxel HU sample
    if "sample_hu_at" in results:
        r = results["sample_hu_at"].result
        if r.get("valid"):
            fields["mean_hu"] = r["mean_hu"]

    # Side determination for pleural_effusion based on CAM world coord
    if finding == "pleural_effusion" and "world_mm" in fields:
        x = fields["world_mm"][0]
        if x > 30:
            fields["side"] = "right-sided"
        elif x < -30:
            fields["side"] = "left-sided"
        else:
            fields["side"] = "bilateral"

    return fields


def route(sid: str, finding: str, probability: float) -> StructuredFinding:
    """Run the recipe for `finding` and return a fully-rendered StructuredFinding."""
    recipe = recipes.lookup(finding)
    results: dict[str, ToolResult] = {}
    tool_results_list: list[ToolResult] = []

    for tc in recipe.tools:
        tr = _run_tool(sid, tc, results, finding=finding)
        results[tc.name] = tr
        tool_results_list.append(tr)

    fields = _gather_fields(recipe, results, finding)

    # Organ label key for the generic templates
    if recipe.organ:
        fields["organ_label_key"] = recipe.organ
    else:
        # Use the RATE canonical category as the label key for Tier B fallback
        from src.agent.canonical import CANONICAL_TO_CATEGORY
        fields["organ_label_key"] = CANONICAL_TO_CATEGORY.get(finding) or None

    sf = StructuredFinding(
        finding=finding,
        probability=probability,
        organ=recipe.organ,
        recipe_tier=recipe.tier,
        tool_results=tool_results_list,
        fields=fields,
    )
    sf.sentence = templates.render_finding(sf)
    return sf
