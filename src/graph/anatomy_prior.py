"""Static anatomical prior graph over abdominal organs (Phase 3).

A fixed, clinically-grounded typed edge set among the Pillar-0 node organs. Per-study
graphs (`study_graph.py`) overlay these edges onto whichever organs are present in a
case, then add dynamic finding-derived edges later. Edges encode the relationships a
radiologist reasons over: spatial adjacency, GI continuity, and vascular supply/drainage.

Edge types (plan_1.md Phase 3): adjacent_to, drains_to, vascular_supply. `contains`
and `same_pathology_signature` are added when region nodes / findings arrive.

Public API:
    NODE_ORGANS                 -> canonical organ vocabulary (matches extraction)
    EDGE_TYPES                  -> ordered edge-type names (index = type id)
    prior_edges()               -> list[(src, dst, type)] directed
    build_prior() -> nx.DiGraph -> the prior as a typed directed graph (for inspection)
"""
from __future__ import annotations

# Must match the node set used in scripts/18_pillar0_embed_parallel.py.
NODE_ORGANS = [
    "liver", "spleen", "kidney_left", "kidney_right", "pancreas", "gall_bladder",
    "adrenal_gland_left", "adrenal_gland_right", "bladder", "stomach",
    "aorta", "postcava", "esophagus", "duodenum", "colon",
    "portal_vein_and_splenic_vein",
]

EDGE_TYPES = ["adjacent_to", "drains_to", "vascular_supply"]

# Spatial adjacency (undirected — stored once, symmetrized at build time).
_ADJACENT = [
    ("liver", "gall_bladder"), ("liver", "stomach"), ("liver", "duodenum"),
    ("liver", "kidney_right"), ("liver", "adrenal_gland_right"),
    ("spleen", "stomach"), ("spleen", "kidney_left"), ("spleen", "pancreas"),
    ("spleen", "adrenal_gland_left"),
    ("pancreas", "duodenum"), ("pancreas", "stomach"),
    ("kidney_left", "adrenal_gland_left"), ("kidney_right", "adrenal_gland_right"),
    ("stomach", "duodenum"), ("esophagus", "stomach"),
    ("aorta", "postcava"), ("colon", "bladder"),
]

# GI luminal continuity (directed, oral → aboral).
_DRAINS_TO = [
    ("esophagus", "stomach"), ("stomach", "duodenum"), ("duodenum", "colon"),
    # venous drainage to the portal / systemic veins
    ("spleen", "portal_vein_and_splenic_vein"),
    ("pancreas", "portal_vein_and_splenic_vein"),
    ("portal_vein_and_splenic_vein", "liver"),
    ("liver", "postcava"), ("kidney_left", "postcava"), ("kidney_right", "postcava"),
    ("adrenal_gland_right", "postcava"),
]

# Arterial supply from the aorta (directed, aorta → organ).
_VASCULAR_SUPPLY = [
    ("aorta", organ) for organ in
    ["liver", "spleen", "kidney_left", "kidney_right", "pancreas", "stomach",
     "adrenal_gland_left", "adrenal_gland_right", "duodenum", "colon",
     "esophagus", "gall_bladder", "portal_vein_and_splenic_vein"]
]


def prior_edges() -> list[tuple[str, str, str]]:
    """All prior edges as directed (src, dst, type). Adjacency is emitted both ways."""
    edges: list[tuple[str, str, str]] = []
    for a, b in _ADJACENT:
        edges.append((a, b, "adjacent_to"))
        edges.append((b, a, "adjacent_to"))
    edges += [(a, b, "drains_to") for a, b in _DRAINS_TO]
    edges += [(a, b, "vascular_supply") for a, b in _VASCULAR_SUPPLY]
    # sanity: every endpoint is a known organ
    bad = {n for a, b, _ in edges for n in (a, b) if n not in NODE_ORGANS}
    if bad:
        raise ValueError(f"prior edges reference unknown organs: {sorted(bad)}")
    return edges


def build_prior():
    """Return the prior as a networkx DiGraph (nodes=NODE_ORGANS, edges typed)."""
    import networkx as nx
    g = nx.DiGraph()
    g.add_nodes_from(NODE_ORGANS)
    for a, b, t in prior_edges():
        g.add_edge(a, b, edge_type=t)
    return g
