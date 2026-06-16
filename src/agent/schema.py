"""Typed dataclasses for the agentic report pipeline.

The pipeline lifecycle, in shapes:

    prediction (probe sigmoid)        :  dict[finding, float]
    routing per positive finding      :  Recipe -> [ToolCall] -> [ToolResult] -> StructuredFinding
    aggregation                       :  list[StructuredFinding] -> structured summary text
    LLM rendering                     :  summary -> prose paragraph
    final                             :  FullReport (everything + audit trail)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """One tool to execute. `args` keys may include `$ref` strings that point at
    earlier tool results (e.g. `"voxel_idx": "$cam_peak.voxel_idx"`)."""
    name: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """Output of one ToolCall."""
    name: str
    args: dict[str, Any]
    result: Any                       # JSON-serialisable
    latency_s: float
    cache_hit: bool = False


@dataclass
class Recipe:
    """Per-finding recipe: which organ this finding implicates, which tools fire,
    which template renders the sentence."""
    organ: str | None                 # MerlinPlus class (e.g. "liver") or None
    tools: list[ToolCall]
    template_key: str                 # key into templates.SENTENCES
    tier: str = "B"                   # "A" hand-written, "B" generic-organ, "C" generic-fallback


@dataclass
class StructuredFinding:
    """One finding with enrichment ready for templating."""
    finding: str                      # canonical snake_case
    probability: float
    organ: str | None
    recipe_tier: str
    tool_results: list[ToolResult] = field(default_factory=list)
    fields: dict[str, Any] = field(default_factory=dict)   # template-substitutable values
    sentence: str = ""                # rendered template; populated by templates.render_finding


@dataclass
class FullReport:
    """The whole pipeline output for one study."""
    study_id: str
    probabilities: dict[str, float]   # canonical -> sigmoid prob (filtered or not)
    positives: list[str]              # canonical findings above threshold, in render order
    structured: list[StructuredFinding]
    summary_text: str                 # structured intermediate text fed to LLM
    prose: str                        # LLM-rendered final paragraph
    threshold: float
    encoder: str
    probe_path: str
    latency_total_s: float = 0.0
