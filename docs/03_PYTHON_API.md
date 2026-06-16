# 03 — Python API

The only entry point the worker should call is `src.agent.pipeline.generate_report`. Everything below is reference for that surface area.

## Schema

All defined in `src/agent/schema.py`:

```python
@dataclass
class ToolCall:
    name: str                      # tool name in src.agent.tools.REGISTRY
    args: dict                     # kwargs (may contain $-ref placeholders)

@dataclass
class ToolResult:
    name: str
    args: dict
    result: dict                   # JSON-serialisable
    latency_s: float
    cache_hit: bool = False

@dataclass
class StructuredFinding:
    finding: str                   # canonical snake_case name
    probability: float             # CALIBRATED (post-Platt)
    organ: str | None              # 'liver', 'spleen', ..., or RATE category, or None
    recipe_tier: str               # 'A' / 'B' / 'C'
    tool_results: list[ToolResult] # per-finding tool trace
    fields: dict                   # template fields gathered from tool results
    sentence: str = ""             # final template-rendered sentence

@dataclass
class FullReport:
    study_id: str
    probabilities: dict[str, float]    # 220 entries: canonical → calibrated prob
    positives: list[str]               # filtered finding names (post COREQUIRES/SUBSUMPTION/cap)
    structured: list[StructuredFinding]
    summary_text: str                  # rendered "Findings:" block
    prose: str                         # LLM output (production: "(LLM skipped)")
    threshold: float
    encoder: str                       # 'merlin+pillar0_L2' (':nc' suffix if contrast_phase='nc')
    probe_path: str
    latency_total_s: float
```

## Main entry: `generate_report`

```python
from src.agent import pipeline

def generate_report(
    sid: str,
    *,
    threshold: float = 0.5,            # Used only as fallback for findings without Youden-J
    min_threshold: float = 0.20,       # Floor applied on top of per-finding threshold
    max_findings: int = 12,            # Cap on positives rendered
    probe_path: Path | None = None,    # Defaults to DEFAULT_PROBE
    contrast_phase: str = "ce",        # 'ce' or 'nc' — see docs/07_CALIBRATION.md
    skip_llm: bool = False,            # Set True in production (no LLM)
) -> FullReport:
    ...
```

### How positives are selected

For each of the 220 canonical findings:

```python
effective_threshold = max(
    per_finding_youden_j[finding],     # from probe checkpoint
    min_threshold,                      # universal floor (0.20 default)
)
positive = (calibrated_prob[finding] >= effective_threshold)
```

Then `render.filter_positives()` applies in order:
1. **COREQUIRES** — catch-all findings (e.g. `metastatic_disease`) suppressed unless a corroborating specific finding is also positive.
2. **SUBSUMPTION** — generic findings (e.g. `fracture`) suppressed when a specific finding (e.g. `spinal_fracture`) is also positive.
3. **Per-category cap** — only the top-K probabilities per RATE category survive.

The surviving positives are sorted by probability descending, truncated to `max_findings`, then handed to the per-finding router.

### Contrast-phase argument

Pass `contrast_phase="nc"` to get the non-contrast calibration path:
- NC-specific Platt scaling on 26 high-prevalence findings (the rest fall back to default)
- NC-recomputed Youden-J thresholds on those 26 findings
- Encoder label suffix `:nc` for the audit trail

For phase auto-detection, read DICOM `ContrastBolusAgent` / `ContrastBolusVolume` at intake time and pass through (see [docs/05_ORTHANC_INTEGRATION.md](05_ORTHANC_INTEGRATION.md)).

## Lower-level entry: `predict`

```python
def predict(
    sid: str,
    probe_path: Path | None = None,
    contrast_phase: str = "ce",
) -> tuple[
    dict[str, float],     # probabilities (post Platt)
    list[str],            # canonical finding names (in checkpoint order)
    str,                  # encoder label
    dict[str, float],     # per-finding Youden-J thresholds
    dict[str, float],     # per-finding val AUROCs
]:
    ...
```

Use this if you want the probabilities without triggering the router/template work — e.g. for a dashboard probability bar chart, or to apply your own filtering.

## Return-value layout — minimal example

```python
report = pipeline.generate_report("AC421363f", skip_llm=True, contrast_phase="ce")

# 1. The structured Findings: block — what the report says
print(report.summary_text)
# Findings:
#   Thorax (visible): left-sided pleural effusion at the lung base (axial slice 261)
#   ...

# 2. Per-positive trace — what tools fired and what they returned
for sf in report.structured:
    print(f"{sf.finding}  p={sf.probability:.3f}  tier={sf.recipe_tier}")
    for tr in sf.tool_results:
        print(f"  {tr.name}({tr.args}) → {tr.result}  [{tr.latency_s:.2f}s]")

# 3. Full probability vector — useful for downstream scoring / dashboards
top5 = sorted(report.probabilities.items(), key=lambda x: -x[1])[:5]
```

## Optional: Grad-CAM regeneration

`generate_report` already triggers lazy heatmap generation via the `cam_peak` / `cam_connected_components` tools whenever a Tier A/B recipe needs a slice number that isn't cached. If you need a heatmap for an arbitrary (sid, encoder, finding) tuple outside the report flow:

```python
from src.explain import cam
out_path = cam.ensure_concat_heatmap(sid="AC421363f", encoder="merlin",
                                     finding="hepatic_cyst")
# returns a Path to the cached/generated NIfTI, or None on failure
```

See [docs/09_GRAD_CAM.md](09_GRAD_CAM.md) for the full surface.

## Concurrency

- **One inference per process** is the contract. The encoders are cached in `lru_cache` at module level — calling `predict` twice in parallel from threads will share the same model and the same forward CUDA stream. If you need parallel inference, run multiple worker processes (each loads its own Merlin + Pillar-0 once).
- The probe checkpoint is also cached at module level (`pipeline._probe_cache`). Single process, single GPU, sequential inferences is the validated pattern.
- For multi-GPU horizontal scaling, see [docs/06_GPU_BROKER_INTEGRATION.md](06_GPU_BROKER_INTEGRATION.md).
