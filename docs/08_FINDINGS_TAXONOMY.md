# 08 — Findings taxonomy

The probe outputs **220 canonical findings** (snake_case names), derived from 225 RATE questions by OR-collapsing 5 duplicate canonicals. The mapping lives in `data/rate_canonical_map.csv` and is loaded by `src/agent/canonical.py`.

## Categories (17)

| category | n_canonicals |
|---|---|
| Liver | 32 |
| Gastrointestinal | 30 |
| Genitourinary | 27 |
| Female pelvis | 21 |
| Pancreas | 17 |
| Device | 12 |
| Gallbladder | 11 |
| Spleen | 10 |
| Male Pelvis | 10 |
| Musculoskeletal | 10 |
| Biliary Tree | 8 |
| Visible Thoracic | 8 |
| Adrenal gland | 7 |
| Great Vessel | 7 |
| Peritoneum | 5 |
| Retroperitoneum | 4 |
| Multi Organs | 2 |

Use `src.agent.canonical.CATEGORY_TO_CANONICALS[category]` to iterate findings in a category.

## Sentence template tiers

Each canonical resolves to a `Recipe` via `src.agent.recipes.lookup(canonical)`:

### Tier A — 16 hand-written, measurement-bearing templates

Findings with finding-specific phrasing, tool calls for measurements, and concrete clinical wording. Defined in `src/agent/templates.py:SENTENCES` and `src/agent/recipes.py:HAND_WRITTEN`.

| canonical | template highlights | tools used |
|---|---|---|
| `hepatic_steatosis` | "diffusely decreased attenuation (mean X HU, liver-spleen ±Y)" | liver-to-spleen HU ratio + organ_morphometrics |
| `hepatic_cyst` | "in segment N, measuring A × B cm, mean HU C" | cam_peak, cam_cc, liver_segment_at, sample_hu_at, lesion_in_organ |
| `hepatic_lesion` | same as hepatic_cyst minus "simple cyst" qualifier | same |
| `splenomegaly` | "volume X mL, craniocaudal Y mm" | organ_morphometrics |
| `renal_cyst` | "in the {side} kidney, mean HU X" | cam_peak, kidney_side_at, sample_hu_at, cam_cc |
| `aortic_atherosclerosis` | "mural calcifications throughout" | none (Tier A but no enrichment) |
| `ascites` | "free intraperitoneal fluid (axial slice N)" | cam_peak |
| `pleural_effusion` | "{side} pleural effusion at the lung base (slice N)" | cam_peak |
| `appendicitis` | "enlarged tubular blind-ending structure (slice N)" | cam_peak |
| `lymphadenopathy` | "enlarged lymph nodes (slice N)" | cam_peak |
| `pancreatitis` | "peripancreatic stranding (slice N)" | cam_peak |
| `cardiomegaly` | "enlarged cardiac silhouette (slice N)" | cam_peak |
| `atelectasis` | "dependent atelectasis at the lung bases (slice N)" | cam_peak |
| `osteopenia` | "generalized osteopenia of the visualised skeleton" | none |
| `gallbladder_stones` | "discrete radiopaque calculi (slice N) measuring X × Y cm" | cam_peak, cam_cc |
| `hydronephrosis` | "in the {side} kidney (slice N)" | cam_peak, kidney_side_at |

### Tier B — 204 organ-anchored generic templates

When a canonical has a RATE category but no hand-written entry, `recipes.lookup` returns:

```
Recipe(
    organ=None,
    tools=[ToolCall("cam_peak", {"encoder": "merlin"})],
    template_key="_generic_organ",
    tier="B",
)
```

The Tier B template renders as `"{Organ Category}: {finding humanised} (axial slice N)"`. Example: `Pancreas: intraductal papillary mucinous neoplasm (axial slice 258).`

The cam_peak tool **lazily triggers Grad-CAM generation** if a heatmap isn't cached for `(sid, merlin, finding)`. First time a finding hits, this is ~3 s; cached afterwards.

### Tier C — generic fallback

Used only if a finding has no RATE category (shouldn't happen with the current canonical map, but exists as a safety net):

```
Recipe(organ=None, tools=[], template_key="_generic_fallback", tier="C")
```

Renders as the bare humanised finding name.

## Filtering rules (applied before render)

`src/agent/render.py:filter_positives` runs over the set of above-threshold canonicals:

### `COREQUIRES` — catch-all suppression

Some findings are too general to plausibly stand alone. They are dropped unless at least one corroborating specific finding is also above its threshold:

| catch-all | requires at least one of |
|---|---|
| `metastatic_disease` | `hepatic_metastases`, `peritoneal_carcinomatosis`, `hepatocellular_carcinoma`, `cholangiocarcinoma`, `renal_cell_carcinoma`, `colonic_carcinoma`, `rectal_carcinoma`, `gastric_carcinoma`, `duodenal_carcinoma`, `esophageal_carcinoma`, `ductal_pancreatic_carcinoma`, `neuroendocrine_tumor`, `ovarian_cancer`, `cervical_mass`, `vaginal_cancer`, `prostate_cancer`, `bladder_transitional_cell_carcinoma`, `gallbladder_cancer`, `gastrointestinal_lymphoma`, `hepatic_lymphoma`, `splenic_lymphoma`, `renal_lymphoma`, `wilms_tumor` |

Mechanism: catches the over-firing pattern where the probe flags `metastatic_disease` based on any single high-prob lesion in an unusual organ.

### `SUBSUMPTION_RULES` — specific dominates generic

When a specific finding is positive, drop the generic counterpart:

| generic dropped | when any of these is positive |
|---|---|
| `fracture` | `spinal_fracture`, `rib_fracture`, `femoral_fracture`, `pelvic_girdle_fracture` |
| `hepatic_lesion` | `hepatic_cyst`, `hepatic_hemangioma`, `hepatic_adenoma`, `hepatic_metastases`, `hepatocellular_carcinoma`, `hepatic_focal_nodular_hyperplasia` |
| `hepatic_mass` | `hepatocellular_carcinoma`, `hepatic_metastases`, `fibrolamellar_carcinoma`, `cholangiocarcinoma`, `hepatic_adenoma`, `hepatic_lymphoma` |
| `adrenal_mass` | `adrenal_adenoma`, `adrenal_myelolipoma`, `pheochromocytoma`, `adrenal_hyperplasia` |
| `ovarian_tumor` | `ovarian_cancer`, `ovarian_teratoma` |
| `pancreatic_tumor` | `ductal_pancreatic_carcinoma`, `neuroendocrine_tumor`, `IPMN`, `mucinous_cystic_neoplasm`, `serous_cystic_neoplasm`, `solid_pseudopapillary_tumor` |
| `testicular_mass` | `testicular_infarct`, `testicular_torsion` |
| `bowel_obstruction` | `small_bowel_obstruction`, `large_bowel_obstruction` |

Note: `renal_hypodensity` is **NOT** subsumed by `simple_renal_cyst` / `complex_renal_cyst`. The two are semantically distinct in RATE labels and subsuming costs ~900+ FNs (see `docs/10_PERFORMANCE.md`).

### `CATEGORY_CAPS` — per-organ overcalling cap

Caps the number of positive findings the renderer takes per RATE category, sorted by probability descending:

| category | cap |
|---|---|
| Multi Organs | 1 |
| Pancreas, Spleen, Gallbladder, Biliary Tree, Adrenal gland, Great Vessel, Peritoneum, Retroperitoneum | 2 |
| Visible Thoracic, Female pelvis, Male Pelvis, **default** | 3 |
| Liver, Genitourinary, Gastrointestinal, Musculoskeletal | 4 |
| Device | 5 |

Rationale: the probe over-fires intra-organ on co-occurrence-prone findings (vascular cluster, pancreatic cluster). Caps trim the lowest-confidence tail without hurting the dominant finding.

## Negative-organ "normal" statements

When an organ category has **no positive sentence** AND the maximum probability of any canonical in that category is below `negative_threshold=0.40`, the renderer emits one of these (defined in `render.py:NEGATIVE_STATEMENTS`):

```
Liver: normal in size and attenuation, no focal lesion.
Spleen: normal in size and attenuation.
Pancreas: unremarkable, no ductal dilation.
Gallbladder: unremarkable, no wall thickening or stones.
Biliary tree: no intrahepatic or extrahepatic ductal dilation.
Adrenal glands: normal in size and contour.
Genitourinary: kidneys and bladder unremarkable, no hydronephrosis or stones.
Peritoneum: no free fluid or free air.
Abdominal aorta: normal in caliber, no aneurysm or dissection.
Retroperitoneum: no lymphadenopathy.
Visible thorax: lung bases clear, no pleural effusion.
Bowel: no obstruction, wall thickening, or pneumatosis.
```

This makes the output read like a real radiology Findings: block where organs without findings are explicitly called normal.

## Adding a new Tier A template

1. Add a template string to `src/agent/templates.py:SENTENCES`.
2. Add a `Recipe(..., tier="A")` to `src/agent/recipes.py:HAND_WRITTEN`.
3. (If the recipe needs a new tool) add the pure-function tool to `src/agent/tools.py:REGISTRY`.
4. Tests don't enforce template coverage; consider adding one in `tests/`.
