# 14 - CAC-DRS Chest CT probe concept

## Goal

Train a small supervised probe that predicts coronary artery calcium burden from CT
embeddings, using a CAC-DRS / Agatston-labeled dataset as supervision.

This is a research plan only. It is not a clinical claim, not a deployment protocol,
and not a replacement for validated CAC scoring software or radiologist review.

## Starting point

Available data:

- Gated cardiac calcium-score CT studies.
- Numeric Agatston score per study.
- Potentially CAC-DRS class derivable from the numeric score.

Target idea:

- Use `YalaLab/Pillar0-ChestCT` as a frozen 3D CT encoder.
- Train a lightweight probe on top of its 1152-dimensional embedding.
- First solve the easier in-domain task on gated calcium-score CT.
- Later test whether the learned signal transfers to routine, non-gated chest CT.

The model card for `YalaLab/Pillar0-ChestCT` describes it as a volumetric chest CT
foundation model with a `CLIPMultimodalAtlas` architecture and a 1152-dimensional
embedding. Its config expects 11 CT window channels and a 256x256x256 volume.

## Why this is plausible

The current abdomen pipeline already uses this pattern:

```text
CT volume
-> frozen encoder embeddings
-> small supervised probe
-> calibrated finding probabilities
```

For CAC-DRS, the analogous version is:

```text
Chest/cardiac CT
-> Pillar0-ChestCT embedding, 1152-d
-> CAC probe
-> Agatston risk class and/or numeric score estimate
```

The probe should be cheap to train because the expensive 3D encoder is frozen.
The hard part is not compute; it is label quality, protocol shift, and validation.

## Label targets

Use multiple heads if possible. They answer related but different questions.

### Binary CAC

```text
CAC absent vs present
Agatston == 0 vs Agatston > 0
```

This is the simplest first baseline and should be trained even if the final goal is
more detailed. It catches whether the embedding carries any CAC signal at all.

### CAC-DRS class

Typical Agatston-derived bins:

```text
A0: 0
A1: 1-99
A2: 100-299
A3: >=300
```

Confirm these bins against the exact CAC-DRS version and reporting policy before
using them in any formal study.

### Numeric Agatston

Raw Agatston is highly skewed, with many zeros and a long positive tail. Do not
regress the raw number directly as the only target. Prefer:

```text
y = log1p(Agatston)
```

Then report both:

- regression error on `log1p(Agatston)`;
- derived class accuracy after converting predicted score back into bins.

### Optional vessel involvement

If annotations include vessel involvement, train a multitask head:

```text
left main, LAD, LCx, RCA: present/absent
number of involved vessels: N class
```

Do not infer vessel-level output from study-level Agatston alone.

## Recommended model heads

Start with frozen embeddings and simple heads:

```text
embedding 1152
-> binary head: CAC > 0
-> ordinal/class head: A0/A1/A2/A3
-> regression head: log1p(Agatston)
```

Implementation options:

- Linear head first.
- One-hidden-layer MLP only if the linear baseline saturates.
- Class-weighted or focal loss for class imbalance.
- Ordinal loss is preferable to plain multiclass cross-entropy if class order matters.

A practical first version:

```python
class CACProbe(nn.Module):
    def __init__(self, in_dim=1152):
        super().__init__()
        self.any_cac = nn.Linear(in_dim, 1)
        self.class_head = nn.Linear(in_dim, 4)
        self.score_head = nn.Linear(in_dim, 1)
```

Loss:

```text
loss =
  BCEWithLogits(any_cac)
  + CE or ordinal loss(A0/A1/A2/A3)
  + HuberLoss(log1p_score)
```

Tune the weights after inspecting validation behavior. Do not let the regression
tail dominate the classification task.

## Data split

Avoid leakage. Split by patient, not by study.

Preferred splits:

- train/validation/test by patient ID;
- scanner/protocol-stratified test set if available;
- external or later-period test set if available.

Keep a small untouched test set for final reporting. Use validation only for model
selection, calibration, and threshold choice.

## Domain shift

The first dataset is gated cardiac calcium-score CT. That is not the same domain as
routine chest CT.

Expected differences:

| Source | Characteristics | Risk |
|---|---|---|
| Gated CAC CT | heart centered, non-contrast, calcium protocol, less motion | easier, in-domain |
| Routine chest CT | larger FOV, non-gated, variable contrast/protocol, cardiac motion | transfer may degrade |

Therefore:

1. Train and validate first on gated CAC CT.
2. Treat routine chest CT as a separate external validation target.
3. Do not assume a gated-trained probe is valid on routine chest CT without a labeled
   routine chest CT sample.

If routine chest CT labels are scarce, a useful next experiment is:

- train on gated CAC CT;
- freeze the encoder and probe;
- evaluate on a small manually reviewed non-gated chest CT set;
- fine-tune only calibration or a small adapter if needed.

## Metrics

Report metrics per target.

Binary CAC:

- AUROC;
- AUPRC;
- sensitivity/specificity at chosen operating points;
- negative predictive value if the intended use is exclusion.

CAC-DRS class:

- accuracy;
- macro-F1;
- quadratic weighted kappa;
- adjacent-error rate;
- severe-error rate, for example A0 predicted as A3 or A3 predicted as A0.

Numeric Agatston:

- MAE on `log1p(Agatston)`;
- Spearman correlation;
- class agreement after binning;
- calibration plot for class probabilities.

Clinical-style summary:

```text
CAC present: probability 0.93
Predicted CAC-DRS: A2
Predicted log1p(Agatston): 5.12
Derived Agatston estimate: 166
```

For reporting, prefer the class output over claiming precise Agatston unless numeric
agreement is strong and externally validated.

## Output format

The first useful output is structured:

```json
{
  "cac_present_probability": 0.93,
  "cac_drs_class": "A2",
  "class_probabilities": {
    "A0": 0.02,
    "A1": 0.16,
    "A2": 0.68,
    "A3": 0.14
  },
  "log1p_agatston": 5.12,
  "agatston_estimate": 166
}
```

Text can be rendered deterministically:

```text
Coronary artery calcium is present. Estimated category: CAC-DRS A2.
```

Avoid reporting unsupported vessel-level localization unless vessel labels or a
separate localization method were trained and validated.

## What segmentation would add

Study-level labels are sufficient to train the probe above.

Segmentation, boxes, or slice-level labels are only required if the goal changes to:

- lesion localization;
- vessel-specific scoring;
- true Agatston calculation from lesion area and peak HU;
- visual overlays for review;
- differentiating coronary calcium from valve, aortic, mitral annular, or artifact calcium.

Without localization labels, the probe predicts category-level burden, not a measured
Agatston score in the classical software sense.

## First experiment checklist

1. Normalize the label table:

```text
study_id, patient_id, agatston, protocol, scanner, series_path
```

2. Convert gated CAC CT DICOM to NIfTI or another volume format.
3. Implement `Pillar0-ChestCT` preprocessing:

```text
CT -> canonical orientation -> 256^3 -> 11 CT windows -> [1, 11, 256, 256, 256]
```

4. Extract one 1152-d embedding per study and cache it.
5. Train linear baselines:

```text
any CAC
A0/A1/A2/A3
log1p(Agatston)
```

6. Calibrate class probabilities on validation data.
7. Evaluate patient-level held-out test performance.
8. Only after the gated baseline is understood, test on routine chest CT.

## Integration point in this repository

This should be a parallel research branch, not a modification of the abdomen pipeline.

Suggested future files:

```text
src/embeddings/pillar0_chest.py
scripts/60_extract_chest_cac_embeddings.py
scripts/61_train_cacdrs_probe.py
checkpoints/chest_cacdrs_probe.pt
docs/14_CACDRS_CHESTCT_PROBE.md
```

The existing abdomen probe code is the closest template:

- `scripts/41_merlin_rate_probe.py`
- `src/agent/pipeline.py`
- `docs/07_CALIBRATION.md`

## Open questions before implementation

- Are the gated CTs all non-contrast calcium-score protocol?
- Is there one Agatston value per study, or per vessel?
- Are there patient IDs for leakage-safe splitting?
- Are routine chest CTs available with matching CAC labels for external validation?
- Should the first target be risk category, numeric estimate, or both?
- Is the intended use research triage, report assistance, or quantitative scoring?

The conservative first target is:

```text
Gated CAC CT -> CAC present + A0/A1/A2/A3 category
```

Treat numeric Agatston as a secondary target until validation proves it is reliable.
