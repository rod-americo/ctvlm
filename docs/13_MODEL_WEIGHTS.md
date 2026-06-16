# 13 — Model weights

The deployment bundle ships **three** sets of weights:

| weight set | path in bundle | size | role |
|---|---|---|---|
| Concat probe | `checkpoints/concat_rate_probe.pt` | 2.8 MB | 3200→220 linear head + Platt + Youden-J + NC calibration |
| Merlin (`stanfordmimi/Merlin`) | `weights/merlin/i3_resnet_clinical_longformer_best_clip_04-02-2024_23-21-36_epoch_99.pt` | 1.1 GB | 3D I3D-inflated ResNet-152 abdomen-CT image-text encoder (MIT license) |
| Pillar-0 (`YalaLab/Pillar0-AbdomenCT`) | `weights/pillar0_hf_cache/` | 356 MB | YalaLab Atlas abdomen-CT encoder, mirrored HF cache layout |

Bundling the encoder weights lets the worker start **offline** — no HF token required, no network dependency for cold boot.

## Where to put them on the worker

After extracting `ctvlm-deploy.zip` to `/opt/ctvlm/` (or wherever your repo lives), copy the weights into the locations the encoders expect:

```bash
EXTRACT=/opt/ctvlm
# 1. The concat probe — env var CTVLM_CHECKPOINTS_DIR should point here
mkdir -p /opt/ctvlm/models
cp $EXTRACT/checkpoints/concat_rate_probe.pt /opt/ctvlm/models/

# 2. Merlin — drops into the merlin Python package's local checkpoint dir
MERLIN_PKG=$(python -c "import merlin, pathlib; print(pathlib.Path(merlin.__file__).parent)")
mkdir -p $MERLIN_PKG/models/checkpoints
cp $EXTRACT/weights/merlin/*.pt $MERLIN_PKG/models/checkpoints/

# 3. Pillar-0 — mirror the HF cache layout into HF_HOME
mkdir -p /opt/ctvlm/hf_cache/hub
cp -r $EXTRACT/weights/pillar0_hf_cache/models--YalaLab--Pillar0-AbdomenCT \
      /opt/ctvlm/hf_cache/hub/

# 4. Export the env vars (or set in systemd unit / Docker env)
export CTVLM_CHECKPOINTS_DIR=/opt/ctvlm/models
export HF_HOME=/opt/ctvlm/hf_cache
export CTVLM_HF_CACHE=/opt/ctvlm/hf_cache
```

After that, `from src.embeddings import merlin, pillar0; merlin.load_model(); pillar0.load_model()` finds local files first and skips the download.

## Verifying weights loaded from disk (not network)

```bash
# Block egress to HF and confirm the encoders still load
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -c "
from src.embeddings import merlin, pillar0
merlin.load_model()
pillar0.load_model()
print('OK — both encoders loaded offline')
"
```

If this prints `OK`, the deployment is self-contained.

## What's NOT in the bundle (intentionally)

- **MedGemma-4B-it + LoRA adapter**: the LLM prose layer is *disabled in production* (see [docs/01_OVERVIEW.md](01_OVERVIEW.md) "Why no LLM"). If you ever want to re-enable it for research, the base model lives at `google/medgemma-4b-it` on HF and the LoRA adapter is at `$CTVLM_CHECKPOINTS_DIR/hybrid_medgemma-4b-it_text/lora/` in the dev environment (~3.5 GB combined; not shipped to keep the production bundle lean).
- **Clinical-Longformer** (Merlin's text tower): only used for text-side embeddings in the report-generation research path. Not loaded by the inference pipeline. Auto-downloads if you ever import the text-tower branch.

## License notes

- **Merlin** (`stanfordmimi/Merlin`): MIT, redistributable.
- **Pillar-0** (`YalaLab/Pillar0-AbdomenCT`): permissive (the recipient should still review the upstream model card before redistributing beyond their organisation).
- **Concat probe** (`concat_rate_probe.pt`): trained on the Merlin Abdominal CT dataset; downstream use should respect the source data license. See [docs/11_LIMITATIONS.md](11_LIMITATIONS.md) for the regulatory-status statement.

The bundle's weights are sufficient for a self-contained worker — no internet access required at runtime once installed.
