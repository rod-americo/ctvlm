# Viewer

Two co-running processes:

| | port | role |
|---|---|---|
| `streamlit_app.py` | 8501 | Streamlit dashboard: case browser, finding filter, probability table, report panel, label-override capture |
| `ct_files_server.py` | 8502 | Sidecar HTTP server: streams CT NIfTI + organ-mask NIfTI + Grad-CAM heatmap NIfTI to NiiVue; also serves the NiiVue viewer HTML page |

The two are decoupled because Streamlit's per-session execution can't reliably bind a separate socket — the file server runs as its own long-lived process.

## Run (Docker — recommended)

```bash
docker compose up viewer
# → http://localhost:8501
```

## Run (without Docker)

In two shells:

```bash
# Shell A
python viewer/ct_files_server.py --port 8502

# Shell B
streamlit run viewer/streamlit_app.py \
    --server.address 0.0.0.0 --server.port 8501
```

## What the viewer needs on disk

| input | path (default) | source |
|---|---|---|
| CT volumes | `$CTVLM_MERLIN_ROOT/AC<sid>.nii.gz` | DICOM → NIfTI conversion |
| Organ masks (optional, for the mask multi-select) | `$CTVLM_MERLIN_PLUS_DIR/AC<sid>/<organ>.nii.gz` | TotalSegmentator / MerlinPlus segmentations |
| Encoder features | `$CTVLM_WORK_ROOT/{merlin_global,pillar0_emb}/AC<sid>.npy` | Auto-extracted on first inference call |
| Grad-CAM heatmaps | `$CTVLM_WORK_ROOT/heatmaps/<encoder>/AC<sid>/<finding>.nii.gz` | Lazily generated on click |
| Probe checkpoint | `$CTVLM_CHECKPOINTS_DIR/concat_rate_probe.pt` | Ships with this repo under `checkpoints/` |

If only the encoder features are cached (e.g. the bundled `ctvlm_features_25k_*.tar`), the viewer can still render the probability table; the NiiVue viewer just won't have a CT to render.

## Cross-origin / CORS

The file server sends `Access-Control-Allow-Origin: *`. If you embed the dashboard in a different host, point the iframe's `src` at `http://<host>:8502/viewer.html?...`. The query string takes:

| param | example | meaning |
|---|---|---|
| `sid` | `AC421363f` | study ID; resolves CT path |
| `masks` | `liver,spleen,kidney_left,kidney_right` | comma-list of organ masks to overlay |
| `heatmap` | `merlin:hepatic_cyst,pillar0:hepatic_cyst` | per-encoder heatmap overlay |
| `slice` | `261` | axial slice to jump to (1-indexed) |

## Security note

The bundled file server has **no authentication**. Do not expose it to a public network. In production deployments, put it behind your existing auth layer (reverse proxy with auth headers, VPN-only access, etc.).
