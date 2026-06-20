# Thor deployment

This profile is for running ctvlm on `thor` with Docker Compose and an NVIDIA GPU.

## What this runs

- `viewer`: Streamlit UI on `127.0.0.1:8501` and NIfTI/heatmap file server on `127.0.0.1:8502`.
- `worker`: reference GPU worker. Keep it stopped until the broker integration in `deploy/example_worker.py` is replaced or wired to your queue.
- Persistent host data defaults to `/srv/ctvlm`.

The ports bind to localhost by default because the file server has no authentication. Expose it through your existing VPN/reverse proxy/auth layer, not directly to the public internet.

## Prerequisites on thor

- Docker Engine with Compose v2.
- NVIDIA driver and NVIDIA Container Toolkit.
- A GPU with 24 GB VRAM recommended.
- Hugging Face token whose account has accepted:
  - `stanfordmimi/Merlin`
  - `YalaLab/Pillar0-AbdomenCT`

## First deploy

```bash
sudo mkdir -p /srv/ctvlm/{work,hf_cache}
sudo chown -R "$USER":"$USER" /srv/ctvlm

cp .env.example .env
$EDITOR .env

docker compose -f docker-compose.thor.yml build viewer worker
docker compose -f docker-compose.thor.yml up -d viewer
docker compose -f docker-compose.thor.yml logs -f viewer
```

First start downloads encoder weights into `/srv/ctvlm/hf_cache`. Later starts reuse that cache.

If `rad-vision-engine` is installed on the host and mounted into the container, set
`RAVE_WIN` in `.env` to its `windowing_utils.py`. If unset, ctvlm uses the bundled
CT-window fallback so the container remains self-contained.

## Smoke checks

```bash
docker compose -f docker-compose.thor.yml ps
curl -sI http://127.0.0.1:8501/
curl -sI http://127.0.0.1:8502/
docker compose -f docker-compose.thor.yml exec viewer python scripts/smoke_test.py
```

For a real uploaded DICOM series, confirm that the worker or viewer writes:

```text
/srv/ctvlm/work/ct_volumes/<sid>.nii.gz
/srv/ctvlm/work/merlin_global/<sid>.npy
/srv/ctvlm/work/pillar0_emb/<sid>.npy
/srv/ctvlm/work/heatmaps/<encoder>/<sid>/<finding>.nii.gz
```

## Update

```bash
git pull --ff-only
docker compose -f docker-compose.thor.yml build viewer worker
docker compose -f docker-compose.thor.yml up -d viewer
```

## Rollback

```bash
git checkout <previous-good-commit>
docker compose -f docker-compose.thor.yml build viewer worker
docker compose -f docker-compose.thor.yml up -d viewer
```

Keep `/srv/ctvlm/hf_cache` and `/srv/ctvlm/work` intact during rollback; they are runtime state, not app code.
