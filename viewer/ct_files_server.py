"""Standalone sidecar HTTP server that streams CTs + MerlinPlus masks and serves the
NiiVue viewer page used by the Streamlit dashboard (scripts/28).

Lives as its own process so the viewer is up the moment the user opens
http://localhost:8501 — streamlit's per-session lazy execution can't be relied on for
binding a separate port. Runs forever; exit with Ctrl-C / pkill.

Usage:
    python scripts/ct_files_server.py            # binds :8502, serves everything
    python scripts/ct_files_server.py --port 8512
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data import merlinplus as mp           # noqa: E402
from src.config import paths                    # noqa: E402

# IMPORTANT: dist/index.min.js on unpkg is a UMD bundle whose "ESM" export is just an
# encoded source blob — `import { Niivue }` returns an empty object and loadVolumes
# silently never resolves. esm.sh re-exports the real ESM build with proper named
# exports + automatic transitive-dep resolution.
NIIVUE_CDN = "https://esm.sh/@niivue/niivue@0.69.0"

# Confirmed colormap names in NiiVue 0.69 (single-hue + a couple of perceptual ones).
# Cycles if more than this many mask overlays are selected.
MASK_COLORMAPS = ["red", "green", "blue", "yellow", "cyan", "violet",
                  "warm", "plasma", "viridis", "magma", "hot", "jet"]

# CAM heatmap colormaps per encoder — distinct hues so side-by-side overlays read.
HEATMAP_COLORMAPS = {"merlin": "hot", "pillar0": "winter"}
HEAT_ROOT = paths.work_root / "heatmaps"
HEAT_ROOT.mkdir(parents=True, exist_ok=True)

# Per-(encoder, sid, finding) in-flight lock so two browser requests for the same
# overlay don't both spin up a forward pass.
_inflight: dict[tuple[str, str, str], threading.Lock] = {}
_inflight_lock = threading.Lock()


def _safe_organ(s: str) -> str: return re.sub(r"[^A-Za-z0-9_]", "", s)
def _safe_sid(s: str) -> str: return re.sub(r"[^A-Za-z0-9_-]", "", s)
def _safe_finding(s: str) -> str: return re.sub(r"[^A-Za-z0-9_]", "", s)
def _safe_encoder(s: str) -> str: return re.sub(r"[^a-z0-9]", "", s.lower())


def _heatmap_cache_path(encoder: str, sid: str, finding: str) -> Path:
    return HEAT_ROOT / encoder / sid / f"{finding}.nii.gz"


def _generate_heatmap(encoder: str, sid: str, finding: str) -> Path:
    """Generate one heatmap on demand; idempotent — return cache path if it already exists."""
    cache = _heatmap_cache_path(encoder, sid, finding)
    if cache.exists():
        return cache
    # only one in-flight generation per (encoder, sid, finding); others wait
    key = (encoder, sid, finding)
    with _inflight_lock:
        lock = _inflight.setdefault(key, threading.Lock())
    with lock:
        if cache.exists():
            return cache
        from src.explain import cam as CAM
        try:
            generated = CAM.ensure_concat_heatmap(
                sid, encoder, finding, heat_root=HEAT_ROOT, verbose=True
            )
            if generated is None:
                raise RuntimeError(f"failed to generate heatmap for {encoder}/{sid}/{finding}")
            cache = generated
        finally:
            # Release retained activations + fragmented cache regardless of
            # success/failure so a failed run doesn't snowball into OOM.
            import torch
            torch.cuda.empty_cache()
    return cache


def _stream_file(h: BaseHTTPRequestHandler, path: Path, ctype="application/octet-stream"):
    if not path.exists():
        h.send_response(404); h.end_headers(); return
    size = path.stat().st_size
    h.send_response(200)
    h.send_header("Content-Type", ctype)
    h.send_header("Content-Length", str(size))
    h.send_header("Access-Control-Allow-Origin", "*")
    h.send_header("Cache-Control", "public, max-age=300")
    h.end_headers()
    with path.open("rb") as f:
        while chunk := f.read(1 << 16):
            try:
                h.wfile.write(chunk)
            except (BrokenPipeError, ConnectionResetError):
                return


VIEWER_HTML = """<!doctype html>
<html><head><meta charset='utf-8'><title>CT-VLM viewer</title>
<style>
  html, body { margin: 0; height: 100%; background: #111; color: #ddd; font-family: sans-serif; }
  #gl { width: 100%; height: 100vh; display: block; }
  #status { position: absolute; top: 4px; left: 8px; font-size: 12px; opacity: 0.6; pointer-events:none; }
  .err { color: #f87171 !important; opacity: 1 !important; }
</style>
</head><body>
<canvas id='gl'></canvas>
<div id='status'>loading…</div>
<script type='module'>
  import { Niivue } from '__NIIVUE_CDN__';
  const params = new URLSearchParams(location.search);
  const sid = params.get('sid');
  const masks = (params.get('masks') || '').split(',').filter(Boolean);
  // heatmap param: comma-separated 'encoder:finding' pairs, e.g. 'merlin:pleural_effusion,pillar0:pleural_effusion'
  const heatmaps = (params.get('heatmap') || '').split(',').filter(Boolean).map(s => {
    const [encoder, finding] = s.split(':');
    return { encoder, finding };
  });
  const slice = parseInt(params.get('slice') || '3', 10);
  const heatOpacity = parseFloat(params.get('heatopacity') || '0.55');
  const status = document.getElementById('status');
  const err = (msg) => { status.classList.add('err'); status.textContent = msg; };
  if (!sid) { err('no study_id'); throw new Error('no sid'); }
  try {
    const nv = new Niivue({
      backColor: [0.07, 0.07, 0.07, 1],
      show3Dcrosshair: false,
      crosshairColor: [1, 1, 0, 0.6],
      dragMode: 1,
      isResizeCanvas: true,
      // Radiology convention: patient's L on viewer's R (axial flipped). NiiVue's
      // default is neurological (patient L on viewer L), which radiologists read wrong.
      isRadiologicalConvention: true,
    });
    await nv.attachTo('gl');
    nv.setSliceType(slice === 3 ? nv.sliceTypeMultiplanar :
                    slice === 0 ? nv.sliceTypeAxial :
                    slice === 1 ? nv.sliceTypeSagittal :
                    nv.sliceTypeCoronal);
    const CMS = __MASK_COLORMAPS__;
    const HEAT_CMS = __HEATMAP_COLORMAPS__;
    const vols = [{ url: `/ct/${sid}.nii.gz`, colormap: 'gray', opacity: 1.0 }];
    masks.forEach((organ, i) => {
      vols.push({
        url: `/mask/${sid}/${organ}.nii.gz`,
        colormap: CMS[i % CMS.length],
        opacity: 0.55,
      });
    });
    heatmaps.forEach(({ encoder, finding }) => {
      vols.push({
        url: `/heatmap/${encoder}/${sid}/${finding}.nii.gz`,
        colormap: HEAT_CMS[encoder] || 'hot',
        opacity: heatOpacity,
      });
    });
    status.textContent = `loading ${vols.length} volumes…`;
    if (heatmaps.length) status.textContent += ' (heatmap may take ~20s to generate on first request)';
    await nv.loadVolumes(vols);
    if (nv.volumes && nv.volumes.length > 0) {
      const v = nv.volumes[0];
      v.cal_min = -160; v.cal_max = 240;   // abdomen 400/40
      nv.updateGLVolume();
    }
    // Each heatmap volume: clamp its colormap to its own dynamic range so the
    // hot/winter palette saturates at the CAM hotspot rather than the global max.
    const heatStart = 1 + masks.length;
    for (let i = heatStart; i < (nv.volumes?.length || 0); i++) {
      const v = nv.volumes[i];
      v.cal_min = 0.55;       // hide low-attribution voxels (we percentile+minmax in save)
      v.cal_max = 1.0;
      nv.updateGLVolume();
    }
    status.textContent = '';
  } catch (e) { err(`viewer error: ${e.message || e}`); console.error(e); }
</script>
</body></html>"""


def _viewer_html() -> bytes:
    return (VIEWER_HTML
            .replace("__NIIVUE_CDN__", NIIVUE_CDN)
            .replace("__MASK_COLORMAPS__", json.dumps(MASK_COLORMAPS))
            .replace("__HEATMAP_COLORMAPS__", json.dumps(HEATMAP_COLORMAPS))).encode("utf-8")


class CTHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        # Single-line access log — handy when debugging the viewer.
        sys.stderr.write(f"[{self.log_date_time_string()}] {self.address_string()} "
                         f"{fmt % args}\n")

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/viewer.html"):
            html = _viewer_html()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            try: self.wfile.write(html)
            except (BrokenPipeError, ConnectionResetError): pass
            return
        m = re.fullmatch(r"/ct/([A-Za-z0-9_-]+)\.nii\.gz", path)
        if m:
            _stream_file(self, mp.ct_path(_safe_sid(m.group(1))), "application/gzip"); return
        m = re.fullmatch(r"/mask/([A-Za-z0-9_-]+)/([A-Za-z0-9_]+)\.nii\.gz", path)
        if m:
            sid = _safe_sid(m.group(1)); organ = _safe_organ(m.group(2))
            _stream_file(self, mp.case_dir(sid) / f"{organ}.nii.gz", "application/gzip"); return
        # /heatmap/<encoder>/<sid>/<finding>.nii.gz — lazy generation + cache
        m = re.fullmatch(r"/heatmap/([a-z0-9]+)/([A-Za-z0-9_-]+)/([A-Za-z0-9_]+)\.nii\.gz", path)
        if m:
            encoder = _safe_encoder(m.group(1))
            sid = _safe_sid(m.group(2))
            finding = _safe_finding(m.group(3))
            try:
                cache = _generate_heatmap(encoder, sid, finding)
            except FileNotFoundError as e:
                sys.stderr.write(f"[heatmap] 404 {encoder}/{sid}/{finding}: {e}\n")
                self.send_response(404); self.end_headers(); return
            except Exception as e:
                sys.stderr.write(f"[heatmap] 500 {encoder}/{sid}/{finding}: "
                                 f"{type(e).__name__}: {e}\n")
                self.send_response(500); self.end_headers(); return
            _stream_file(self, cache, "application/gzip"); return
        if path == "/heartbeat":
            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            try: self.wfile.write(b"ok")
            except (BrokenPipeError, ConnectionResetError): pass
            return
        self.send_response(404); self.end_headers()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8502)
    ap.add_argument("--bind", default="0.0.0.0")
    args = ap.parse_args()
    srv = ThreadingHTTPServer((args.bind, args.port), CTHandler)
    print(f"ct_files_server: serving NiiVue viewer + CT/mask streams on "
          f"http://{args.bind}:{args.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
