# Campus YOLOE + OCR update

This package adds PP-OCRv5 Server text detection and recognition to the uploaded
live-yolo-webui application. Both OCR models run in PyTorch on the same selected
device as YOLOE. No PaddlePaddle runtime or cloud OCR API is used.

## Install on your GB10

1. Stop the running UI with Ctrl+C. Activate `conda activate indoor-nav`.
2. Upload `campus-ocr-integration.tar.gz` into `~/indoor-nav/live-yolo-webui`.
3. From that project folder, run:

```bash
mkdir -p /tmp/campus-ocr-update
tar -xzf campus-ocr-integration.tar.gz -C /tmp/campus-ocr-update
python /tmp/campus-ocr-update/install_ocr_update.py "$PWD"
```

The installer verifies the original files against the versions you uploaded.
If they changed in the meantime, it stops without editing anything. It backs up
changed files to `~/indoor-nav/campus-ocr-backups/<timestamp>/`. It preserves all
unrelated project files. No Git commit or upload is performed.

You already installed Transformers and verified the two OCR imports. If setting
up a different machine, install `python -m pip install -r requirements-ocr.txt`
inside the working YOLO environment, keeping its existing GPU torch/torchvision.

## Launch

```bash
python run_yolo.py --port 8091 --http --weights yoloe-26l-seg.pt --imgsz 640 --device 0
```

Open the same forwarded address, http://localhost:8091. Hard-refresh the page
(Ctrl+Shift+R), click Start camera, and point at a clear sign. OCR is enabled by
default. Watch the **Text from latest OCR scan** panel below the camera section.
Initial startup downloads missing assets and exercises both OCR models and
postprocessors before the camera server starts accepting connections.

A faster YOLO-only fallback, which doesn't load OCR models:

```bash
python run_yolo.py --port 8091 --http --weights yoloe-26l-seg.pt --imgsz 640 --device 0 --no-ocr
```

The existing HTTPS/certificate flags and localhost restriction on `--http` remain.

## What changed

- `yolo_live/ocr.py`: text detection, perspective-corrected crops, batched text
  recognition, score filtering, normalized text polygons, and processing time.
- `engine.py`: startup loads OCR once; YOLO and OCR use the existing single worker.
  OCR failures are logged and returned separately while YOLO detections continue.
- `server.py`: periodic OCR scheduling per connection, an OCR toggle in settings,
  and startup options. Default weights now match the existing YOLOE loader.
- `direction.py`: fixed missing commas after `classroom door` and
  `handicap push button`; removed duplicate labels; corrected `revolving door`;
  added `elevator door` to match the existing intended UI target.
- Frontend: dynamic target list, OCR controls, text snapshot/polygons, recognition
  scores, location in image, and OCR processing time/age. Sent image defaults to
  1280 pixels and JPEG quality 0.9 with OCR enabled.
- `sp2text.py` was empty in the upload. No speech recognition or speech synthesis
  is added by this update. Existing camera direction still follows the selected
  YOLO object class; text doesn't select or verify a destination yet.

## Timing and freshness

- One frame per client is in flight. No background frame queue is added.
- OCR runs on the next accepted frame at least 1 second after the previous OCR
  completes. It is not guaranteed to run once per second under load.
- YOLO and OCR run sequentially on an OCR-sampled frame. That response takes
  longer; intervening frames run YOLO only. We do not promise unchanged FPS.
- The original 1.5-second freshness limit for camera guidance is retained.
- OCR snapshots show their capture age and disappear at 5 seconds, on pause,
  disconnect, settings changes, or an OCR error. Text from an old snapshot is
  never drawn over a newer YOLO frame. This 5-second display window is history,
  not proof that a sign is still in the live camera view.
- Text is cropped from the full **received** frame (default up to 1280 pixels),
  not YOLO's 640-pixel input. Increasing transmitted resolution affects bandwidth.
- At most 24 text crops are recognized per scan, in batches of 8. Extra regions
  are skipped, and the UI indicates this. Dense whiteboards may need a larger cap.
- `detector_ms` remains decode + YOLO + result extraction. `ocr.ms` reports the
  OCR stage; `server_ms` includes both. Frame-to-overlay includes the round trip.

Tune the startup flags only after measuring on GB10:

```bash
# Less frequent scans; lower recognition score threshold if needed.
python run_yolo.py --port 8091 --http --device 0 --ocr-interval 2 --ocr-confidence 0.70
```

Other option: `--ocr-max-regions 40`. Higher limits can increase latency.
A checkbox disables OCR scans without unloading models; `--no-ocr` avoids loading.

## What the text means

Text detection runs over the entire image, even if YOLO misses a sign. OCR returns
text and recognition scores. Blank, nonfinite, and below-threshold readings are
removed. Perspective correction helps with skew; it cannot guarantee text recovery
from blur, glare, tiny lettering, upside-down text, or extreme viewpoints.

“Near door in image” is a geometric heuristic using YOLO objects from the **same
frame**, not a verified room-to-door mapping. No repeated-frame confirmation,
route planning, distance estimation, or room-number target guidance is claimed.
The UI renders OCR text as textContent, never HTML.

## Validation included

```bash
python ocr_checks/test_integration.py
node ocr_checks/test_ui.cjs
```

CPU checks cover crop geometry, recognition filtering and limits, OCR errors,
serialized GPU scheduling/cancellation, WebSocket scheduling/toggles/revisions,
session isolation, and repaired class names. The model interfaces are mocked in
these checks; they do not measure real OCR accuracy. UI checks cover safe text
rendering, snapshot pairing/expiry, stale guidance, errors and class synchronization.

The package was checked for Python and JavaScript syntax. Real PP-OCRv5 inference,
GPU compatibility, throughput and camera accuracy have not been tested here.
Startup warmup plus your first live run is the next check on your GB10. Compare a
stationary room sign, a skewed sign, no text, and a moving camera. Toggle OCR off
and on to measure its effect. If startup or scanning fails, send the traceback.

## Rollback

For a quick demo fallback, launch with `--no-ocr`. To restore the exact old code,
stop the server and copy the files in `backup-info.json`'s `existing` list from the
printed backup folder back into the project. Files in `added` were introduced by
this package; remove them only if you want a full rollback. Preserve later edits.

## Reference APIs

- https://huggingface.co/docs/transformers/model_doc/pp_ocrv5_server_det
- https://huggingface.co/docs/transformers/model_doc/pp_ocrv5_server_rec
- https://huggingface.co/PaddlePaddle/PP-OCRv5_server_det_safetensors
- https://huggingface.co/PaddlePaddle/PP-OCRv5_server_rec_safetensors
