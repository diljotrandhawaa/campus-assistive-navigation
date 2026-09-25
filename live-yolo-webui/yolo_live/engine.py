"""Shared YOLO, per-camera BoT-SORT tracking, and optional OCR."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import logging
import time

from .direction import CLASSES, position
from .tracking import CameraTracker


LOG = logging.getLogger("yolo_live")


def decode_jpeg(data):
    import numpy as np
    from PIL import Image

    with Image.open(BytesIO(data)) as source:
        width, height = source.size
        if (
            source.format != "JPEG"
            or min(width, height) < 32
            or max(width, height) > 2048
            or width * height > 2_100_000
        ):
            raise ValueError(
                "Send a JPEG of at most 2048 pixels per side "
                "and 2.1 megapixels."
            )

        return np.asarray(source.convert("RGB"))[:, :, ::-1].copy()


class YOLOBackend:
    def __init__(self, weights, device, imgsz, ocr=None):
        self.weights = weights
        self.device = device
        self.imgsz = imgsz
        self.ocr = ocr
        self.model = None

    def load(self):
        import numpy as np
        import torch
        from ultralytics import YOLOE

        if self.device != "cpu" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable. Activate the working indoor-nav environment."
            )

        self.model = YOLOE(self.weights)
        self.model.set_classes(list(CLASSES))
        self.model.predict(
            np.zeros((480, 640, 3), dtype=np.uint8),
            device=self.device,
            imgsz=self.imgsz,
            conf=0.25,
            verbose=False,
        )

        # Fail at startup if the tracker dependency is missing, not after the
        # browser has opened its camera. ReID is disabled: no extra AI weights.
        from ultralytics.trackers.bot_sort import BOTSORT  # noqa: F401

        if self.ocr is not None:
            LOG.info("Loading and warming up OCR...")
            self.ocr.load()

    def infer(self, jpeg, confidence, run_ocr=False, target=None, tracker=None):
        started = time.perf_counter()
        image = decode_jpeg(jpeg)
        height, width = image.shape[:2]

        result = self.model.predict(
            source=image,
            device=self.device,
            imgsz=self.imgsz,
            conf=min(confidence, CameraTracker.low_confidence) if tracker is not None else confidence,
            max_det=60,
            verbose=False,
        )[0]

        boxes = result.boxes.cpu().numpy()
        assignments, tracking = {}, None
        tracking_started = time.perf_counter()
        if tracker is not None:
            assignments, tracking = tracker.update(
                boxes, image, result.names, target, confidence, time.monotonic(),
            )
        tracking_ms = (time.perf_counter() - tracking_started) * 1000

        detections = []
        for index, row in enumerate(boxes.data.tolist()):
            x1, y1, x2, y2, score, class_id = row[:6]
            track_id = assignments.get(index)
            # Keep the usual UI cutoff plus current tracked target observations
            # below that cutoff, so confidence dips can preserve the lock.
            if score < confidence and track_id is None:
                continue
            box = [
                max(0, min(1, x1 / width)),
                max(0, min(1, y1 / height)),
                max(0, min(1, x2 / width)),
                max(0, min(1, y2 / height)),
            ]
            detections.append({
                "label": result.names[int(class_id)],
                "confidence": round(score, 4),
                "box": box,
                "position": position(box),
                "track_id": track_id,
            })

        response = {
            "detections": detections,
            "width": width,
            "height": height,
            "detector_ms": round((time.perf_counter() - started) * 1000, 1),
            "model_inference_ms": round(
                float(result.speed.get("inference", 0)), 1,
            ),
            "tracking_ms": round(tracking_ms, 1),
            "tracking": tracking,
            "ocr": {
                "status": "skipped" if self.ocr is not None else "disabled",
            },
        }

        if run_ocr and self.ocr is not None:
            try:
                response["ocr"] = self.ocr.infer(image, detections)
            except Exception:
                LOG.exception("OCR failed; returning YOLO detections")
                response["ocr"] = {
                    "status": "error",
                    "message": "OCR failed. Check the GB10 terminal.",
                }

        return response


class SharedDetector:
    def __init__(self, backend):
        self.backend = backend
        self.lock = asyncio.Lock()
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="yolo-gpu",
        )

    async def load(self):
        await asyncio.get_running_loop().run_in_executor(
            self.executor, self.backend.load,
        )

    async def try_infer(
        self, jpeg, confidence, run_ocr=False, target=None, tracker=None,
    ):
        if self.lock.locked():
            return None

        async with self.lock:
            job = asyncio.get_running_loop().run_in_executor(
                self.executor,
                self.backend.infer,
                jpeg,
                confidence,
                run_ocr,
                target,
                tracker,
            )
            try:
                return await asyncio.shield(job)
            except asyncio.CancelledError:
                try:
                    await asyncio.shield(job)
                finally:
                    raise

    async def close(self):
        await asyncio.to_thread(
            self.executor.shutdown,
            wait=True,
            cancel_futures=True,
        )
