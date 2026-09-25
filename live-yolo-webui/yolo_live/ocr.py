"""PP-OCRv5 on PyTorch. Called only by the shared inference worker."""
import math
import time

from .direction import position

DET_MODEL = "PaddlePaddle/PP-OCRv5_server_det_safetensors"
REC_MODEL = "PaddlePaddle/PP-OCRv5_server_rec_safetensors"


def crop_text(rgb, raw_box):
    """Accept XYXY or four corners; rectify a text line from the received image."""
    import cv2
    import numpy as np

    h, w = rgb.shape[:2]
    points = np.asarray(raw_box, dtype=np.float32)
    if points.shape == (4,):
        x1, y1, x2, y2 = points
        points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError("Unexpected OCR box shape or coordinates")
    points[:, 0] = np.clip(points[:, 0], 0, w - 1)
    points[:, 1] = np.clip(points[:, 1], 0, h - 1)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    points = points[np.argsort(angles)]
    points = np.roll(points, -int(np.argmin(points.sum(axis=1))), axis=0)
    tl, tr, br, bl = points
    width = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    height = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    if min(width, height) < 3 or abs(cv2.contourArea(points)) < 9:
        return None
    destination = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1],
                            [0, height - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(points, destination)
    crop = cv2.warpPerspective(rgb, matrix, (width, height),
                               flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    if height > width * 1.5:
        crop = np.rot90(crop).copy()
    polygon = (points / np.array([w, h])).tolist()
    box = [float(points[:, 0].min() / w), float(points[:, 1].min() / h),
           float(points[:, 0].max() / w), float(points[:, 1].max() / h)]
    return crop, box, polygon


def nearby_object(text_box, detections):
    """Image-space proximity only; never establishes a room's identity."""
    x = (text_box[0] + text_box[2]) / 2
    y = (text_box[1] + text_box[3]) / 2
    candidates = []
    for index, item in enumerate(detections):
        label = item["label"]
        if not any(word in label for word in ("door", "sign", "elevator", "whiteboard")):
            continue
        a, b, c, d = item["box"]
        gap = math.hypot(max(a - x, 0, x - c), max(b - y, 0, y - d))
        if gap <= 0.06:
            candidates.append((gap, (c - a) * (d - b), index))
    if not candidates:
        return None
    index = min(candidates)[2]
    return {"index": index, "label": detections[index]["label"], "relation": "near in image"}


class OCRBackend:
    def __init__(self, device, min_score=0.75, max_regions=24):
        self.device = f"cuda:{device}" if str(device).isdecimal() else str(device)
        self.min_score = min_score
        self.max_regions = max_regions

    def load(self):
        import numpy as np
        from transformers import (AutoImageProcessor, PPOCRV5ServerDetForObjectDetection,
                                  PPOCRV5ServerRecForTextRecognition)

        self.det_processor = AutoImageProcessor.from_pretrained(DET_MODEL)
        self.rec_processor = AutoImageProcessor.from_pretrained(REC_MODEL)
        self.detector = PPOCRV5ServerDetForObjectDetection.from_pretrained(DET_MODEL).to(self.device).eval()
        self.recognizer = PPOCRV5ServerRecForTextRecognition.from_pretrained(REC_MODEL).to(self.device).eval()
        # Exercise both model kernels and postprocessors before opening the UI.
        self.infer(np.full((320, 640, 3), 255, dtype=np.uint8), [])
        self._recognize([np.full((48, 192, 3), 255, dtype=np.uint8)])

    def _recognize(self, crops):
        import torch

        with torch.inference_mode():
            inputs = self.rec_processor(images=crops, return_tensors="pt").to(self.device)
            output = self.recognizer(**inputs)
            return self.rec_processor.post_process_text_recognition(output)

    def infer(self, bgr, detections):
        import numpy as np
        import torch

        started = time.perf_counter()
        rgb = bgr[:, :, ::-1].copy()
        with torch.inference_mode():
            inputs = self.det_processor(images=rgb, return_tensors="pt").to(self.device)
            output = self.detector(**inputs)
            result = self.det_processor.post_process_object_detection(
                output, target_sizes=inputs["target_sizes"])[0]
        boxes = result["boxes"].detach().cpu().tolist()
        scores = result["scores"].detach().cpu().tolist()
        ranked = sorted(zip(boxes, scores), key=lambda entry: entry[1], reverse=True)
        regions = []
        for raw_box, score in ranked:
            if not math.isfinite(score):
                continue
            cropped = crop_text(rgb, raw_box)
            if cropped is not None:
                regions.append((*cropped, score))
            if len(regions) >= self.max_regions:
                break
        items = []
        for start in range(0, len(regions), 8):
            batch = regions[start:start + 8]
            readings = self._recognize([region[0] for region in batch])
            if len(readings) != len(batch):
                raise RuntimeError("OCR returned a different number of readings than crops")
            for (_, box, polygon, det_score), reading in zip(batch, readings):
                text = str(reading["text"]).strip()
                confidence = float(reading["score"])
                if not text or not math.isfinite(confidence) or confidence < self.min_score:
                    continue
                items.append({"text": text, "confidence": round(confidence, 4),
                              "detection_confidence": round(float(det_score), 4),
                              "box": box, "polygon": polygon, "position": position(box),
                              "nearby_object": nearby_object(box, detections)})
        items.sort(key=lambda item: (item["box"][1], item["box"][0]))
        if self.device.startswith("cuda"):
            torch.cuda.synchronize(self.device)
        return {"status": "ok", "items": items,
                "ms": round((time.perf_counter() - started) * 1000, 1),
                "regions_detected": len(boxes), "regions_processed": len(regions),
                "truncated": len(boxes) > len(regions)}
