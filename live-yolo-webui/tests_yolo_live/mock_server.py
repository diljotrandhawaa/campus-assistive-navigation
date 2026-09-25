"""Browser testing only. Mock boxes, no actual detection. Never use for navigation."""
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aiohttp import web
from yolo_live.direction import position
from yolo_live.engine import decode_jpeg, SharedDetector
from yolo_live.server import create_app


class MockBackend:
    def load(self):
        pass

    def infer(self, data, confidence):
        # Test runner switches modes through a file, without exposing production routes.
        mode_file = Path("/tmp/yolo-ui-test-mode")
        mode = mode_file.read_text().strip() if mode_file.exists() else "right"
        if mode == "slow":
            time.sleep(1.8)
        if mode == "error":
            raise RuntimeError("Intentional test exception")
        image = decode_jpeg(data)
        h, w = image.shape[:2]
        cx = {"left": .3, "centered": .5}.get(mode, .7)
        box = [cx - .15, .25, cx + .15, .6]
        objects = [] if mode == "lost" else [{"label": "whiteboard", "confidence": .84,
                                               "box": box, "position": position(box)}]
        return {"width": w, "height": h, "detections": objects, "detector_ms": 20.0,
                "model_inference_ms": 15.0}


if __name__ == "__main__":
    print("TEST SERVER: all detections are synthetic; not YOLO.", flush=True)
    web.run_app(create_app(SharedDetector(MockBackend()),
                          {"model": "TEST ONLY / simulated detections", "device": "mock", "imgsz": 640}),
                host="127.0.0.1", port=18091)
