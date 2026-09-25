"""CPU integration checks; actual model accuracy/GB10 timing are separate checks."""
import asyncio
from contextlib import nullcontext
from io import BytesIO
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from aiohttp.test_utils import TestClient, TestServer
from yolo_live.direction import CLASSES
from yolo_live.engine import YOLOBackend, SharedDetector
from yolo_live.ocr import OCRBackend, crop_text, nearby_object
from yolo_live.server import create_app


class Tensor:
    def __init__(self, value): self.value = value
    def detach(self): return self
    def cpu(self): return self
    def tolist(self): return self.value


class Batch(dict):
    def to(self, device): return self


def jpeg():
    buffer = BytesIO()
    Image.new('RGB', (100, 80), (255, 0, 0)).save(buffer, 'JPEG')
    return buffer.getvalue()


class GeometryTests(unittest.TestCase):
    def test_rotated_quad_and_xyxy_are_valid(self):
        image = np.full((100, 200, 3), 255, dtype=np.uint8)
        for box in ([10, 20, 150, 50], [[15, 10], [170, 35], [165, 65], [10, 40]]):
            crop, normalized, polygon = crop_text(image, box)
            self.assertGreater(crop.shape[1], crop.shape[0])
            self.assertTrue(all(0 <= value <= 1 for point in polygon for value in point))
            self.assertEqual(len(normalized), 4)
        self.assertIsNone(crop_text(image, [10, 10, 11, 11]))
        with self.assertRaises(ValueError): crop_text(image, [1, 2, 3])

    def test_proximity_not_unconditional_association(self):
        objects = [{'label': 'door', 'box': [.1, .1, .5, .9]}]
        self.assertEqual(nearby_object([.15, .2, .3, .3], objects)['index'], 0)
        self.assertIsNone(nearby_object([.8, .1, .9, .2], objects))

    def test_class_labels_repaired(self):
        for label in ('classroom door', 'doorknob', 'handicap push button', 'elevator', 'elevator door'):
            self.assertIn(label, CLASSES)
        self.assertEqual(len(CLASSES), len(set(CLASSES)))


class BackendTests(unittest.TestCase):
    def test_ocr_filtering_no_text_and_region_cap(self):
        backend = OCRBackend('cpu', min_score=.75, max_regions=3)
        box = [[2, 2], [60, 2], [60, 20], [2, 20]]
        prediction = {'boxes': Tensor([box] * 4), 'scores': Tensor([.99, .98, .97, .96])}
        backend.det_processor = SimpleNamespace()
        class Processor:
            def __call__(self, **kwargs):
                # OCR must get RGB; the supplied test image is red in BGR order.
                self.color = kwargs['images'][0, 0].tolist()
                return Batch(pixel_values='pixels', target_sizes='sizes')
            def post_process_object_detection(self, output, target_sizes):
                self.target_sizes = target_sizes
                return prediction
        class ListProcessor(Processor):
            def post_process_object_detection(self, output, target_sizes):
                self.target_sizes = target_sizes
                return [prediction]
        backend.det_processor = ListProcessor()
        backend.detector = lambda **kwargs: None
        backend._recognize = lambda crops: [{'text': '204', 'score': .95},
                                             {'text': 'bad', 'score': .4},
                                             {'text': '', 'score': float('nan')}][:len(crops)]
        bgr = np.zeros((80, 100, 3), dtype=np.uint8); bgr[:, :, 2] = 255
        with patch.dict(sys.modules, {'torch': SimpleNamespace(inference_mode=nullcontext)}):
            result = backend.infer(bgr, [])
            self.assertEqual(backend.det_processor.color, [255, 0, 0])
            self.assertEqual(result['regions_processed'], 3)
            self.assertTrue(result['truncated'])
            self.assertEqual([item['text'] for item in result['items']], ['204'])
            prediction['boxes'], prediction['scores'] = Tensor([]), Tensor([])
            self.assertEqual(backend.infer(bgr, [])['items'], [])

    def test_yolo_continues_after_ocr_failure_and_toggle_skips(self):
        class BrokenOCR:
            def infer(self, *args): raise RuntimeError('simulated failure')
        backend = YOLOBackend('fake.pt', 'cpu', 640, BrokenOCR())
        result = SimpleNamespace(boxes=SimpleNamespace(data=Tensor([[10, 10, 50, 70, .9, 0]])),
                                 names={0: 'door'}, speed={'inference': 1})
        backend.model = SimpleNamespace(predict=lambda **kwargs: [result])
        self.assertEqual(backend.infer(jpeg(), .25, False)['ocr']['status'], 'skipped')
        with self.assertLogs('yolo_live', level='ERROR'):
            response = backend.infer(jpeg(), .25, True)
        self.assertEqual(response['ocr']['status'], 'error')
        self.assertEqual(response['detections'][0]['label'], 'door')


class WorkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_busy_and_cancel_keep_gpu_serialized(self):
        entered, finish = threading.Event(), threading.Event()
        class Backend:
            def infer(self, *args):
                entered.set(); finish.wait(timeout=3); return {'done': True}
        worker = SharedDetector(Backend())
        task = asyncio.create_task(worker.try_infer(b'x', .25, True))
        try:
            await asyncio.to_thread(entered.wait, 2)
            self.assertIsNone(await worker.try_infer(b'y', .25))
            task.cancel()
            await asyncio.sleep(.01)
            self.assertTrue(worker.lock.locked())
            finish.set()
            with self.assertRaises(asyncio.CancelledError): await task
            self.assertFalse(worker.lock.locked())
        finally:
            finish.set(); await worker.close()


class WebSocketTests(unittest.IsolatedAsyncioTestCase):
    async def test_schedule_toggle_stale_revision_and_session_isolation(self):
        calls = []
        class FakeDetector:
            async def load(self): pass
            async def close(self): pass
            async def try_infer(self, payload, confidence, run_ocr):
                calls.append(run_ocr)
                return {'detections': [], 'width': 100, 'height': 80, 'detector_ms': 2,
                        'ocr': {'status': 'ok' if run_ocr else 'skipped', 'items': [], 'ms': 3}}
        client = TestClient(TestServer(create_app(FakeDetector(), {'ocr_enabled': True, 'ocr_interval': 30})))
        await client.start_server()
        async def configure(ws, revision, enabled=True):
            await ws.send_json({'type': 'configure', 'target': 'door', 'confidence': .25,
                                'revision': revision, 'ocr': enabled})
            self.assertEqual((await ws.receive_json())['type'], 'configured')
        async def frame(ws, index, revision):
            await ws.send_json({'type': 'frame', 'id': index, 'revision': revision})
            await ws.send_bytes(jpeg())
            return await ws.receive_json()
        try:
            ws = await client.ws_connect('/ws'); await ws.receive_json()
            await configure(ws, 1)
            first = await frame(ws, 1, 1)
            self.assertEqual(first['ocr']['status'], 'ok')
            self.assertEqual((await frame(ws, 2, 1))['ocr']['status'], 'skipped')
            self.assertEqual((await frame(ws, 3, 0))['type'], 'discarded')
            self.assertEqual(calls, [True, False])
            await configure(ws, 2, False)
            self.assertEqual((await frame(ws, 4, 2))['ocr']['status'], 'disabled')
            ws2 = await client.ws_connect('/ws'); await ws2.receive_json()
            await configure(ws2, 1)
            self.assertEqual((await frame(ws2, 1, 1))['ocr']['status'], 'ok')
            await ws.close(); await ws2.close()
        finally:
            await client.close()


if __name__ == '__main__': unittest.main(verbosity=2)
