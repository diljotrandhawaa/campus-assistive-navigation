"""Run from the add-on folder: python -m unittest discover -s tests_yolo_live -v"""
import asyncio
from io import BytesIO
import threading
import unittest

from aiohttp.test_utils import TestClient, TestServer
from PIL import Image

from yolo_live.direction import AlignmentController, position
from yolo_live.engine import SharedDetector, decode_jpeg
from yolo_live.server import create_app, parse_settings


def detection(x, label="whiteboard", confidence=0.8):
    return {"label": label, "confidence": confidence,
            "box": [x - 0.15, 0.25, x + 0.15, 0.65], "position": "middle-center"}


def jpeg_bytes():
    output = BytesIO()
    Image.new("RGB", (320, 240), "white").save(output, "JPEG")
    return output.getvalue()


class DirectionTests(unittest.TestCase):
    def test_pan_right_center_then_left(self):
        controller = AlignmentController()
        for time, x, state in [(0, .7, "right"), (.1, .61, "right"), (.2, .52, "centered"),
                               (.3, .5, "centered"), (.4, .38, "left")]:
            self.assertEqual(controller.update([detection(x)], time)["state"], state)

    def test_hysteresis_does_not_flicker_at_five_percent(self):
        controller = AlignmentController()
        self.assertEqual(controller.update([detection(.5)], 0)["state"], "centered")
        for time, x in enumerate([.549, .551, .565, .535], 1):
            self.assertEqual(controller.update([detection(x)], time * .1)["state"], "centered")
        self.assertEqual(controller.update([detection(.61)], .6)["state"], "right")

    def test_missing_target_clears_direction_immediately(self):
        controller = AlignmentController()
        controller.update([detection(.7)], 0)
        result = controller.update([], .1)
        self.assertEqual(result["state"], "lost")
        self.assertIsNone(result["offset"])
        self.assertIsNone(result["target_index"])

    def test_other_classes_are_not_the_target(self):
        self.assertEqual(AlignmentController().update([detection(.5, "chair")], 0)["state"], "lost")

    def test_nearby_target_wins_over_confidence_flip(self):
        controller = AlignmentController()
        controller.update([detection(.7, confidence=.9), detection(.2, confidence=.8)], 0)
        result = controller.update([detection(.69, confidence=.5), detection(.2, confidence=.99)], .1)
        self.assertEqual(result["target_index"], 0)
        self.assertEqual(result["state"], "right")

    def test_distant_replacement_not_immediately_selected(self):
        controller = AlignmentController()
        controller.update([detection(.8)], 0)
        self.assertEqual(controller.update([detection(.2)], .1)["state"], "lost")
        self.assertEqual(controller.update([detection(.2)], .9)["state"], "left")

    def test_image_position(self):
        self.assertEqual(position([.0, .0, .2, .2]), "top-left")
        self.assertEqual(position([.7, .7, .9, .9]), "bottom-right")

    def test_settings_validation(self):
        self.assertEqual(parse_settings({"target": "door", "confidence": .4, "revision": 1}), ("door", .4, 1))
        for bad in [{"target": "fake", "revision": 1}, {"confidence": float("nan"), "revision": 1},
                    {"confidence": 1, "revision": 1}, {"revision": "1"}, {"revision": True}]:
            with self.assertRaises(ValueError):
                parse_settings(bad)

    def test_decode_jpeg(self):
        self.assertEqual(decode_jpeg(jpeg_bytes()).shape, (240, 320, 3))
        output = BytesIO()
        Image.new("RGB", (320, 240)).save(output, "PNG")
        with self.assertRaises(ValueError):
            decode_jpeg(output.getvalue())


class MockBackend:
    def load(self):
        pass

    def infer(self, data, confidence):
        image = decode_jpeg(data)
        h, w = image.shape[:2]
        return {"detections": [detection(.7)], "width": w, "height": h,
                "detector_ms": 12.0, "model_inference_ms": 8.0}


class ConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_busy_drops_instead_of_queuing(self):
        started, finish = threading.Event(), threading.Event()

        class BlockingBackend(MockBackend):
            def infer(self, data, confidence):
                started.set()
                finish.wait(3)
                return super().infer(data, confidence)

        engine = SharedDetector(BlockingBackend())
        await engine.load()
        first = asyncio.create_task(engine.try_infer(jpeg_bytes(), .25))
        try:
            self.assertTrue(await asyncio.to_thread(started.wait, 2))
            self.assertIsNone(await engine.try_infer(jpeg_bytes(), .25))
        finally:
            finish.set()
            result = await first
            await engine.close()
        self.assertEqual(result["width"], 320)

    async def test_cancel_keeps_gpu_owned_until_thread_finishes(self):
        started, finish = threading.Event(), threading.Event()

        class BlockingBackend(MockBackend):
            def infer(self, data, confidence):
                started.set()
                finish.wait(3)
                return super().infer(data, confidence)

        engine = SharedDetector(BlockingBackend())
        task = asyncio.create_task(engine.try_infer(jpeg_bytes(), .25))
        try:
            self.assertTrue(await asyncio.to_thread(started.wait, 2))
            task.cancel()
            await asyncio.sleep(0)
            self.assertIsNone(await engine.try_infer(jpeg_bytes(), .25))
        finally:
            finish.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
            await engine.close()


class ServerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = TestClient(TestServer(create_app(SharedDetector(MockBackend()), {"model": "TEST ONLY"})))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()

    async def connect(self, target="whiteboard", revision=1):
        ws = await self.client.ws_connect("/ws")
        self.assertEqual((await ws.receive_json())["type"], "ready")
        await ws.send_json({"type": "configure", "target": target, "revision": revision})
        self.assertEqual((await ws.receive_json())["type"], "configured")
        return ws

    async def frame(self, ws, revision=1, frame_id=1):
        await ws.send_json({"type": "frame", "id": frame_id, "revision": revision})
        await ws.send_bytes(jpeg_bytes())
        return await ws.receive_json(timeout=3)

    async def test_page_and_health(self):
        response = await self.client.get("/")
        self.assertIn("Camera direction", await response.text())
        self.assertEqual((await (await self.client.get("/health")).json())["ready"], True)

    async def test_frame_round_trip_and_session_isolation(self):
        first, second = await self.connect(), await self.connect("door")
        result = await self.frame(first, frame_id=21)
        self.assertEqual(result["id"], 21)
        self.assertEqual(result["direction"]["state"], "right")
        self.assertEqual((await self.frame(second))["direction"]["state"], "lost")

    async def test_stale_configuration_is_rejected(self):
        ws = await self.connect(revision=2)
        self.assertEqual((await self.frame(ws, revision=1))["type"], "discarded")

    async def test_invalid_image_gives_actionable_error(self):
        ws = await self.connect()
        await ws.send_json({"type": "frame", "id": 1, "revision": 1})
        await ws.send_bytes(b"not a jpeg")
        self.assertEqual((await ws.receive_json())["type"], "error")

    async def test_unrelated_web_origin_rejected(self):
        response = await self.client.get("/ws", headers={"Origin": "https://unrelated.invalid"})
        self.assertEqual(response.status, 403)


if __name__ == "__main__":
    unittest.main()
