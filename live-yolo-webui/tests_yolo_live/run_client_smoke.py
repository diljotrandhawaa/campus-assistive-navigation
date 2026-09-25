"""Run a real local WebSocket with simulated detector and simulated browser DOM.
Requires Node 22+ on the test machine. Does not require PyTorch or a GPU.
"""
import asyncio
from pathlib import Path
import sys
import tempfile

from PIL import Image
from aiohttp import web

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mock_server import MockBackend
from yolo_live.engine import SharedDetector
from yolo_live.server import create_app


async def main():
    runner = web.AppRunner(create_app(SharedDetector(MockBackend()),
                                     {"model": "TEST ONLY", "device": "mock", "imgsz": 640}))
    await runner.setup()
    try:
        await web.TCPSite(runner, "127.0.0.1", 18091).start()
        with tempfile.TemporaryDirectory(prefix="yolo-client-test-") as folder:
            image = Path(folder) / "test.jpg"
            Image.new("RGB", (320, 240), "white").save(image, "JPEG")
            process = await asyncio.create_subprocess_exec(
                "node", str(Path(__file__).with_name("client_smoke.cjs")), str(image))
            code = await process.wait()
            if code:
                raise SystemExit(code)
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
