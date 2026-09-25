#!/usr/bin/env python3
"""Talks to fusion_server.py exactly like the iPhone app, using a fake LiDAR frame.

The fake depth map is a flat surface 2.0 m in front of the camera, so every
detected object should come back with distance_m close to 2.0.

    python fusion_client_test.py --url ws://127.0.0.1:8092/ws --image some_photo.jpg
"""
import argparse
import asyncio
import io
import json
import ssl
import struct
import time

import aiohttp
import numpy as np
from PIL import Image

DEPTH_W, DEPTH_H = 256, 192


def build_frame(image_path, frame_id, distance=2.0):
    if image_path:
        image = Image.open(image_path).convert("RGB")
    else:
        image = Image.new("RGB", (480, 640), (128, 128, 128))
    image.thumbnail((640, 640))
    buf = io.BytesIO()
    image.save(buf, "JPEG", quality=70)
    jpeg = buf.getvalue()

    depth = np.full((DEPTH_H, DEPTH_W), distance, dtype="<f2")
    confidence = np.full((DEPTH_H, DEPTH_W), 2, dtype=np.uint8)
    identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]  # camera at origin, looking -Z
    meta = {
        "type": "frame", "id": frame_id, "revision": 1,
        "image": {"w": image.width, "h": image.height},
        "depth": {"w": DEPTH_W, "h": DEPTH_H, "format": "f16"},
        "intrinsics": [212.0, 212.0, 128.0, 96.0],
        "transform": identity,
        "forward": [0.0, -1.0],
        "floor_y": -1.25,
        "corridor_half_width": 0.35,
        "timestamp": time.time(),
    }
    payload = struct.pack("<I", len(jpeg)) + jpeg + depth.tobytes() + confidence.tobytes()
    return meta, payload


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8092/ws")
    ap.add_argument("--image")
    ap.add_argument("--frames", type=int, default=10)
    args = ap.parse_args()

    ctx = None
    if args.url.startswith("wss://"):
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(args.url, ssl=ctx, max_msg_size=8_000_000) as ws:
            ready = await ws.receive_json()
            assert ready["type"] == "ready" and ready.get("protocol") == "fusion-1", ready
            print(f"Server ready: model={ready.get('model')} classes={len(ready.get('classes', []))}")
            await ws.send_json({"type": "configure", "confidence": 0.35, "revision": 1})
            assert (await ws.receive_json())["type"] == "configured"

            trips = []
            for i in range(1, args.frames + 1):
                meta, payload = build_frame(args.image, i)
                t = time.perf_counter()
                await ws.send_str(json.dumps(meta))
                await ws.send_bytes(payload)
                reply = await ws.receive_json()
                trips.append((time.perf_counter() - t) * 1000)
                if reply["type"] != "result":
                    print("Reply:", reply)
                    continue
                def fmt(o):
                    d = "?" if o["distance_m"] is None else f"{o['distance_m']:.2f} m"
                    return f"{o['label']} #{o['track_id']} {o['side']} {d}"
                desc = ", ".join(fmt(o) for o in reply["objects"]) or "(nothing)"
                print(f"frame {i}: {desc}   detector {reply['detector_ms']} ms, server {reply['server_ms']} ms")
            print(f"Payload {len(payload) / 1024:.0f} KB/frame, round trip avg {sum(trips) / len(trips):.0f} ms")


if __name__ == "__main__":
    asyncio.run(main())
