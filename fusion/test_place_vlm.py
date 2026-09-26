#!/usr/bin/env python3
"""Tests for place_vlm.py with a fake OpenAI-style VLM server (no GPU, no zrt needed).

    python test_place_vlm.py
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import place_vlm as P

STATE = {"answers": [], "requests": [], "delay": 0.0}


class Fake(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send(200, {"object": "list", "data": [{"id": "vlm", "object": "model"}]})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        STATE["requests"].append(req)
        time.sleep(STATE["delay"])
        text = STATE["answers"].pop(0) if STATE["answers"] else "unknown"
        self._send(200, {"choices": [{"message": {"content": text}}]})


def main():
    srv = HTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_port}/v1"
    results = []

    def check(name, ok):
        results.append(ok)
        print(f"{'PASS' if ok else 'FAIL'} {name}")

    client = P.VLMClient(url, "vlm")
    check("server check finds model 'vlm'", client.check())

    # frames: one per 45° slice, newest 4, oldest first
    ctx = P.PlaceContext(client)
    t = 100.0
    for i, h in enumerate([0, 10, 50, 95, 140, 185, 230]):   # slices 0,0,1,2,3,4,5
        ctx.add_frame(f"img{i}".encode(), h, t + i * 0.1)
    picked = ctx._pick()
    check("one frame per direction, newest 4", picked == [b"img3", b"img4", b"img5", b"img6"])

    # an answer, with the request in the right shape
    STATE["answers"] = ["hallway"]
    ctx.tick(t + 1, lambda: "Seen: 3 door")
    ctx.pending.result(5)
    req = STATE["requests"][-1]
    imgs = [c for c in req["messages"][1]["content"] if c["type"] == "image_url"]
    check("request: model vlm, 4 images, max 3 tokens, no JSON format, hints + 'One word:'",
          req["model"] == "vlm" and len(imgs) == 4 and imgs[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
          and req["max_tokens"] == 3 and "response_format" not in req
          and req["messages"][1]["content"][-1]["text"] == "Hints: Seen: 3 door\nOne word:")
    cur = ctx.current(t + 2)
    check("current = {'place': 'hallway', 'planner_place': 'hallway'} only",
          cur == {"place": "hallway", "planner_place": "hallway"})

    # throttle: no new ask within 3 s
    n = len(STATE["requests"])
    ctx.tick(t + 2)
    check("no second ask within 3 s", len(STATE["requests"]) == n)

    # a new place once doesn't flip; twice in a row does; unknown never replaces
    STATE["answers"] = ["room"]
    ctx.tick(t + 5); ctx.pending.result(5)
    check("'room' once keeps hallway", ctx.current(t + 6)["place"] == "hallway")
    STATE["answers"] = ["unknown"]
    ctx.tick(t + 8); ctx.pending.result(5)
    check("'unknown' keeps hallway", ctx.current(t + 9)["place"] == "hallway")
    STATE["answers"] = ["Lobby.", "lobby"]
    ctx.tick(t + 11); ctx.pending.result(5)
    check("'Lobby.' once keeps hallway", ctx.current(t + 12)["place"] == "hallway")
    ctx.tick(t + 14); ctx.pending.result(5)
    cur = ctx.current(t + 15)
    check("'lobby' twice in a row replaces it; planner_place hallway", cur == {"place": "lobby", "planner_place": "hallway"})
    check("unknown after 20 s without a new answer", ctx.current(t + 40)["place"] == "unknown")

    # get(): old answer -> asks now and waits
    STATE["answers"] = ["room", "room"]
    STATE["delay"] = 0.3
    fresh = P.PlaceContext(client)
    for i in range(3):
        fresh.add_frame(b"new", i * 60, time.monotonic())
    t0 = time.perf_counter()
    got = fresh.get(time.monotonic(), "hints", wait=3)
    check(f"get() asks and waits ({time.perf_counter() - t0:.2f} s): room", got["place"] == "room")
    STATE["delay"] = 0.0

    # replies
    check("chatty reply parsed: 'The place is a Hallway.'", P.parse_place("The place is a Hallway.") == "hallway")
    check("non-place reply ignored", P.parse_place("kitchen") is None)
    check("bad image bytes are sent as they are (shrink falls back)", P.shrink(b"not a jpeg") == b"not a jpeg")

    # server down -> unknown, no crash
    dead = P.PlaceContext(P.VLMClient("http://127.0.0.1:9/v1", "vlm", timeout=1))
    dead.add_frame(b"x", 0, time.monotonic())
    check("server down -> unknown", dead.get(time.monotonic(), "", wait=2)["place"] == "unknown")
    check("no VLM (--no-vlm) -> unknown", P.PlaceContext(None).get(time.monotonic())["place"] == "unknown")

    # consumers
    import exit_planner
    import way_around
    from target_lock import TargetLock

    class Ctx:
        def __init__(self, place, conf=0.9):
            self.place, self.conf = place, conf

        def get(self, now, hints=None, wait=0):
            return {"place": self.place, "planner_place": P.PLANNER_PLACE[self.place]}

        def current(self, now):
            return self.get(now)

    classes = ["door", "exit sign", "sign", "chair"]
    for place, label in (("room", "door"), ("hallway", "exit"), ("stairwell", "exit"), ("unknown", "exit")):
        lock = TargetLock()
        sess = {"place": Ctx(place), "scene": None}
        r = exit_planner.plan_exit("find an exit", "find an exit", lock, sess, classes)
        check(f"exit planner: VLM '{place}' -> target {label}", r["target"] == label and r["place"] == P.PLANNER_PLACE[place])
    check("way_around.place_of reads the VLM layer", way_around.place_of({"place": Ctx("room")}, 0) == "room"
          and way_around.place_of({"place": Ctx("lobby")}, 0) == "hallway")

    srv.shutdown()
    print(f"\n{'ALL PASSED' if all(results) else 'SOME FAILED'} ({len(results)} checks)")


if __name__ == "__main__":
    main()
