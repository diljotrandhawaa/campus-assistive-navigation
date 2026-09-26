#!/usr/bin/env python3
"""Tests for way_around.py (no GPU, no Ollama, no phone).

1. A Python copy of the phone's FreeSpace.layout() turns synthetic rooms into the exact
   "layout" the phone sends; way_around then builds the LLM input, checks a (fake) LLM
   answer and phrases it.
2. A WebSocket run through fusion_server.build_app with a stand-in detector:
   not a room -> "default"; room -> "Processing, wait." then the answer.

    python test_way_around.py
"""
import asyncio
import json
import math

import way_around as W

CELL, STRIP, MAXR = 0.10, 0.40, 4.0


# ---------------------------------------------------------------- copy of FreeSpace.swift
def heading(v):
    d = math.degrees(math.atan2(v[0], -v[1]))
    return d + 360 if d < 0 else d


def direction(h):
    r = math.radians(h)
    return (math.sin(r), -math.cos(r))


def signed(a):
    x = (a + 180) % 360
    return x - 180


class Grid:
    """User at (0, 0) facing -z (heading 0). Shapes are given as (ahead, lateral) boxes."""

    def __init__(self, seen=range(36)):
        self.cells = set()
        self.seen = set(seen)

    def box(self, a0, a1, l0, l1):
        for i in range(int(math.floor(l0 / CELL)), int(math.floor(l1 / CELL)) + 1):
            for j in range(int(math.floor(-a1 / CELL)), int(math.floor(-a0 / CELL)) + 1):
                self.cells.add((i, j))
        return self

    @staticmethod
    def center(k):
        return ((k[0] + 0.5) * CELL, (k[1] + 0.5) * CELL)

    def known(self, h):
        return (int(math.floor(h / 10)) % 36) in self.seen

    def clearance(self, pos, h, ignore=None):
        f = direction(h)
        r = (-f[1], f[0])
        best = MAXR
        for k in self.cells:
            c = self.center(k)
            if ignore and math.dist(c, ignore[0]) <= ignore[1]:
                continue
            rel = (c[0] - pos[0], c[1] - pos[1])
            ahead = rel[0] * f[0] + rel[1] * f[1]
            if 0.05 < ahead < best and abs(rel[0] * r[0] + rel[1] * r[1]) < STRIP:
                best = ahead
        return best

    def layout(self, pos, goal, target_distance, ignore=None):
        out = {"sweep": [], "obstacle": None, "gap_left_m": None, "gap_right_m": None}
        for deg in range(-90, 91, 15):
            h = goal + deg
            out["sweep"].append({"deg": deg, "free_m": round(self.clearance(pos, h, ignore), 2)
                                 if self.known(h % 360) else None})
        f = direction(goal)
        r = (-f[1], f[0])
        limit = min(target_distance or MAXR, MAXR)
        local = {}
        for k in self.cells:
            c = self.center(k)
            if ignore and math.dist(c, ignore[0]) <= ignore[1]:
                continue
            rel = (c[0] - pos[0], c[1] - pos[1])
            a, l = rel[0] * f[0] + rel[1] * f[1], rel[0] * r[0] + rel[1] * r[1]
            if 0.05 < a < MAXR and abs(l) < 3:
                local[k] = (a, l)
        seeds = [(v[0], k) for k, v in local.items() if v[0] < limit and abs(v[1]) < STRIP]
        if not seeds:
            return out
        seed_ahead, seed = min(seeds)

        def grow(start, allowed):
            got, queue = {start}, [start]
            while queue:
                k = queue.pop()
                for dx in range(-2, 3):
                    for dz in range(-2, 3):
                        n = (k[0] + dx, k[1] + dz)
                        if n not in got and n in local and allowed(n, local[n]):
                            got.add(n)
                            queue.append(n)
            return got
        cluster = grow(seed, lambda n, v: v[0] >= seed_ahead - 0.2)
        half = CELL / 2
        pts = [local[k] for k in cluster]
        near, far = min(p[0] for p in pts) - half, max(p[0] for p in pts) + half
        ed = {"left": min(p[1] for p in pts) - half, "right": max(p[1] for p in pts) + half}
        depth = far - near

        def seen(lateral):
            a = max(near, 0.3)
            v = (f[0] * a + r[0] * lateral, f[1] * a + r[1] * lateral)
            return self.known(heading(v))

        def gap(s):
            g = 0.0
            for _ in range(3):
                edge = ed["left"] if s < 0 else ed["right"]
                nxt = None
                for k, v in local.items():
                    if k in cluster or not (near - 0.3 < v[0] < far + 0.3):
                        continue
                    d = (v[1] - edge) * s
                    if d > 0 and (nxt is None or d < nxt[1]):
                        nxt = (k, d)
                if nxt is None:
                    return round(max(0, 3 - edge * s), 2) if seen(edge + 0.3 * s) else None
                g = max(0, nxt[1] - half)
                if g >= 0.8:
                    return round(g, 2)
                part = grow(nxt[0], lambda n, v: n not in cluster)
                aheads = [local[k][0] for k in part]
                if max(aheads) - min(aheads) > depth + 1.0:
                    return round(g, 2)
                cluster.update(part)
                for k in part:
                    ed["left"] = min(ed["left"], local[k][1] - half)
                    ed["right"] = max(ed["right"], local[k][1] + half)
            return round(g, 2)
        out["gap_left_m"] = gap(-1)
        out["gap_right_m"] = gap(1)
        out["obstacle"] = {"near_m": round(near, 2), "far_m": round(far, 2), "left_m": round(ed["left"], 2),
                           "right_m": round(ed["right"], 2), "left_seen": seen(ed["left"] - 0.3),
                           "right_seen": seen(ed["right"] + 0.3)}
        return out


# ---------------------------------------------------------------- fake LLM
class FakeLLM:
    def __init__(self, answer=None, error=None):
        self.answer, self.error, self.calls = answer, error, []

    def _post(self, messages, num_predict=80, schema=None):
        self.calls.append({"messages": messages, "schema": schema})
        if self.error:
            raise self.error
        return {"message": {"content": json.dumps(self.answer)}}

    def warmup(self):
        return True


def request(grid, label="door", dist=4.5, bearing=0.0, hint=None):
    return {"type": "blocked", "request_id": 1, "obstacle": hint,
            "target": {"label": label, "distance_m": dist, "bearing_deg": bearing, "width_m": 0.9},
            "layout": grid.layout((0.0, 0.0), 0.0, dist)}


def objects_at(*items):
    """(label, ahead, lateral) in the user's facing frame -> GB10 detections."""
    return (0.0, [{"label": n, "distance_m": a, "lateral_m": l} for n, a, l in items])


def run(name, grid, llm_answer, expect_say, expect_method, bearing=0.0, objects=None, hint=None, error=None):
    msg = request(grid, bearing=bearing, hint=hint)
    data = W.build_input(msg, objects, 0.0)
    llm = FakeLLM(llm_answer, error)
    ans = W.decide(llm, data)
    ok = ans["say"] == expect_say and ans["method"].startswith(expect_method)
    print(f"{'PASS' if ok else 'FAIL'} {name}: '{ans['say']}' [{ans['method']}] "
          f"gaps L={data['gap_left_m']} R={data['gap_right_m']} steps L={data['steps_left']} R={data['steps_right']}")
    if not ok:
        print("   input:", json.dumps(data))
    return ok, llm, data


def unit_tests():
    results = []
    W.SWAP_SIDES = False          # the decision logic itself is tested with the original sides
    # A: table 1.8 m ahead, wall close on the left -> around on the right
    g = Grid().box(1.8, 2.6, -1.0, 0.5).box(0.2, 4.0, -1.7, -1.6)
    ok, llm, data = run("table, wall on the left", g, {"a": "around", "s": "right", "o": "table"},
                        "Table ahead, gap on the right.", "llm", objects=objects_at(("table", 2.0, -0.2)))
    results.append(ok)
    sch = llm.calls[0]["schema"]
    results.append(sch["properties"]["o"]["enum"][0] == "table")
    results.append("Processing" not in llm.calls[0]["messages"][0]["content"]
                   and "shortest spoken sentence" in llm.calls[0]["messages"][0]["content"])

    # B: chair close (0.6 m), target slightly left -> side-step left
    g = Grid().box(0.6, 1.0, -0.25, 0.25)
    ok, _, _ = run("chair close, target slightly left", g, {"a": "sidestep", "s": "left", "o": "chair"},
                   "Chair ahead, gap on the left.", "llm", bearing=-15, hint="chair")
    results.append(ok)

    # C: table, left side not seen yet, wall right -> look left
    g = Grid(seen=list(range(33, 36)) + list(range(0, 10))).box(1.4, 2.0, -1.5, 0.3).box(0.2, 4.0, 0.6, 0.7)
    ok, _, _ = run("table, left not seen, wall right", g, {"a": "look", "s": "left", "o": "table"},
                   "Table ahead. No gap found yet.", "llm", hint="table")
    results.append(ok)

    # D: sofa between two walls -> blocked
    g = Grid().box(1.2, 1.9, -1.0, 1.0).box(0.2, 4.0, -1.45, -1.4).box(0.2, 4.0, 1.35, 1.4)
    ok, _, _ = run("sofa between walls", g, {"a": "blocked", "s": None, "o": "sofa"},
                   "Sofa is blocking the way. No gap found.", "llm", hint="sofa")
    results.append(ok)

    # E: LLM picks a side with no gap -> rejected, rules answer
    g = Grid().box(1.8, 2.6, -1.0, 0.5).box(0.2, 4.0, -1.7, -1.6)
    ok, _, _ = run("LLM picks the walled side", g, {"a": "around", "s": "left", "o": "table"},
                   "Table ahead, gap on the right.", "rule (llm answer did not fit)", hint="table")
    results.append(ok)

    # F: LLM times out -> rules answer
    ok, _, _ = run("LLM timeout", g, None, "Table ahead, gap on the right.", "rule (llm failed: TimeoutError)",
                   hint="table", error=TimeoutError())
    results.append(ok)

    # G: person in the right gap -> left (rules agree with the LLM)
    g = Grid().box(2.0, 2.6, -0.6, 0.6)
    objs = objects_at(("desk", 2.1, 0.0), ("person", 2.3, 1.4))
    ok, _, data = run("desk, person on the right", g, {"a": "around", "s": "left", "o": "desk"},
                      "Desk ahead, gap on the left.", "llm", bearing=10, objects=objs)
    results.append(ok)
    results.append(W.rule_decision(data)["s"] == "left")

    # H: the LLM says "around" for a close obstacle -> the distance decides: side-step
    g = Grid().box(0.7, 1.0, -0.3, 0.9)
    ok, _, _ = run("close obstacle, LLM says around", g, {"a": "around", "s": "left", "o": None},
                   "Box ahead, gap on the left.", "llm", hint="box")
    results.append(ok)
    # I: two chairs 0.4 m apart (too narrow between them), wall on the right -> around both, on the left
    g = Grid().box(1.5, 1.9, -0.3, 0.3).box(1.5, 1.9, -1.2, -0.7).box(0.2, 4.0, 0.9, 1.0)
    ok, _, data = run("two chairs, gap beyond the second", g, {"a": "around", "s": "left", "o": "chair"},
                      "Chair ahead, gap on the left.", "llm", hint="chair")
    results.append(ok and data["steps_left"] == 3 and data["gap_right_m"] < 0.8)
    # J: SWAP_SIDES flips the final answer (speech, side, steps)
    W.SWAP_SIDES = True
    g = Grid().box(1.8, 2.6, -1.0, 0.5).box(0.2, 4.0, -1.7, -1.6)
    ans = W.decide(FakeLLM({"a": "around", "s": "right", "o": "table"}), W.build_input(request(g, hint="table"), None, 0.0))
    ok = ans["say"] == "Table ahead, gap on the left." and ans["side"] == "left" and ans["method"].endswith("+swapped")
    print(f"{'PASS' if ok else 'FAIL'} SWAP_SIDES: right -> '{ans['say']}' [{ans['method']}]")
    g = Grid().box(0.6, 1.0, -0.25, 0.25)
    ans = W.decide(FakeLLM({"a": "sidestep", "s": "left", "o": "chair"}), W.build_input(request(g, bearing=-15, hint="chair"), None, 0.0))
    ok2 = ans["say"] == "Chair ahead, gap on the right." and ans["side"] == "right" and ans["steps"] is None
    print(f"{'PASS' if ok2 else 'FAIL'} SWAP_SIDES: sidestep left -> '{ans['say']}'")
    results += [ok, ok2]
    W.SWAP_SIDES = False
    return all(results), len(results)


# ---------------------------------------------------------------- WebSocket end to end
class StubDetector:
    classes = ["chair", "table", "door", "sofa", "desk", "person"]
    model_path = "stub.pt"
    imgsz = 640
    ocr = None

    def new_session(self):
        return {"kind": "builtin", "target": None}

    def set_prompts(self, classes):
        pass


class StubInterpreter:
    def warmup(self):
        return True


async def e2e():
    from aiohttp import ClientSession
    from aiohttp.test_utils import TestServer

    import fusion_server as F
    g = Grid().box(1.8, 2.6, -1.0, 0.5).box(0.2, 4.0, -1.7, -1.6)
    msg = request(g, hint="table")
    ok = True
    for place, llm, expect in (("unknown", FakeLLM({"a": "around", "s": "right", "o": "table"}), ["default"]),
                               ("room", FakeLLM({"a": "around", "s": "right", "o": "table"}), ["processing", "answer"]),
                               ("room", None, ["answer"])):
        W.place_of = (lambda p: (lambda s, n: p))(place)
        server = TestServer(F.build_app(StubDetector(), interpreter=StubInterpreter(), llm=llm))
        await server.start_server()
        async with ClientSession() as http:
            async with http.ws_connect(server.make_url("/ws")) as ws:
                ready = await ws.receive_json()
                assert ready["type"] == "ready"
                await ws.send_json(msg)
                got = []
                while len(got) < len(expect):
                    m = await asyncio.wait_for(ws.receive_json(), 5)
                    if m.get("type") == "way":
                        got.append(m)
        await server.close()
        statuses = [m["status"] for m in got]
        good = statuses == expect and (expect[-1] != "answer" or got[-1]["say"] == "Table ahead, gap on the right.")
        if "processing" in expect:
            good = good and got[0]["say"] == "Processing, wait."
        print(f"{'PASS' if good else 'FAIL'} websocket place={place} llm={'yes' if llm else 'no'}: "
              + " | ".join(f"{m['status']}: {m.get('say', '')}" for m in got))
        ok = ok and good
    return ok


if __name__ == "__main__":
    u, n = unit_tests()
    e = asyncio.run(e2e())
    print(f"\n{'ALL PASSED' if u and e else 'SOME FAILED'} ({n} unit checks + 3 websocket runs)")
