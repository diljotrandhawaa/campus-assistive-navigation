#!/usr/bin/env python3
"""Tests for the re-find look-around (place_search._watch + search_planner mode="refind").

    python test_refind.py
"""
import math

import numpy as np

import search_planner as SP
from place_search import PlaceSearch


class Lock:
    def __init__(self, label="chair"):
        self.label, self.spec = label, {"label": label, "classes": {label}, "keywords": []}
        self.side = self.exclude = self.prefer = None

    def cancel(self):
        self.label, self.spec = None, None


class Detector:
    classes = ["chair"]

    def set_prompts(self, c):
        pass


class Frame:
    def __init__(self, heading):
        r = math.radians(heading)
        self.forward = np.array([math.sin(r), -math.cos(r)])
        self.cam_pos = np.array([0.0, 1.3, 0.0])


def make(label="chair"):
    lock = Lock(label)
    sess = {"scene": None, "ocr": True}
    return PlaceSearch(Detector(), lock, sess, ["chair"], None), sess, lock


def run(ps, sess, reports, dt=0.1, turn=0.0, start=100.0):
    """reports: function t -> report. turn: degrees per second the user turns."""
    out, t = [], start
    for i in range(len(reports)):
        sess["last_frame"] = Frame((turn * (t - start)) % 360)
        for m in ps.on_frame(reports[i], t):
            out.append((round(t - start, 1), m.get("stage"), m.get("say"), m.get("found", False)))
        t += dt
    return out


def main():
    ok_all = []

    def check(name, ok, detail=""):
        ok_all.append(ok)
        print(f"{'PASS' if ok else 'FAIL'} {name}" + (f"  {detail}" if detail else ""))

    check("look-around is 10 s, no extension", SP.SCAN_TIMEOUT == 10.0 and not hasattr(SP, "SCAN_EXTRA"))

    seen_left = {"label": "chair", "state": "tracking", "bearing_deg": -20}
    lost = {"label": "chair", "state": "searching"}

    # 1. seen on the left, then lost; the user doesn't turn -> starts at 7 s, 10 s scan, then Help every 2 s
    ps, sess, lock = make()
    reports = [seen_left] * 10 + [lost] * 300          # 1 s seen, then 30 s lost
    out = run(ps, sess, reports)
    says = [(t, st, s) for t, st, s, f in out if s]
    first = says[0] if says else None
    check("starts 7 s after losing it, turning toward where it was seen",
          first is not None and 7.5 <= first[0] <= 8.2 and first[2] == "I lost the chair. Turn slowly to your left, all the way around.",
          str(first))
    helps = [t for t, st, s in says if s == "Help!"]
    check("after the 10 s look-around: 'Help!'", helps and 17.5 <= helps[0] <= 18.5, f"first Help at {helps[:1]}")
    gaps = np.diff(helps)
    check("'Help!' every 2 s", len(helps) >= 5 and all(1.9 <= g <= 2.2 for g in gaps), f"{len(helps)} times, gaps {gaps[:3]}")

    # 2. found again during the look-around -> planner hands back (found), no Help
    ps, sess, lock = make()
    reports = [seen_left] * 10 + [lost] * 100 + [seen_left] * 20
    out = run(ps, sess, reports)
    check("found during the look-around -> handed back to find mode, no Help",
          any(f for *_, f in out) and not any(s == "Help!" for _, _, s, _ in out) and ps.planner is None)

    # 3. a target with a fixed room spot (anchored, still 'tracking' out of frame) never triggers it
    ps, sess, lock = make()
    anchored = {"label": "chair", "state": "tracking", "anchored": True, "bearing_deg": 120}
    out = run(ps, sess, [seen_left] * 10 + [anchored] * 150)
    check("anchored target out of frame: no look-around", not out and ps.planner is None)

    # 4. full turn done early (user turns 40°/s) -> Help before 10 s are up
    ps, sess, lock = make()
    out = run(ps, sess, [seen_left] * 10 + [lost] * 200, turn=40.0)
    helps = [t for t, st, s, f in out if s == "Help!"]
    check("full turn done early -> Help right after the turn", helps and helps[0] < 17.5, f"first Help at {helps[:1]}")

    # 5. cancel stops the Help
    ps.stop()
    lock.cancel()
    out = run(ps, sess, [None] * 50, start=200.0)
    check("cancel stops Help", not out and ps.planner is None)

    # 6. lost for only 5 s -> nothing
    ps, sess, lock = make()
    out = run(ps, sess, [seen_left] * 10 + [lost] * 50 + [seen_left] * 30)
    check("lost only 5 s -> no look-around", not [m for m in out if m[2]])

    print(f"\n{'ALL PASSED' if all(ok_all) else 'SOME FAILED'} ({len(ok_all)} checks)")


if __name__ == "__main__":
    main()
