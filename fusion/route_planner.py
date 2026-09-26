"""A walking route to the locked target around big obstacles (tables, counters, rows of chairs).

Uses the 15 s LiDAR floor map (space_shape.SpaceMap) and the target's fixed room position
(target_lock anchor). Like a GPS on a small grid:
  1. 20 cm planning cells: obstacles, padded by the user's half-width plus a margin (0.35 m), can't be
     crossed; cells seen free cost 1; cells not seen yet cost more (they may be blocked) but
     can be crossed, so the route can lead into places the camera hasn't looked at yet.
  2. Shortest path (A*) from the user to the area in front of the target.
  3. The next WAYPOINT: the farthest point along that path (up to 3 m) that can be walked to
     in a straight line. The phone steers to it; once the straight line to the target is
     free again the route says "direct" and normal guidance takes over.

Safety stays on the phone: STOP within 2 ft comes from its own LiDAR, never from this route.

    route = plan_route(space_map, (x, z), anchor)   # -> {"status": "direct" | "detour" | "blocked", ...}
"""
import heapq
import math

import numpy as np

PLAN_CELL = 0.20      # m
BODY = 0.35           # m of padding around obstacles (half a person's width + margin);
                      # with 20 cm cells a gap needs to be about 1 m wide to route through
UNKNOWN_COST = 3.0    # an unseen cell costs this much more than a seen-free one
LOOKAHEAD = 3.0       # m: how far ahead the waypoint may be
GOAL_REACH = 0.6      # m: arriving this close to the target counts as reaching it


def _plan_grids(space_map):
    """(blocked, free) on the 20 cm planning grid, plus the scale from map cells."""
    import cv2
    blocked, free = space_map.masks()
    k = int(round(PLAN_CELL / 0.10))
    n = blocked.shape[0] // k
    b = blocked[:n * k, :n * k].reshape(n, k, n, k).any(axis=(1, 3))
    f = free[:n * k, :n * k].reshape(n, k, n, k).mean(axis=(1, 3)) >= 0.5
    pad = int(math.ceil(BODY / PLAN_CELL))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1))
    padded = cv2.dilate(b.astype(np.uint8), kernel).astype(bool)
    return b, padded, f & ~padded, k


def _to_plan(space_map, k, x, z):
    i, j = space_map.world_to_cell(x, z)
    return i // k, j // k


def _to_world(space_map, k, pi, pj):
    x0, z0 = space_map.cell_to_world(pi * k, pj * k)
    return x0 + (k - 1) * 0.05, z0 + (k - 1) * 0.05


def _line_cells(a, b):
    n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1]))) + 1
    return [(int(round(a[0] + (b[0] - a[0]) * t)), int(round(a[1] + (b[1] - a[1]) * t)))
            for t in np.linspace(0.0, 1.0, n)]


def _line_clear(padded, a, b):
    return all(not padded[j, i] for i, j in _line_cells(a, b))


def _astar(padded, free, start, goals):
    n = padded.shape[0]
    goal_set = set(goals)
    gi = np.mean([g[0] for g in goals])
    gj = np.mean([g[1] for g in goals])
    h = lambda c: math.hypot(c[0] - gi, c[1] - gj)
    best = {start: 0.0}
    came = {}
    heap = [(h(start), 0.0, start)]
    steps = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
             (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]
    while heap:
        _, g, c = heapq.heappop(heap)
        if c in goal_set:
            path = [c]
            while c in came:
                c = came[c]
                path.append(c)
            return path[::-1]
        if g > best.get(c, 1e18):
            continue
        for di, dj, step in steps:
            ni, nj = c[0] + di, c[1] + dj
            if not (0 <= ni < n and 0 <= nj < n) or padded[nj, ni]:
                continue
            if di and dj and (padded[c[1], ni] or padded[nj, c[0]]):
                continue                                  # no squeezing past a corner
            cost = g + step * (1.0 if free[nj, ni] else UNKNOWN_COST)
            if cost < best.get((ni, nj), 1e18):
                best[(ni, nj)] = cost
                came[(ni, nj)] = c
                heapq.heappush(heap, (cost + h((ni, nj)), cost, (ni, nj)))
    return None


def plan_route(space_map, start, anchor):
    """start: user's room (x, z). anchor: target {"x", "z", "width_m"}.
    Returns {"status": "direct"} | {"status": "detour", "waypoint": {"x","z"}, "length_m",
    "via_unseen": bool} | {"status": "blocked"} | None (no map yet)."""
    if space_map is None or space_map.origin is None:
        return None
    raw, padded, free, k = _plan_grids(space_map)
    n = padded.shape[0]
    s = _to_plan(space_map, k, *start)
    g = _to_plan(space_map, k, anchor["x"], anchor["z"])
    if not (0 <= s[0] < n and 0 <= s[1] < n and 0 <= g[0] < n and 0 <= g[1] < n):
        return None

    # The user's own spot and the target itself (the door is part of a wall) are not obstacles.
    ii, jj = np.meshgrid(np.arange(n), np.arange(n))
    near_start = np.hypot(ii - s[0], jj - s[1]) * PLAN_CELL <= BODY + 0.1
    near_goal = np.hypot(ii - g[0], jj - g[1]) * PLAN_CELL <= anchor.get("width_m", 0.9) / 2 + BODY
    padded = padded & ~near_start & ~(near_goal & ~raw)
    padded[near_goal & raw] = True                   # ...but the wall/door surface itself stays solid
    reach = np.hypot(ii - g[0], jj - g[1]) * PLAN_CELL <= GOAL_REACH
    goals = [(int(i), int(j)) for j, i in zip(*np.nonzero(reach & ~padded))]
    if not goals:
        return {"status": "blocked"}

    # Straight line to the target free of (padded) obstacles: nothing to plan.
    nearest_goal = min(goals, key=lambda c: math.hypot(c[0] - s[0], c[1] - s[1]))
    if _line_clear(padded, s, nearest_goal):
        return {"status": "direct"}

    path = _astar(padded, free, s, goals)
    if path is None:
        return {"status": "blocked"}

    # Waypoint: farthest path point within LOOKAHEAD reachable in a straight line.
    way = path[1] if len(path) > 1 else path[0]
    travelled = 0.0
    for a, b in zip(path, path[1:]):
        travelled += math.hypot(b[0] - a[0], b[1] - a[1]) * PLAN_CELL
        if travelled > LOOKAHEAD or not _line_clear(padded, s, b):
            break
        way = b
    wx, wz = _to_world(space_map, k, *way)
    length = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])) * PLAN_CELL
    unseen = any(not free[j, i] for i, j in _line_cells(s, way)[1:])
    return {"status": "detour", "waypoint": {"x": round(wx, 2), "z": round(wz, 2)},
            "length_m": round(length, 1), "via_unseen": bool(unseen)}


class RouteKeeper:
    """Per phone: re-plans at most twice a second and adds the route to the target report."""

    EVERY = 0.5

    def __init__(self):
        self.last = -1e9
        self.route = None
        self.anchor_key = None

    def update(self, target, space_map, pos, now):
        anchor = (target or {}).get("anchor")
        if not anchor or target.get("state") not in ("acquired", "tracking") or pos is None:
            self.route = None
            return None
        key = (round(anchor["x"], 1), round(anchor["z"], 1))
        if now - self.last >= self.EVERY or key != self.anchor_key:
            self.last, self.anchor_key = now, key
            try:
                self.route = plan_route(space_map, pos, anchor)
            except Exception as e:  # noqa: BLE001  (never let routing break the frame loop)
                print(f"Route error: {e}")
                self.route = None
        return self.route
