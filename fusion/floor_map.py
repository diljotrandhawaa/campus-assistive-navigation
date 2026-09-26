"""Quick LiDAR checks for walking, and the most open spot on the shared floor map.

  obstacle_ahead(frame)  something within reach in a direction (this frame only)
  open_distance(frame)   how far the way straight ahead is free (this frame only)
  best_spot(space, pos)  the known-free spot farthest from obstacles (roughly the middle of
                         the open part of the room) reachable in a straight line, on the
                         shared 5 s floor map (space_shape.SpaceMap)
All depth -> room math is in geometry.FrameData.
"""
import numpy as np

MAX_RANGE = 5.0      # m


def _strip(frame, bearing_deg=0.0, half_width=0.35):
    """Distances ahead of body-height points inside a person-wide strip in a direction
    (degrees from the walking direction, + = right); None if the frame shows too little."""
    x, y, z = frame.world(step=4, min_d=0.2, max_d=MAX_RANGE)
    if x.size < 100:
        return None
    band = frame.body_band(y)
    rel = np.stack([x[band] - frame.cam_pos[0], z[band] - frame.cam_pos[2]])
    b = np.radians(bearing_deg)
    fwd = np.cos(b) * frame.forward + np.sin(b) * frame.right
    right = np.array([-fwd[1], fwd[0]])
    ahead, lateral = fwd @ rel, right @ rel
    return ahead[(ahead > 0.1) & (np.abs(lateral) < half_width)]


def obstacle_ahead(frame, max_m=1.2, half_width=0.35, bearing_deg=0.0):
    """True if something between knee and head height is within max_m in the given direction."""
    hits = _strip(frame, bearing_deg, half_width)
    return hits is not None and int((hits < max_m).sum()) >= 8


def open_distance(frame, max_m=5.0, half_width=0.35):
    """How far the way straight ahead is free (m, up to max_m). None if too little is seen."""
    hits = _strip(frame, 0.0, half_width)
    if hits is None:
        return None
    if hits.size < 8:
        return max_m                      # nothing in the way within LiDAR range
    return float(min(max_m, np.percentile(hits, 5)))


def _line_clear(known_free, a, b):
    n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1]))) + 1
    for s in np.linspace(0.0, 1.0, n):
        i = int(round(a[0] + s * (b[0] - a[0])))
        j = int(round(a[1] + s * (b[1] - a[1])))
        if not known_free[j, i]:
            return False
    return True


def best_spot(space, pos, min_clearance=0.5, min_travel=1.0, max_travel=6.0):
    """Room (x, z) of the most open reachable spot on the shared floor map, with its
    clearance and distance, or None."""
    import cv2
    from space_shape import CELL
    if space is None or space.origin is None:
        return None
    _, known_free = space.masks()
    if known_free.sum() < 50:
        return None
    # Distance from every known-free cell to the nearest non-free cell (obstacle or unexplored).
    clearance = cv2.distanceTransform(known_free.astype(np.uint8), cv2.DIST_L2, 5) * CELL
    mi, mj = space.world_to_cell(*pos)
    n = known_free.shape[0]
    if not (0 <= mi < n and 0 <= mj < n):
        return None
    jj, ii = np.nonzero(known_free & (clearance >= min_clearance))
    if ii.size == 0:
        return None
    travel = np.hypot(ii - mi, jj - mj) * CELL
    ok = (travel >= min_travel) & (travel <= max_travel)
    if not ok.any():
        return None
    ii, jj, travel = ii[ok], jj[ok], travel[ok]
    score = clearance[jj, ii] - 0.1 * travel
    known_free = known_free.copy()
    known_free[mj, mi] = True  # the user's own cell
    for k in np.argsort(-score)[:40]:
        if _line_clear(known_free, (mi, mj), (ii[k], jj[k])):
            x, z = space.cell_to_world(int(ii[k]), int(jj[k]))
            return {"x": float(x), "z": float(z), "clearance_m": float(clearance[jj[k], ii[k]]),
                    "travel_m": float(travel[k])}
    return None
