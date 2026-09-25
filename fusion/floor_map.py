"""A small LiDAR floor map around the user, and where the most open spot is.

Every frame's depth points are dropped onto a 12 m x 12 m grid of 10 cm cells (ARKit world
coordinates, floor plane):
  - points between just above the floor and head height are OBSTACLES (walls, furniture),
  - the cells along each line of sight from the camera to a point are FREE.
best_spot() picks the known-free cell farthest from obstacles (roughly the middle of the open
part of the room) that is reachable in a straight line; obstacle_ahead() is a quick check of
the walking direction in the current frame.
"""
import numpy as np

CELL = 0.10          # m
SIZE = 120           # cells -> 12 m
MIN_CONFIDENCE = 1   # ARKit medium
MAX_RANGE = 5.0      # m


def _world_points(frame, step=4):
    """(x, z, height) of this frame's confident depth points, plus the camera position."""
    ys, xs = np.mgrid[0:frame.h:step, 0:frame.w:step]
    d = frame.depth[ys, xs].astype(np.float64)
    ok = (frame.confidence[ys, xs] >= MIN_CONFIDENCE) & np.isfinite(d) & (d > 0.2) & (d < MAX_RANGE)
    d, xs, ys = d[ok], xs[ok], ys[ok]
    cam = np.stack([(xs + 0.5 - frame.cx) / frame.fx * d, -((ys + 0.5 - frame.cy) / frame.fy * d),
                    -d, np.ones_like(d)])
    w = frame.T @ cam
    return w[0], w[2], w[1]


def obstacle_ahead(frame, max_m=1.2, half_width=0.35, bearing_deg=0.0):
    """True if something between knee and head height is within max_m in the given direction
    (degrees from the walking direction, + = right)."""
    x, z, y = _world_points(frame)
    band = (y > frame.floor_y + 0.15) & (y < frame.cam_pos[1] + 0.3)
    rel = np.stack([x - frame.cam_pos[0], z - frame.cam_pos[2]])[:, band]
    b = np.radians(bearing_deg)
    fwd = np.cos(b) * frame.forward + np.sin(b) * frame.right
    right = np.array([-fwd[1], fwd[0]])
    ahead, lateral = fwd @ rel, right @ rel
    hits = (ahead > 0.1) & (ahead < max_m) & (np.abs(lateral) < half_width)
    return int(hits.sum()) >= 8


def open_distance(frame, max_m=5.0, half_width=0.35):
    """How far the way straight ahead is free (m, up to max_m), from this frame's depth alone.
    None if the frame shows too little to tell."""
    x, z, y = _world_points(frame)
    if x.size < 100:
        return None
    band = (y > frame.floor_y + 0.15) & (y < frame.cam_pos[1] + 0.3)
    rel = np.stack([x - frame.cam_pos[0], z - frame.cam_pos[2]])[:, band]
    right = np.array([-frame.forward[1], frame.forward[0]])
    ahead, lateral = frame.forward @ rel, right @ rel
    hits = ahead[(ahead > 0.1) & (np.abs(lateral) < half_width)]
    if hits.size < 8:
        return max_m                      # nothing in the way within LiDAR range
    return float(min(max_m, np.percentile(hits, 5)))


class FloorMap:
    def __init__(self):
        self.origin = None                       # world (x, z) of the grid centre
        self.free = np.zeros((SIZE, SIZE), np.int16)
        self.occ = np.zeros((SIZE, SIZE), np.int16)

    def _cells(self, x, z):
        ix = np.floor((x - self.origin[0]) / CELL).astype(int) + SIZE // 2
        iz = np.floor((z - self.origin[1]) / CELL).astype(int) + SIZE // 2
        inside = (ix >= 0) & (ix < SIZE) & (iz >= 0) & (iz < SIZE)
        return ix[inside], iz[inside]

    def add_frame(self, frame):
        cx, cz = float(frame.cam_pos[0]), float(frame.cam_pos[2])
        if self.origin is None:
            self.origin = (cx, cz)
        x, z, y = _world_points(frame)
        if x.size == 0:
            return
        obstacle = (y > frame.floor_y + 0.15) & (y < frame.cam_pos[1] + 0.3)
        # Obstacles
        ix, iz = self._cells(x[obstacle], z[obstacle])
        np.add.at(self.occ, (iz, ix), 1)
        # Free space along each line of sight (stop 20 cm short of an obstacle)
        dx, dz = x - cx, z - cz
        length = np.hypot(dx, dz)
        keep = length > 0.3
        dx, dz, length, obstacle = dx[keep], dz[keep], length[keep], obstacle[keep]
        reach = np.where(obstacle, (length - 0.2) / length, 1.0)
        t = np.linspace(0.0, 1.0, 30)[:, None] * reach[None, :]
        fx, fz = (cx + t * dx[None, :]).ravel(), (cz + t * dz[None, :]).ravel()
        ix, iz = self._cells(fx, fz)
        cells = np.unique(iz * SIZE + ix)
        self.free.ravel()[cells] += 1
        np.minimum(self.free, 1000, out=self.free)
        np.minimum(self.occ, 1000, out=self.occ)

    def masks(self):
        blocked = (self.occ >= 3) & (self.occ * 3 > self.free)
        known_free = (self.free >= 3) & ~blocked
        return blocked, known_free

    def _line_clear(self, known_free, a, b):
        n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1]))) + 1
        for s in np.linspace(0.0, 1.0, n):
            i = int(round(a[0] + s * (b[0] - a[0])))
            j = int(round(a[1] + s * (b[1] - a[1])))
            if not known_free[j, i]:
                return False
        return True

    def best_spot(self, pos, min_clearance=0.5, min_travel=1.0, max_travel=6.0):
        """World (x, z) of the most open reachable spot, with its clearance and distance, or None."""
        import cv2
        if self.origin is None:
            return None
        blocked, known_free = self.masks()
        if known_free.sum() < 50:
            return None
        # Distance from every known-free cell to the nearest non-free cell (obstacle or unexplored).
        clearance = cv2.distanceTransform(known_free.astype(np.uint8), cv2.DIST_L2, 5) * CELL
        me = self._cells(np.array([pos[0]]), np.array([pos[1]]))
        if me[0].size == 0:
            return None
        mi, mj = int(me[0][0]), int(me[1][0])
        jj, ii = np.nonzero(known_free & (clearance >= min_clearance))
        if ii.size == 0:
            return None
        travel = np.hypot(ii - mi, jj - mj) * CELL
        ok = (travel >= min_travel) & (travel <= max_travel)
        if not ok.any():
            return None
        ii, jj, travel = ii[ok], jj[ok], travel[ok]
        score = clearance[jj, ii] - 0.1 * travel
        known_free[mj, mi] = True  # the user's own cell
        for k in np.argsort(-score)[:40]:
            if self._line_clear(known_free, (mi, mj), (ii[k], jj[k])):
                x = self.origin[0] + (ii[k] - SIZE // 2 + 0.5) * CELL
                z = self.origin[1] + (jj[k] - SIZE // 2 + 0.5) * CELL
                return {"x": float(x), "z": float(z), "clearance_m": float(clearance[jj[k], ii[k]]),
                        "travel_m": float(travel[k])}
        return None
