"""The shape of the free space around the user, from the last 5 s of LiDAR (data only).

While the user turns to scan, every few frames' depth points go on a 16 m x 16 m floor grid
(10 cm cells, ARKit room coordinates): points between just above the floor and head height are
obstacles (walls, furniture); the cells along each line of sight are free. Old evidence fades
(half-life 8 s), so the map describes "here, recently".

shape() measures the free area the user stands in: its length and width along its main
direction (PCA), and whether there are walls along both long sides. A hallway is long and
narrow with walls on both sides; a room is wider and squarer; a lobby is wide and open.
This works whichever way the user is facing, because it uses the whole scan.
"""
import math

import numpy as np

CELL = 0.10
SIZE = 160                  # 16 m
WINDOW = 5.0                # s: a cell not seen again within this is forgotten
MAX_RANGE = 5.0             # LiDAR is reliable to about here
EVERY_N_FRAMES = 3          # map update rate (cost ~ a few ms)

# What counts as a hallway


class SpaceMap:
    def __init__(self):
        self.origin = None
        self.free = np.zeros((SIZE, SIZE), np.float32)
        self.occ = np.zeros((SIZE, SIZE), np.float32)
        self.seen = np.full((SIZE, SIZE), -1e9, np.float32)   # last time each cell was seen
        self.last_t = None
        self.frames = 0
        self.pos = None
        self._shape_cache = (None, -1e9)

    def _cells(self, x, z):
        ix = np.floor((x - self.origin[0]) / CELL).astype(int) + SIZE // 2
        iz = np.floor((z - self.origin[1]) / CELL).astype(int) + SIZE // 2
        inside = (ix >= 0) & (ix < SIZE) & (iz >= 0) & (iz < SIZE)
        return ix[inside], iz[inside]

    def add(self, frame, now):
        self.frames += 1
        cx, cz = float(frame.cam_pos[0]), float(frame.cam_pos[2])
        self.pos = (cx, cz)
        if self.frames % EVERY_N_FRAMES:
            return
        if self.origin is None:
            self.origin = (cx, cz)
        elif math.hypot(cx - self.origin[0], cz - self.origin[1]) > 4.0:
            self._recenter(cx, cz)           # keep what was seen, centred on the user again
        self.last_t = now
        x, y, z = frame.world(step=4, min_d=0.2, max_d=MAX_RANGE)
        if x.size == 0:
            return
        obstacle = frame.body_band(y)
        oix, oiz = self._cells(x[obstacle], z[obstacle])
        np.add.at(self.occ, (oiz, oix), 1.0)
        dx, dz = x - cx, z - cz
        length = np.hypot(dx, dz)
        keep = length > 0.3
        dx, dz, length, obstacle = dx[keep], dz[keep], length[keep], obstacle[keep]
        reach = np.where(obstacle, (length - 0.2) / length, 1.0)
        t = np.linspace(0.0, 1.0, 25)[:, None] * reach[None, :]
        ix, iz = self._cells((cx + t * dx[None, :]).ravel(), (cz + t * dz[None, :]).ravel())
        cells = np.unique(iz * SIZE + ix)
        self.free.ravel()[cells] += 1.0
        # Last 5 s only: cells seen now are fresh; cells not seen again within WINDOW are forgotten.
        self.seen.ravel()[cells] = now
        self.seen[oiz, oix] = now          # obstacle cells (computed before the >0.3 m filter above)
        old = now - self.seen > WINDOW
        self.free[old] = 0.0
        self.occ[old] = 0.0

    def _recenter(self, cx, cz):
        """Shift the grid so the user is in the middle again (whole cells; edges are dropped)."""
        di = int(round((cx - self.origin[0]) / CELL))
        dj = int(round((cz - self.origin[1]) / CELL))
        for name in ("free", "occ", "seen"):
            old = getattr(self, name)
            new = np.full_like(old, -1e9) if name == "seen" else np.zeros_like(old)
            src = old[max(0, dj):SIZE + min(0, dj), max(0, di):SIZE + min(0, di)]
            new[max(0, -dj):max(0, -dj) + src.shape[0], max(0, -di):max(0, -di) + src.shape[1]] = src
            setattr(self, name, new)
        self.origin = (self.origin[0] + di * CELL, self.origin[1] + dj * CELL)

    def masks(self):
        """blocked (obstacles), free (seen free recently) as bool arrays [z, x]."""
        blocked = (self.occ >= 1.5) & (self.occ * 3 > self.free)
        free = (self.free >= 1.5) & ~blocked
        return blocked, free

    def world_to_cell(self, x, z):
        return (int(math.floor((x - self.origin[0]) / CELL)) + SIZE // 2,
                int(math.floor((z - self.origin[1]) / CELL)) + SIZE // 2)

    def cell_to_world(self, i, j):
        return (self.origin[0] + (i - SIZE // 2 + 0.5) * CELL, self.origin[1] + (j - SIZE // 2 + 0.5) * CELL)

    def shape(self, now=None):
        """{length_m, width_m, ratio, area_m2, walls_both_sides} of the free area around the
        user, or None if not enough has been seen. Cached for 1 s."""
        cached, at = self._shape_cache
        if now is not None and now - at < 1.0:
            return cached
        result = self._shape()
        if now is not None:
            self._shape_cache = (result, now)
        return result

    def _shape(self):
        import cv2
        if self.origin is None or self.pos is None:
            return None
        blocked, free = self.masks()
        free = free.astype(np.uint8)
        free = cv2.morphologyEx(free, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))   # fill ray gaps
        free[blocked] = 0
        n, labels = cv2.connectedComponents(free, connectivity=4)
        if n <= 1:
            return None
        mi, mj = self._cells(np.array([self.pos[0]]), np.array([self.pos[1]]))
        if mi.size == 0:
            return None
        label = labels[mj[0], mi[0]]
        if label == 0:   # standing on an unmapped cell: the nearest free area
            jj, ii = np.nonzero(labels)
            k = np.argmin((ii - mi[0]) ** 2 + (jj - mj[0]) ** 2)
            if math.hypot(ii[k] - mi[0], jj[k] - mj[0]) * CELL > 0.8:
                return None
            label = labels[jj[k], ii[k]]
        jj, ii = np.nonzero(labels == label)
        if ii.size < 200:                # less than 2 m² seen
            return None
        pts = np.stack([ii, jj], 1).astype(np.float64) * CELL
        mean = pts.mean(0)
        _, vecs = np.linalg.eigh(np.cov((pts - mean).T))
        major, minor = vecs[:, 1], vecs[:, 0]
        u, v = (pts - mean) @ major, (pts - mean) @ minor
        length = float(np.percentile(u, 98) - np.percentile(u, 2))
        width = float(np.percentile(v, 95) - np.percentile(v, 5))
        # Walls along both long sides: obstacle cells just outside the free area's width.
        bj, bi = np.nonzero(blocked)
        walls = False
        if bi.size:
            bp = np.stack([bi, bj], 1).astype(np.float64) * CELL - mean
            bu, bv = bp @ major, bp @ minor
            along = (bu > np.percentile(u, 2)) & (bu < np.percentile(u, 98))
            vlo, vhi = np.percentile(v, 5), np.percentile(v, 95)
            left = along & (bv < vlo + 0.2) & (bv > vlo - 0.6)
            right = along & (bv > vhi - 0.2) & (bv < vhi + 0.6)
            walls = int(left.sum()) >= 10 and int(right.sum()) >= 10
        center = mean + np.array([self.origin[0], self.origin[1]]) - (SIZE // 2) * CELL
        return {"length_m": round(length, 1), "width_m": round(width, 1),
                "ratio": round(length / max(width, 0.1), 1), "area_m2": round(ii.size * CELL * CELL, 1),
                "walls_both_sides": bool(walls),
                "center": (float(center[0]), float(center[1])),        # room x, z
                "axis": (float(major[0]), float(major[1]))}            # main direction (x, z)
