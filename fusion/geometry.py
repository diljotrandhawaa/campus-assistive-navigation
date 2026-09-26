"""Frame geometry shared by every server module (one copy, used everywhere).

FrameData is one ARKit frame's LiDAR depth + camera pose, as sent by the phone (same
geometry as DepthSnapshot.swift). Everything that turns depth pixels into room coordinates
goes through FrameData.world(): object distances (measure), the hallway hint (space_hint),
the floor maps (space_shape), obstacle checks (floor_map) and the target anchor (target_lock).

Room coordinates are ARKit world coordinates: x/z on the floor plane, y up.
Headings are compass-like degrees: 0 = where ARKit started facing, 90 = turned right.
"""
import math

import numpy as np

MIN_CONFIDENCE = 1       # ARKit: 0 low, 1 medium, 2 high
FLOOR_MARGIN = 0.10      # points this close above the floor count as floor (m)
MAX_DEPTH = 5.5          # LiDAR is unreliable beyond this (m)
BOX_SHRINK = 0.10        # ignore the outer 10% of each box (background leaks in)


def heading_deg(forward):
    """Walking direction (x, z) -> degrees: 0 = where ARKit started facing, 90 = right."""
    return math.degrees(math.atan2(float(forward[0]), -float(forward[1]))) % 360.0


def signed_diff(a, b):
    """Smallest signed angle from b to a, in (-180, 180]; + = a is to the right of b."""
    return (a - b + 180.0) % 360.0 - 180.0


class FrameData:
    """One ARKit frame's depth + pose, as sent by the phone."""

    def __init__(self, meta, depth, confidence):
        d = meta["depth"]
        self.w, self.h = int(d["w"]), int(d["h"])
        self.depth = depth.reshape(self.h, self.w)
        self.confidence = confidence.reshape(self.h, self.w)
        self.fx, self.fy, self.cx, self.cy = (float(v) for v in meta["intrinsics"])
        # 16 floats, column-major (simd_float4x4 columns) -> row-major matrix
        self.T = np.asarray(meta["transform"], dtype=np.float64).reshape(4, 4).T
        self.cam_pos = self.T[:3, 3]
        f = np.asarray(meta["forward"], dtype=np.float64)          # horizontal (x, z), unit
        self.forward = f / max(np.linalg.norm(f), 1e-6)
        self.right = np.array([-self.forward[1], self.forward[0]])
        self.floor_y = float(meta["floor_y"])
        self.half_width = float(meta.get("corridor_half_width", 0.35))

    # ---------------------------------------------------------------- depth -> room
    def world(self, x0=0, y0=0, x1=None, y1=None, step=1, min_d=0.05, max_d=MAX_DEPTH):
        """Room (x, y, z) arrays of the confident depth pixels in a depth-map rectangle
        (inclusive; default: the whole frame), every `step` pixels."""
        x1 = self.w - 1 if x1 is None else x1
        y1 = self.h - 1 if y1 is None else y1
        ys, xs = np.mgrid[y0:y1 + 1:step, x0:x1 + 1:step]
        d = self.depth[ys, xs].astype(np.float64)
        ok = (self.confidence[ys, xs] >= MIN_CONFIDENCE) & np.isfinite(d) & (d > min_d) & (d < max_d)
        d, xs, ys = d[ok], xs[ok], ys[ok]
        # Depth pixel -> camera space (image y down, camera +Y up, looks along -Z) -> room.
        cam = np.stack([(xs + 0.5 - self.cx) / self.fx * d, -((ys + 0.5 - self.cy) / self.fy * d),
                        -d, np.ones_like(d)])
        w = self.T @ cam
        return w[0], w[1], w[2]

    def relative(self, x, z):
        """Room x/z -> (ahead, lateral) from the camera along the walking direction (+ = right)."""
        rel = np.stack([np.asarray(x) - self.cam_pos[0], np.asarray(z) - self.cam_pos[2]])
        return self.forward @ rel, self.right @ rel

    def to_world(self, ahead, lateral):
        """(ahead, lateral) from the camera -> room (x, z)."""
        return (float(self.cam_pos[0] + self.forward[0] * ahead + self.right[0] * lateral),
                float(self.cam_pos[2] + self.forward[1] * ahead + self.right[1] * lateral))

    def body_band(self, y, low=0.15, high=0.3):
        """Heights a walking person can bump into: above the floor, up to just over the head."""
        return (y > self.floor_y + low) & (y < self.cam_pos[1] + high)

    # ---------------------------------------------------------------- boxes
    def sensor_rect(self, box):
        """Normalized upright-portrait box [u1,v1,u2,v2] -> depth-map pixel rect.
        Portrait (u, v) maps to sensor (sx, sy) with sx = v, sy = 1 - u."""
        u1, v1, u2, v2 = box
        bw, bh = u2 - u1, v2 - v1
        u1, u2 = u1 + bw * BOX_SHRINK, u2 - bw * BOX_SHRINK
        v1, v2 = v1 + bh * BOX_SHRINK, v2 - bh * BOX_SHRINK
        x0, x1 = int(v1 * self.w), int(v2 * self.w)
        y0, y1 = int((1 - u2) * self.h), int((1 - u1) * self.h)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(self.w - 1, x1), min(self.h - 1, y1)
        return x0, y0, x1, y1

    def measure(self, box):
        """Returns (distance_m, lateral_m, in_path) or (None, None, False)."""
        x0, y0, x1, y1 = self.sensor_rect(box)
        if x1 < x0 or y1 < y0:
            return None, None, False
        x, y, z = self.world(x0, y0, x1, y1)
        if x.size < 8:
            return None, None, False
        ahead, lateral = self.relative(x, z)
        keep = (y > self.floor_y + FLOOR_MARGIN) & (ahead > 0.05)
        if keep.sum() < 8:
            return None, None, False
        ahead, lateral = np.sort(ahead[keep]), np.sort(lateral[keep])
        near = float(ahead[len(ahead) // 4])        # 25th percentile = object's near surface
        side = float(lateral[len(lateral) // 2])    # median sideways offset
        in_path = float(np.mean(np.abs(lateral) < self.half_width)) >= 0.10
        return near, side, in_path

    def bearing(self, box):
        """Horizontal direction of the box centre relative to the walking direction, in degrees
        (+ = right). Uses only the camera pose and lens, so it works at any distance, even
        beyond LiDAR range."""
        u, v = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        px, py = v * self.w, (1 - u) * self.h          # portrait (u, v) -> sensor pixel
        ray = self.T[:3, :3] @ np.array([(px - self.cx) / self.fx, -(py - self.cy) / self.fy, -1.0])
        flat = np.array([ray[0], ray[2]])
        if np.linalg.norm(flat) < 1e-6:
            return None
        return float(np.degrees(np.arctan2(self.right @ flat, self.forward @ flat)))

    # ---------------------------------------------------------------- whole frame
    def space_hint(self):
        """Coarse shape of the space ahead from this frame's depth (used to tell a hallway
        from a room): is it open straight ahead, and are there walls close on each side?"""
        x, y, z = self.world(step=4, min_d=0.2)
        if x.size < 200:
            return None
        keep = self.body_band(y, low=0.3)                  # waist-to-head band
        ahead, lateral = self.relative(x[keep], z[keep])
        center = ahead[np.abs(lateral) < 0.4]
        open_ahead = center.size < 15 or np.percentile(center, 10) > 3.5
        band = (ahead > 1.5) & (ahead < 5.0)
        left = (lateral < -0.4) & (lateral > -2.2) & band
        right = (lateral > 0.4) & (lateral < 2.2) & band
        return {"open_ahead": bool(open_ahead), "wall_left": int(left.sum()) >= 30,
                "wall_right": int(right.sum()) >= 30}
