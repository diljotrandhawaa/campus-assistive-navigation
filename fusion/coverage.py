"""Which directions the user has already looked at during a search.

The circle around the user is split into 12 slices of 30°. A slice counts as "checked" once the
camera has faced it (within half the field of view) for DWELL seconds. Headings come from the
ARKit walking direction the phone sends with every frame, so this costs nothing extra.

    cov = HeadingCoverage()
    new = cov.update(heading_deg(frame.forward), now)   # -> number of slices newly checked
    cov.count()                                           # -> 0..12
    cov.next_turn(heading)                                # -> ("right", 60.0) to the nearest unchecked slice
"""
import math

import numpy as np


def heading_deg(forward):
    """Walking direction (x, z) -> compass-like degrees: 0 = where ARKit started facing,
    90 = turned right, 270 = turned left."""
    return math.degrees(math.atan2(float(forward[0]), -float(forward[1]))) % 360.0


def signed_diff(a, b):
    """Smallest signed angle from b to a, in (-180, 180]; + = a is to the right of b."""
    return (a - b + 180.0) % 360.0 - 180.0


class HeadingCoverage:
    def __init__(self, bins=12, fov_deg=50.0, dwell_s=0.4):
        self.bins = bins
        self.width = 360.0 / bins
        self.half_fov = fov_deg / 2.0
        self.dwell = dwell_s
        self.reset()

    def reset(self):
        self.time = np.zeros(self.bins)
        self.last_t = None

    def centers(self):
        return (np.arange(self.bins) + 0.5) * self.width

    def update(self, heading, now):
        dt = 0.0 if self.last_t is None else min(max(now - self.last_t, 0.0), 0.3)
        self.last_t = now
        before = self.covered().sum()
        diffs = np.abs([signed_diff(c, heading) for c in self.centers()])
        self.time[diffs <= self.half_fov] += dt
        return int(self.covered().sum() - before)

    def covered(self):
        return self.time >= self.dwell

    def count(self):
        return int(self.covered().sum())

    def next_turn(self, heading):
        """Direction and angle to the nearest unchecked slice (None if all checked)."""
        missing = [c for c, ok in zip(self.centers(), self.covered()) if not ok]
        if not missing:
            return None
        best = min(missing, key=lambda c: abs(signed_diff(c, heading)))
        d = signed_diff(best, heading)
        return ("right" if d >= 0 else "left"), abs(d)
