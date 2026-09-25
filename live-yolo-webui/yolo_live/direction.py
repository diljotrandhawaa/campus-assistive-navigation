"""Camera alignment from normalized boxes; this does not estimate a walking route."""
from dataclasses import dataclass
import math



# CLASSES = (
#     "table",
#     "door", "glass door", "wooden door", "sliding door", "revolved door", "trapdoor", "door handle", "garage door", "brown door", "green door", "white door", "office door", "classroom door"
#     "doorknob", "door handle", "handle door", "door lever", "door latch", "pull handle", "push handle", "push bar", "panic bar", "door pull", "door hardware", "door lock cylinder", "handicap push button"
#     "elevator",
#     "chair", "whiteboard", "sign", "stairs", "person", "backpack", "cardboard box", "trash can", "sofa", "bench", "window", "toilet", "monitor", "laptop", "keyboard", "printer"
# )

CLASSES = (
    "table",
    "door",
    # "glass door",
    # "wooden door",
    "sliding door",
    "revolving door",
    "trapdoor",
    "door handle",
    "garage door",
    "brown door",
    "green door",
    "white door",
    "office door",
    "classroom door",
    "doorknob",
    "handle door",
    "door lever",
    "door latch",
    "pull handle",
    "push handle",
    "push bar",
    "panic bar",
    "door pull",
    "door hardware",
    "door lock cylinder",
    "handicap push button",
    "elevator",
    "elevator door",
    "chair",
    "whiteboard",
    "sign",
    "stairs",
    "person",
    "backpack",
    "cardboard box",
    "trash can",
    "sofa",
    "bench",
    "window",
    "toilet",
    "monitor",
    "laptop",
    "keyboard",
    "printer",
    "empty chair",
    "restroom sign",
    "wall"
)

def position(box):
    x, y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    horizontal = "left" if x < 1 / 3 else "right" if x > 2 / 3 else "center"
    vertical = "top" if y < 1 / 3 else "bottom" if y > 2 / 3 else "middle"
    return f"{vertical}-{horizontal}"


def overlap(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1e-9)


@dataclass
class AlignmentController:
    target: str = "whiteboard"
    enter_band: float = 0.05
    exit_band: float = 0.08
    smoothing: float = 0.65
    forget_after: float = 0.7
    previous_box: tuple | None = None
    last_seen: float | None = None
    smoothed_offset: float | None = None
    state: str = "lost"

    def _lost(self):
        # Do not repeat an old directional command when the target disappears.
        self.state = "lost"
        self.smoothed_offset = None
        return {"state": "lost", "text": f"{self.target.capitalize()} not detected",
                "target_index": None, "offset": None, "raw_offset": None,
                "matches": 0, "enter_band": self.enter_band, "exit_band": self.exit_band}

    def update(self, detections, now):
        if self.last_seen is not None and now - self.last_seen > self.forget_after:
            self.previous_box = None
            self.smoothed_offset = None
            self.state = "lost"
        candidates = [(i, d) for i, d in enumerate(detections) if d["label"] == self.target]
        if not candidates:
            return self._lost()

        if self.previous_box is None:
            index, selected = max(candidates, key=lambda item: item[1]["confidence"])
        else:
            old = self.previous_box
            old_x, old_y = (old[0] + old[2]) / 2, (old[1] + old[3]) / 2
            nearby = []
            for i, d in candidates:
                box = d["box"]
                distance = math.hypot((box[0] + box[2]) / 2 - old_x,
                                      (box[1] + box[3]) / 2 - old_y)
                iou = overlap(old, box)
                if iou > 0.05 or distance < 0.18:
                    nearby.append((2 * iou - distance, i, d))
            if not nearby:
                # Briefly report missing, rather than jump to another identical object.
                return self._lost()
            _, index, selected = max(nearby, key=lambda item: item[0])

        self.previous_box = tuple(selected["box"])
        self.last_seen = now
        raw = (self.previous_box[0] + self.previous_box[2]) / 2 - 0.5
        previous = self.smoothed_offset
        if (previous is None or abs(raw - previous) > 0.12 or
                abs(raw) <= self.enter_band or raw * previous < 0):
            filtered = raw
        else:
            filtered = self.smoothing * raw + (1 - self.smoothing) * previous
        self.smoothed_offset = filtered

        # A wider exit band prevents jitter when already centered. Use raw position
        # for exiting so smoothing cannot prolong "centered" outside that band.
        if self.state == "centered" and abs(raw) <= self.exit_band:
            state = "centered"
        elif abs(filtered) <= self.enter_band:
            state = "centered"
        else:
            state = "right" if filtered > 0 else "left"
        self.state = state
        message = {"left": "Pan camera left", "right": "Pan camera right",
                   "centered": "Target centered"}[state]
        return {"state": state, "text": message, "target_index": index,
                "offset": round(filtered, 4), "raw_offset": round(raw, 4),
                "matches": len(candidates), "enter_band": self.enter_band,
                "exit_band": self.exit_band}

