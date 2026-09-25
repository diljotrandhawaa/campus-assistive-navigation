"""Lock onto a BoT-SORT ID and give directions only for current detections."""

from dataclasses import dataclass

from .direction import AlignmentController, position


@dataclass
class TargetController(AlignmentController):
    horizontal: str | None = None
    target_track_id: int | None = None
    tracker_generation: int | None = None
    missing_grace: float = 1.0

    def _clear_lock(self):
        self.target_track_id = None
        self.previous_box = None
        self.last_seen = None
        self.smoothed_offset = None
        self.state = "lost"

    def update(self, detections, now, tracking=None, confidence=0.25):
        tracking = tracking or {}
        generation = tracking.get("generation")
        if generation != self.tracker_generation:
            self._clear_lock()
            self.tracker_generation = generation

        candidates = [
            (index, detection)
            for index, detection in enumerate(detections)
            if detection["label"] == self.target
        ]
        matches = len(candidates)

        if self.target_track_id is not None:
            for index, detection in candidates:
                if detection.get("track_id") == self.target_track_id:
                    return self._accept(index, detection, now, matches)

            age = now - self.last_seen if self.last_seen is not None else float("inf")
            alive = self.target_track_id in tracking.get("alive_ids", [])
            if alive and age <= tracking.get("retention_seconds", 3.0):
                # Keep the ID, but never present the last box/offset as a new
                # observation. "stale" also disables the existing UI marker.
                result = self._lost()
                result.update(
                    state="stale",
                    text=(
                        f"Tracking {self.target}…"
                        if age <= self.missing_grace
                        else f"{self.target.capitalize()} temporarily not visible"
                    ),
                    tracking_state="buffering",
                    target_track_id=self.target_track_id,
                    matches=matches,
                    speak=False,
                )
                return result
            self._clear_lock()

        # Confidence and side determine acquisition only. Once locked, a
        # matched ID may cross the screen or drop below the acquisition score.
        eligible = [
            (index, detection) for index, detection in candidates
            if detection.get("track_id") is not None
            and detection["confidence"] >= confidence
            and (
                self.horizontal is None
                or position(detection["box"]).split("-")[-1] == self.horizontal
            )
        ]
        if eligible:
            index, detection = max(eligible, key=lambda item: item[1]["confidence"])
            self.target_track_id = detection["track_id"]
            return self._accept(index, detection, now, matches)

        result = self._lost()
        result.update(
            text=f"Looking for a {self.target}…",
            tracking_state="searching",
            target_track_id=None,
            matches=matches,
            speak=True,
        )
        return result

    def _accept(self, index, detection, now, matches):
        # BoT-SORT has already associated this detection. Bypass the parent's
        # second nearby-box gate; retain its smoothing and centering logic.
        self.previous_box = None
        result = super().update([detection], now)
        result.update(
            target_index=index,
            target_track_id=self.target_track_id,
            tracking_state="tracking",
            matches=matches,
            speak=True,
        )
        return result
