"""BoT-SORT state belonging to one WebSocket camera, updated in the GPU worker."""

from types import SimpleNamespace


class _IndexedBoxes:
    """Preserve input indices through BoT-SORT's high/low confidence splits."""

    def __init__(self, boxes, indices):
        self.boxes = boxes
        self.source_indices = indices

    def __len__(self):
        return len(self.boxes)

    def __getitem__(self, key):
        return _IndexedBoxes(self.boxes[key], self.source_indices[key])

    def __getattr__(self, name):
        return getattr(self.boxes, name)


class CameraTracker:
    # Starting values; tune using footage from the actual camera.
    low_confidence = 0.10
    high_confidence = 0.25
    retention_seconds = 10.0
    track_buffer = 900  # Also cap retention in processed frames.

    def __init__(self):
        self._tracker = None
        self._shape = None
        self._target = None
        self._last_update = None
        self._seen_at = {}
        self.generation = 0

    def _create(self, confidence):
        from ultralytics.trackers.bot_sort import BOTSORT
        import torch

        class CameraBOTSORT(BOTSORT):
            def init_track(self, results, img=None):
                tracks = super().init_track(results, img)
                # Older releases number each confidence subset from zero,
                # which can attach a low-score track to the wrong UI box.
                for track, index in zip(tracks, results.source_indices):
                    track.idx = int(index)
                return tracks

            @staticmethod
            def reset_id():
                # Ultralytics uses a process-wide ID counter. Opening another
                # camera must not reset it and reuse IDs in an existing camera.
                pass

        args = SimpleNamespace(
            tracker_type="botsort",
            track_high_thresh=min(self.high_confidence, confidence),
            track_low_thresh=self.low_confidence,
            new_track_thresh=confidence,
            track_buffer=self.track_buffer,
            match_thresh=0.8,
            fuse_score=True,
            gmc_method="sparseOptFlow",
            proximity_thresh=0.5,
            appearance_thresh=0.8,
            with_reid=True,
            model="yolo26n-reid.onnx",
            device=torch.device("cuda:0"),
        )
        # Calling with just args works with both older frame_rate=30 versions
        # and newer Ultralytics constructors that no longer accept frame_rate.
        self._tracker = CameraBOTSORT(args)
        self._seen_at.clear()
        self.generation += 1

    def update(self, boxes, image, names, target, confidence, now):
        """Return {original detection index: track ID} and live track metadata.

        boxes is an Ultralytics Boxes object with NumPy data in pixel units.
        All matching and image motion estimation stays local to this camera.
        """
        import numpy as np

        shape = image.shape[:2]
        if (
            self._tracker is None
            or shape != self._shape
            or target != self._target
            or (
                self._last_update is not None
                and now - self._last_update > self.retention_seconds
            )
        ):
            self._create(confidence)
        self._shape, self._target, self._last_update = shape, target, now
        tracker = self._tracker
        tracker.args.new_track_thresh = confidence
        tracker.args.track_high_thresh = min(self.high_confidence, confidence)

        # A frame buffer alone lasts too long at a slow camera rate. Expire
        # tracks by real elapsed time as well, before they can be reassociated.
        for pool in ("tracked_stracks", "lost_stracks"):
            retained = []
            for track in getattr(tracker, pool):
                if now - self._seen_at.get(track.track_id, now) > self.retention_seconds:
                    track.mark_removed()
                else:
                    retained.append(track)
            setattr(tracker, pool, retained)

        # Track only the requested label. BoT-SORT's spatial matching must not
        # turn a nearby chair into the selected table when their boxes overlap.
        indices = np.asarray([
            i for i, class_id in enumerate(boxes.cls)
            if names[int(class_id)] == target
        ], dtype=int)
        tracks = tracker.update(
            _IndexedBoxes(boxes[indices], np.arange(len(indices))), image,
        )
        assignments = {}
        for row in tracks:
            # Axis-aligned output: x1,y1,x2,y2,ID,score,class,input-index.
            local_index = int(row[-1])
            if 0 <= local_index < len(indices):
                assignments[int(indices[local_index])] = int(row[4])

        for track in tracker.tracked_stracks:
            if track.frame_id == tracker.frame_id:
                self._seen_at[track.track_id] = now
        live = tracker.tracked_stracks + tracker.lost_stracks
        live_ids = {int(track.track_id) for track in live}
        self._seen_at = {
            track_id: seen for track_id, seen in self._seen_at.items()
            if track_id in live_ids
        }
        return assignments, {
            "generation": self.generation,
            "alive_ids": sorted(live_ids),
            "retention_seconds": self.retention_seconds,
        }
