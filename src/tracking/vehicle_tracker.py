from __future__ import annotations

from typing import Any

from ultralytics import YOLO


class VehicleTracker:
    """
    Vehicle tracking using Ultralytics YOLO + ByteTrack.

    Tracks only:
    - car
    - motorcycle
    - bus
    - truck
    """

    VEHICLE_CLASSES = {
        "car",
        "motorcycle",
        "bus",
        "truck",
    }

    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        confidence_threshold: float = 0.25,
    ) -> None:
        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold

    @staticmethod
    def _center_from_bbox(
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> dict[str, float]:
        return {
            "x": (x1 + x2) / 2.0,
            "y": (y1 + y2) / 2.0,
        }

    def track(self, frame: Any) -> list[dict[str, Any]]:
        """
        Track vehicles in a single frame.

        Returns a list of structured tracked objects.
        """

        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.confidence_threshold,
            verbose=False,
        )

        tracked_objects: list[dict[str, Any]] = []

        for result in results:
            if result.boxes is None or result.boxes.id is None:
                continue

            boxes = result.boxes
            track_ids = boxes.id.int().cpu().tolist()
            class_ids = boxes.cls.int().cpu().tolist()
            confidences = boxes.conf.cpu().tolist()
            coordinates = boxes.xyxy.cpu().tolist()

            for track_id, class_id, confidence, bbox in zip(
                track_ids,
                class_ids,
                confidences,
                coordinates,
            ):
                class_name = self.model.names[int(class_id)]

                if class_name not in self.VEHICLE_CLASSES:
                    continue

                x1, y1, x2, y2 = [float(value) for value in bbox]

                tracked_objects.append(
                    {
                        "track_id": int(track_id),
                        "class_id": int(class_id),
                        "class_name": class_name,
                        "confidence": float(confidence),
                        "bbox": {
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                        },
                        "center": self._center_from_bbox(
                            x1,
                            y1,
                            x2,
                            y2,
                        ),
                    }
                )

        return tracked_objects