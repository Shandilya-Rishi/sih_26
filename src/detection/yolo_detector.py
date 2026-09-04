from pathlib import Path
from typing import Any

from ultralytics import YOLO


class YOLODetector:
    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        confidence_threshold: float = 0.25,
    ) -> None:
        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold

    def detect(self, frame: Any) -> list[dict[str, Any]]:
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            verbose=False,
        )

        detections: list[dict[str, Any]] = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                class_id = int(box.cls.item())
                confidence = float(box.conf.item())
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]

                detections.append(
                    {
                        "class_id": class_id,
                        "class_name": self.model.names[class_id],
                        "confidence": confidence,
                        "bbox": {
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                        },
                    }
                )

        return detections
