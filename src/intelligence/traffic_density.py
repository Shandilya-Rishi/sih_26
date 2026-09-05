from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TrafficDensityResult:
    active_vehicle_count: int
    occupancy_ratio: float
    density_score: float
    status: str


class TrafficDensityEngine:
    """
    MVP traffic-density estimator.

    Combines active vehicle count and road-region occupancy
    into a 0-100 heuristic density score.
    """

    STATUS_THRESHOLDS = {
        "LOW": 25.0,
        "MODERATE": 50.0,
        "HIGH": 75.0,
    }

    def __init__(
        self,
        max_vehicle_count: int = 30,
        max_occupancy: float = 0.30,
    ) -> None:
        self.max_vehicle_count = max_vehicle_count
        self.max_occupancy = max_occupancy

    @staticmethod
    def _bbox_area(bbox: dict[str, float]) -> float:
        width = max(0.0, bbox["x2"] - bbox["x1"])
        height = max(0.0, bbox["y2"] - bbox["y1"])
        return width * height

    def calculate(
        self,
        tracked_objects: list[dict[str, Any]],
        roi_area: float,
    ) -> TrafficDensityResult:
        if roi_area <= 0:
            raise ValueError("ROI area must be greater than zero.")

        active_vehicle_count = len(tracked_objects)

        total_vehicle_area = sum(
            self._bbox_area(obj["bbox"])
            for obj in tracked_objects
        )

        occupancy_ratio = min(
            total_vehicle_area / roi_area,
            1.0,
        )

        count_component = min(
            active_vehicle_count / self.max_vehicle_count,
            1.0,
        )

        occupancy_component = min(
            occupancy_ratio / self.max_occupancy,
            1.0,
        )

        density_score = (
            0.6 * count_component
            + 0.4 * occupancy_component
        ) * 100.0

        if density_score >= self.STATUS_THRESHOLDS["HIGH"]:
            status = "SEVERE"
        elif density_score >= self.STATUS_THRESHOLDS["MODERATE"]:
            status = "HIGH"
        elif density_score >= self.STATUS_THRESHOLDS["LOW"]:
            status = "MODERATE"
        else:
            status = "LOW"

        return TrafficDensityResult(
            active_vehicle_count=active_vehicle_count,
            occupancy_ratio=occupancy_ratio,
            density_score=density_score,
            status=status,
        )