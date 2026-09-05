from src.intelligence.traffic_density import TrafficDensityEngine


def make_vehicle(x1, y1, x2, y2):
    return {
        "track_id": 1,
        "class_id": 2,
        "class_name": "car",
        "confidence": 0.9,
        "bbox": {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
        },
        "center": {
            "x": (x1 + x2) / 2,
            "y": (y1 + y2) / 2,
        },
    }


def test_empty_road_is_low_density():
    engine = TrafficDensityEngine()

    result = engine.calculate([], 100000)

    assert result.active_vehicle_count == 0
    assert result.occupancy_ratio == 0.0
    assert result.density_score == 0.0
    assert result.status == "LOW"


def test_density_increases_with_more_vehicles():
    engine = TrafficDensityEngine()

    vehicles = [
        make_vehicle(10, 10, 110, 110),
        make_vehicle(200, 100, 300, 200),
        make_vehicle(400, 200, 500, 300),
    ]

    result = engine.calculate(vehicles, 100000)

    assert result.active_vehicle_count == 3
    assert result.density_score > 0
    assert result.occupancy_ratio > 0


def test_invalid_roi_raises_error():
    engine = TrafficDensityEngine()

    try:
        engine.calculate([], 0)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected ValueError for zero ROI area."
        )