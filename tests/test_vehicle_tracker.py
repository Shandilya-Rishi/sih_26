from src.tracking.vehicle_tracker import VehicleTracker


def test_center_calculation():
    center = VehicleTracker._center_from_bbox(
        10.0,
        20.0,
        110.0,
        220.0,
    )

    assert center["x"] == 60.0
    assert center["y"] == 120.0


def test_vehicle_classes():
    assert "car" in VehicleTracker.VEHICLE_CLASSES
    assert "motorcycle" in VehicleTracker.VEHICLE_CLASSES
    assert "bus" in VehicleTracker.VEHICLE_CLASSES
    assert "truck" in VehicleTracker.VEHICLE_CLASSES

    assert "person" not in VehicleTracker.VEHICLE_CLASSES


def test_tracker_import():
    assert VehicleTracker is not None