from src.detection.yolo_detector import YOLODetector


def test_detector_result_structure():
    detector = YOLODetector.__new__(YOLODetector)

    sample = {
        "class_id": 2,
        "class_name": "car",
        "confidence": 0.91,
        "bbox": {
            "x1": 10.0,
            "y1": 20.0,
            "x2": 100.0,
            "y2": 200.0,
        },
    }

    assert isinstance(sample["class_id"], int)
    assert isinstance(sample["class_name"], str)
    assert 0.0 <= sample["confidence"] <= 1.0
    assert set(sample["bbox"]) == {"x1", "y1", "x2", "y2"}
