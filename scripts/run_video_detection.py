import argparse
import time
from collections import Counter
from pathlib import Path

import cv2

from src.detection.yolo_detector import YOLODetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pretrained YOLO detection on a road video."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("datasets/raw/test_road.mp4"),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/test_road_yolo.mp4"),
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
    )

    return parser.parse_args()


def draw_detections(frame, detections):
    for detection in detections:
        bbox = detection["bbox"]

        x1 = int(bbox["x1"])
        y1 = int(bbox["y1"])
        x2 = int(bbox["x2"])
        y2 = int(bbox["y2"])

        label = (
            f'{detection["class_name"]} '
            f'{detection["confidence"]:.2f}'
        )

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Input video not found: {args.input}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(args.input))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {args.input}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        fps = 25.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(args.output),
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(
            f"Could not create output video: {args.output}"
        )

    detector = YOLODetector(
        model_name=args.model,
        confidence_threshold=args.confidence,
    )

    frame_count = 0
    class_counts: Counter[str] = Counter()

    start_time = time.perf_counter()

    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            detections = detector.detect(frame)

            for detection in detections:
                class_counts[detection["class_name"]] += 1

            draw_detections(frame, detections)
            writer.write(frame)

            frame_count += 1

    finally:
        cap.release()
        writer.release()

    elapsed = time.perf_counter() - start_time
    processing_fps = frame_count / elapsed if elapsed > 0 else 0.0

    print()
    print("=== URBAN-EYE YOLO VIDEO INFERENCE ===")
    print(f"Input:            {args.input}")
    print(f"Output:           {args.output}")
    print(f"Frames processed: {frame_count}")
    print(f"Elapsed time:     {elapsed:.2f} seconds")
    print(f"Processing FPS:   {processing_fps:.2f}")
    print()
    print("Detection counts:")

    if class_counts:
        for class_name, count in class_counts.most_common():
            print(f"  {class_name}: {count}")
    else:
        print("  No objects detected.")


if __name__ == "__main__":
    main()
