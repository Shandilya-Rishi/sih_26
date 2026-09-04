from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import cv2

from src.tracking.vehicle_tracker import VehicleTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO + ByteTrack vehicle tracking on a road video."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("datasets/raw/test_road.mp4"),
        help="Path to input video.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/test_road_tracking.mp4"),
        help="Path to output video.",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="YOLO confidence threshold.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="Ultralytics model name/path.",
    )

    return parser.parse_args()


def draw_tracks(
    frame,
    tracked_objects: list[dict],
) -> None:
    for obj in tracked_objects:
        bbox = obj["bbox"]

        x1 = int(bbox["x1"])
        y1 = int(bbox["y1"])
        x2 = int(bbox["x2"])
        y2 = int(bbox["y2"])

        class_name = obj["class_name"]
        confidence = obj["confidence"]
        track_id = obj["track_id"]

        label = f"{class_name.upper()} #{track_id} {confidence:.2f}"

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
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

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    capture = cv2.VideoCapture(str(args.input))

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open input video: {args.input}"
        )

    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        fps = 25.0

    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        capture.release()
        raise RuntimeError(
            f"Could not create output video: {args.output}"
        )

    tracker = VehicleTracker(
        model_name=args.model,
        confidence_threshold=args.confidence,
    )

    # Track IDs observed by class.
    unique_ids: dict[str, set[int]] = defaultdict(set)

    frame_count = 0
    start_time = time.perf_counter()

    try:
        while True:
            success, frame = capture.read()

            if not success:
                break

            tracked_objects = tracker.track(frame)

            for obj in tracked_objects:
                unique_ids[obj["class_name"]].add(
                    obj["track_id"]
                )

            draw_tracks(
                frame,
                tracked_objects,
            )

            writer.write(frame)
            frame_count += 1

    finally:
        capture.release()
        writer.release()

    elapsed = time.perf_counter() - start_time
    processing_fps = (
        frame_count / elapsed
        if elapsed > 0
        else 0.0
    )

    print()
    print("=== URBAN-EYE VEHICLE TRACKING ===")
    print(f"Input:            {args.input}")
    print(f"Output:           {args.output}")
    print(f"Frames processed: {frame_count}")
    print(f"Elapsed time:     {elapsed:.2f} seconds")
    print(f"Processing FPS:   {processing_fps:.2f}")
    print()
    print("Unique tracked vehicles:")

    for class_name in (
        "car",
        "motorcycle",
        "bus",
        "truck",
    ):
        print(
            f"  {class_name}: "
            f"{len(unique_ids[class_name])}"
        )


if __name__ == "__main__":
    main()