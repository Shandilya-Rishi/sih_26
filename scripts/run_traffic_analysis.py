from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from src.intelligence.traffic_density import TrafficDensityEngine
from src.tracking.vehicle_tracker import VehicleTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run vehicle tracking and traffic-density analysis."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("datasets/raw/test_road.mp4"),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/test_road_traffic.mp4"),
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

    parser.add_argument(
        "--road-ratio",
        type=float,
        default=0.70,
        help="Fraction of image height treated as road region from the bottom.",
    )

    return parser.parse_args()


def draw_tracks(
    frame,
    tracked_objects: list[dict],
    road_top: int,
) -> None:
    for obj in tracked_objects:
        bbox = obj["bbox"]

        x1 = int(bbox["x1"])
        y1 = int(bbox["y1"])
        x2 = int(bbox["x2"])
        y2 = int(bbox["y2"])

        label = (
            f'{obj["class_name"].upper()} '
            f'#{obj["track_id"]} '
            f'{obj["confidence"]:.2f}'
        )

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

    cv2.line(
        frame,
        (0, road_top),
        (frame.shape[1], road_top),
        (255, 255, 0),
        2,
    )


def draw_density_panel(
    frame,
    active_vehicle_count: int,
    occupancy_ratio: float,
    density_score: float,
    status: str,
) -> None:
    lines = [
        f"Traffic: {status}",
        f"Density Score: {density_score:.1f}/100",
        f"Active Vehicles: {active_vehicle_count}",
        f"Road Occupancy: {occupancy_ratio * 100:.1f}%",
    ]

    y = 30

    for line in lines:
        cv2.putText(
            frame,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 30


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Input video not found: {args.input}"
        )

    if not 0.0 < args.road_ratio <= 1.0:
        raise ValueError(
            "--road-ratio must be greater than 0 and at most 1."
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

    density_engine = TrafficDensityEngine()

    road_top = int(height * (1.0 - args.road_ratio))
    roi_area = width * (height - road_top)

    frame_count = 0
    total_density = 0.0
    maximum_density = 0.0

    status_counts = {
        "LOW": 0,
        "MODERATE": 0,
        "HIGH": 0,
        "SEVERE": 0,
    }

    start_time = time.perf_counter()

    try:
        while True:
            success, frame = capture.read()

            if not success:
                break

            tracked_objects = tracker.track(frame)

            road_objects = []

            for obj in tracked_objects:
                if obj["center"]["y"] >= road_top:
                    road_objects.append(obj)

            result = density_engine.calculate(
                road_objects,
                roi_area,
            )

            total_density += result.density_score
            maximum_density = max(
                maximum_density,
                result.density_score,
            )

            status_counts[result.status] += 1

            draw_tracks(
                frame,
                road_objects,
                road_top,
            )

            draw_density_panel(
                frame,
                result.active_vehicle_count,
                result.occupancy_ratio,
                result.density_score,
                result.status,
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

    average_density = (
        total_density / frame_count
        if frame_count > 0
        else 0.0
    )

    print()
    print("=== URBAN-EYE TRAFFIC ANALYSIS ===")
    print(f"Input:              {args.input}")
    print(f"Output:             {args.output}")
    print(f"Frames processed:   {frame_count}")
    print(f"Elapsed time:       {elapsed:.2f} seconds")
    print(f"Processing FPS:     {processing_fps:.2f}")
    print(f"Average density:    {average_density:.2f}/100")
    print(f"Maximum density:    {maximum_density:.2f}/100")
    print()
    print("Traffic status distribution:")

    for status, count in status_counts.items():
        print(f"  {status}: {count} frames")


if __name__ == "__main__":
    main()