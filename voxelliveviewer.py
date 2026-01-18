"""
voxelliveviewer.py

Live voxelized video viewer.

Requirements:
    pip install opencv-python pyvista numpy

Usage examples:
    python voxelliveviewer.py --list-devices
    python voxelliveviewer.py --device 0
    python voxelliveviewer.py --file /path/to/video.mp4
    python voxelliveviewer.py --device 0 --motion-only --motion-output motion.mp4

If no device or file is provided, the script will prompt you to choose.
Press "q" in the PyVista window to quit.
"""

from __future__ import annotations

import argparse
import os
from typing import List, Tuple, Union

import cv2
import numpy as np
import pyvista as pv


def list_video_devices(max_devices: int = 6) -> List[int]:
    """Return a list of device indices that can be opened."""
    available = []
    for idx in range(max_devices):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(idx)
        cap.release()
    return available


def prompt_for_source() -> Union[int, str]:
    devices = list_video_devices()
    if devices:
        print("Available camera indices:", ", ".join(str(d) for d in devices))
    else:
        print("No camera devices detected by OpenCV.")

    while True:
        selection = input("Enter camera index or video file path: ").strip()
        if selection.isdigit():
            return int(selection)
        if os.path.isfile(selection):
            return selection
        print("Invalid selection. Try a device index or existing file path.")


def frame_to_voxel_points(
    frame: np.ndarray,
    grid_width: int,
    grid_height: int,
    depth: int,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (grid_width, grid_height), interpolation=cv2.INTER_AREA)
    norm = resized.astype(np.float32) / 255.0
    voxel_heights = (norm * depth).astype(np.int32)

    points = []
    intensities = []
    for y in range(grid_height):
        for x in range(grid_width):
            height = voxel_heights[y, x]
            if height <= 0 or norm[y, x] < threshold:
                continue
            for z in range(height):
                points.append((x, y, z))
                intensities.append(height)

    if not points:
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.float32)

    points_arr = np.array(points, dtype=np.float32)
    intensities_arr = np.array(intensities, dtype=np.float32)

    points_arr[:, 0] -= grid_width / 2.0
    points_arr[:, 1] -= grid_height / 2.0
    points_arr[:, 2] -= depth / 2.0

    return points_arr, intensities_arr


def open_capture(source: Union[int, str]) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video source: {source}")
    return cap


def main() -> None:
    parser = argparse.ArgumentParser(description="Live voxelized video viewer")
    parser.add_argument("--list-devices", action="store_true", help="List camera indices")
    parser.add_argument("--device", type=int, help="Camera device index")
    parser.add_argument("--file", type=str, help="Path to a video file")
    parser.add_argument("--grid-width", type=int, default=64, help="Voxel grid width")
    parser.add_argument("--grid-height", type=int, default=48, help="Voxel grid height")
    parser.add_argument("--depth", type=int, default=32, help="Voxel depth")
    parser.add_argument("--threshold", type=float, default=0.1, help="Brightness threshold")
    parser.add_argument("--motion-only", action="store_true", help="Show motion-only view (2D)")
    parser.add_argument("--motion-threshold", type=int, default=25, help="Motion diff threshold")
    parser.add_argument("--motion-output", type=str, help="Optional output video file (MP4)")
    args = parser.parse_args()

    if args.list_devices:
        devices = list_video_devices()
        if devices:
            print("Available camera indices:", ", ".join(str(d) for d in devices))
        else:
            print("No camera devices detected by OpenCV.")
        return

    source: Union[int, str]
    if args.device is not None:
        source = args.device
    elif args.file:
        if not os.path.isfile(args.file):
            raise FileNotFoundError(f"Video file not found: {args.file}")
        source = args.file
    else:
        source = prompt_for_source()

    cap = open_capture(source)

    if args.motion_only:
        prev_gray = None
        writer = None
        if args.motion_output:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(args.motion_output, fourcc, fps, (width, height), False)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if prev_gray is None:
                    prev_gray = gray
                    continue

                diff = cv2.absdiff(prev_gray, gray)
                _, motion = cv2.threshold(diff, args.motion_threshold, 255, cv2.THRESH_TOZERO)
                prev_gray = gray

                cv2.imshow("Motion only", motion)
                if writer is not None:
                    writer.write(motion)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            cv2.destroyAllWindows()
        return

    plotter = pv.Plotter()
    plotter.set_background("black")
    plotter.add_axes()

    should_close = {"value": False}

    def _close():
        should_close["value"] = True

    plotter.add_key_event("q", _close)
    plotter.show(auto_close=False, interactive_update=True)

    try:
        while not should_close["value"]:
            ret, frame = cap.read()
            if not ret:
                break

            points, intensities = frame_to_voxel_points(
                frame,
                grid_width=args.grid_width,
                grid_height=args.grid_height,
                depth=args.depth,
                threshold=args.threshold,
            )

            plotter.clear()
            if points.size:
                cloud = pv.PolyData(points)
                cloud["intensity"] = intensities
                plotter.add_points(
                    cloud,
                    scalars="intensity",
                    render_points_as_spheres=True,
                    point_size=6.0,
                    cmap="hot",
                )

            plotter.update()

    finally:
        cap.release()
        plotter.close()


if __name__ == "__main__":
    main()
