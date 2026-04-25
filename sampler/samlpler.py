"""
Sample every 5th frame from a Cholec80 video.
Extracts frames 1, 6, 11, 16, ... (1-indexed) i.e. indices 0, 5, 10, 15, ... (0-indexed).

Usage:
    python sample_cholec80.py --video /path/to/video01.mp4 --output /path/to/output_dir
    python sample_cholec80.py --video /path/to/video01.mp4 --output /path/to/output_dir --step 5
"""

import os
import cv2
from pathlib import Path


def sample_video(video_path: str, output_dir: str, step: int = 5) -> None:
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {video_path.name}")
    print(f"Total frames: {total_frames}, FPS: {fps:.2f}, Resolution: {width}x{height}")
    print(f"Sampling every {step} frames (frames 1, {1+step}, {1+2*step}, ...)")

    sampled = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            # 1-indexed naming: frame 0 -> frame_00001, frame 5 -> frame_00006, etc.
            frame_number = frame_idx + 1
            filename = f"frame_{frame_number:06d}.png"
            cv2.imwrite(str(output_dir / filename), frame)
            sampled += 1

        frame_idx += 1

    cap.release()

    expected = (total_frames + step - 1) // step
    print(f"Done. Sampled {sampled} frames out of {total_frames} (expected ~{expected}).")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    video = '/home/data/cholec80/videos/video01.mp4'
    output = '/home/data/long_form_surgery_Cholec80/5fps_samples'
    step = 5
    sample_video(video, output, step)