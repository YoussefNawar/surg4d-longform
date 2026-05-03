"""
Sample every 5th frame from a Cholec80 video.
Extracts frames 1, 6, 11, 16, ... (1-indexed) i.e. indices 0, 5, 10, 15, ... (0-indexed).

Usage:
    python sample_cholec80.py --video /path/to/video01.mp4 --output /path/to/output_dir
    python sample_cholec80.py --video /path/to/video01.mp4 --output /path/to/output_dir --step 5
"""

import cv2
import numpy as np
from pathlib import Path


def estimate_crop_box_rgb(rgb: np.ndarray, black_threshold: int = 10) -> tuple[int, int, int, int]:
    """Get crop box from RGB image by detecting black circular camera borders.

    Args:
        rgb: HxWx3 uint8 RGB array
        black_threshold: pixels with all channels below this are considered black

    Returns:
        top, bottom, left, right crop indices
    """
    is_content = np.any(rgb > black_threshold, axis=2)

    mid_col = is_content[:, is_content.shape[1] // 2]
    top = int(mid_col.argmax())
    bottom = int(mid_col.size - np.flip(mid_col).argmax())

    is_content_cropped = is_content[top:bottom, :]
    first_row = is_content_cropped[0]
    left = int(first_row.argmax())
    right = int(first_row.size - np.flip(first_row).argmax())

    return top, bottom, left, right


def sample_video(video_path: str, output_dir: str, step: int = 5, crop: bool = True) -> None:
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

    crop_box = None
    if crop:
        ret, first_frame = cap.read()
        if not ret:
            raise RuntimeError("Cannot read first frame for crop estimation")
        crop_box = estimate_crop_box_rgb(first_frame)
        top, bottom, left, right = crop_box
        print(f"Crop box: top={top}, bottom={bottom}, left={left}, right={right}")
        print(f"Cropped resolution: {right - left}x{bottom - top}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    sampled = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            if crop_box is not None:
                top, bottom, left, right = crop_box
                frame = frame[top:bottom, left:right]

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