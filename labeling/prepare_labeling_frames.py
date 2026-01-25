#!/usr/bin/env python3
"""
Script to prepare frames for labeling from the cholecseg8k dataset.
Copies frames from clips defined in Hydra clip config with a stride of 4.
Also creates full framerate MP4 videos for each clip.
"""

import shutil
from pathlib import Path
import subprocess
from typing import List
import hydra
from omegaconf import DictConfig, ListConfig


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Configuration (paths relative to repository root)
    repo_root = Path(__file__).parent.parent
    dataset_root = repo_root / "data/cholecseg8k"
    labeling_root = repo_root / "labeling/clips"
    frame_stride = 4

    # Get clips from Hydra config
    clips_config: ListConfig = cfg.clips
    clips_list: List[dict] = [dict(clip) for clip in clips_config]

    print("\n" + "=" * 60)
    print(f"Processing {len(clips_list)} clips from config")
    print("=" * 60)

    # Create output directory
    output_dir = labeling_root
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each clip
    for clip in clips_list:
        clip_name = clip["name"]
        video_id = clip["video_id"]
        first_frame = clip["first_frame"]
        last_frame = clip["last_frame"]

        print(f"\nProcessing clip: {clip_name}")

        # Parse video name (e.g., video_id=1 -> "video01")
        video_name = f"video{video_id:02d}"

        # Source directory
        source_dir = dataset_root / video_name / clip_name

        if not source_dir.exists():
            print(f"  WARNING: Source directory not found: {source_dir}")
            continue

        # last_frame is exclusive, so frames are [first_frame, last_frame)
        frames_per_clip = last_frame - first_frame

        print(f"  Processing {frames_per_clip} frames: {first_frame} to {last_frame - 1}")

        # Create clip directory in output
        clip_output_dir = output_dir / clip_name
        clip_output_dir.mkdir(parents=True, exist_ok=True)

        # Copy frames with stride (RGB images only)
        frames_copied = 0
        for i in range(0, frames_per_clip, frame_stride):
            frame_num = first_frame + i

            # Copy only RGB image
            source_file = source_dir / f"frame_{frame_num}_endo.png"
            dest_file = clip_output_dir / f"frame_{frame_num}_endo_{frames_copied:02d}.png"

            if source_file.exists():
                shutil.copy2(source_file, dest_file)
                frames_copied += 1
            else:
                print(f"  WARNING: File not found: {source_file}")

        print(f"  Copied {frames_copied} frames (stride={frame_stride}) from {clip_name}")

        # Create full framerate video from all frames using ffmpeg
        video_path = clip_output_dir / f"{clip_name}.mp4"
        print(f"  Creating video: {video_path.name}")

        # Create a temporary file list for ffmpeg
        file_list_path = clip_output_dir / "frame_list.txt"
        with open(file_list_path, 'w') as f:
            for i in range(frames_per_clip):
                frame_num = first_frame + i
                frame_path = source_dir / f"frame_{frame_num}_endo.png"
                if frame_path.exists():
                    f.write(f"file '{frame_path.absolute()}'\n")
                    f.write("duration 0.04\n")  # 25 fps = 0.04 seconds per frame

        # Use ffmpeg to create video from frame list
        try:
            cmd = [
                'ffmpeg', '-y',  # -y to overwrite output file
                '-f', 'concat',
                '-safe', '0',
                '-i', str(file_list_path),
                '-vsync', 'vfr',
                '-pix_fmt', 'yuv420p',
                '-c:v', 'libx264',
                '-crf', '23',
                str(video_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                print(f"  ✓ Video created with {frames_per_clip} frames")
            else:
                print(f"  ERROR creating video: {result.stderr}")
        except Exception as e:
            print(f"  ERROR: Failed to create video: {e}")
        finally:
            # Clean up temporary file list
            if file_list_path.exists():
                file_list_path.unlink()

    print(f"\n{'='*60}")
    print("Frame preparation complete!")
    print(f"{'='*60}")

    # Print summary
    print("\nSummary:")
    num_clips = len(list(output_dir.glob("*"))) if output_dir.exists() else 0
    print(f"  Total: {num_clips} clips")


if __name__ == "__main__":
    main()

