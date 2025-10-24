#!/usr/bin/env python3
"""
Sample selector for multi-frame evaluation.

Selects continuous sequences of frames with consistent triplet configurations.
"""

import sys
import os
from pathlib import Path
from typing import List, Optional, Dict
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))
from benchmark.benchmark_config import BenchmarkConfig
from benchmark.cholect50_utils import CholecT50Loader

@dataclass
class MultiFrameSample:
    """Sample with multiple frames for temporal evaluation"""
    video_id: int
    start_frame: int
    end_frame: int
    clip_start: int
    image_paths: List[Path]  # Multiple frames
    graph_path: Optional[Path]
    # TODO: For now, loading these also for non-triplet tasks; might change this
    gt_triplets: List[Dict]  # Ground truth for the sequence
    gt_phase: Optional[str]
    
    @property
    def sample_id(self) -> str:
        return f"v{self.video_id:02d}_f{self.start_frame:05d}-{self.end_frame:05d}"
    
    @property
    def num_frames(self) -> int:
        return len(self.image_paths)


class TripletsFrameSelector:
    """Select multi-frame samples for evaluation"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.loader = CholecT50Loader(config.cholect50_root)
        video_id, clip_start = self._find_video_data(config.video_dir)
        self.available_graph = (video_id, clip_start, config.graph_dir)

    def _find_video_data(self, video_dir):
        video_id = int(video_dir.name.split("_")[0].replace("video", ""))
        print(f"video_id: {video_id}")
        clip_name = video_dir.name
        print(f"clip_name: {clip_name}")
        clip_start = int(clip_name.split("_")[1])
        return video_id, clip_start
            
    def select_sequences(self) -> List[MultiFrameSample]:
        """
        Select multi-frame sequences for evaluation.
        
        Args:
            num_sequences: Number of sequences to select
            frames_per_sequence: Number of frames in each sequence
            min_config_length: Minimum length of consistent triplet configuration
        """
        video_id, clip_start, graph_path = self.available_graph

        # Currently, the frame ids in the cholecseg8k dataset are relative to the clip start -> have to add and then divide by the framerate
        FRAMERATE = self.config.triplets_config['FRAMERATE']
        NUM_FRAMES = self.config.triplets_config['NUM_FRAMES']
        TEMP_CONTEXT_FR_MULTIPLIER = self.config.triplets_config['TEMP_CONTEXT_FR_MULTIPLIER']
        
        # Need the video id here to go with that into cholect50; does not correspond to the same frames, different fps!
        video_data = self.loader.load_video_annotations(video_id)
        print(f"Loaded video data for video {video_id}")
            
        # Find image paths
        clip_dir = self.config.video_dir
            
        # Get image paths for selected frames
        image_paths = []
        # image paths are all images in the clip directory
        image_paths = list(sorted(Path(os.path.join(clip_dir, "images")).glob("*.jpg")))
        print(f"first 10 image_paths: {image_paths[:10]}")

        samples = []
        for i in range(clip_start, clip_start + NUM_FRAMES):
            if i % FRAMERATE == 0:
                # Conversion from video framerate to triplet annotation framerate (cholecT50)
                frame_triplet = i // FRAMERATE
                triplets = self.loader.get_frame_triplets(video_data, frame_triplet)
                print(f"triplets for frame {i}/{frame_triplet}: {triplets}")
                end_frame = (i - clip_start) # Convert i from real video frame id to list index
                start_frame = end_frame - TEMP_CONTEXT_FR_MULTIPLIER * FRAMERATE
                if start_frame < 0:
                    start_frame = 0
                samples.append(MultiFrameSample(
                    video_id=video_id,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    clip_start=clip_start,
                    image_paths=image_paths,
                    graph_path=graph_path,
                    gt_triplets=triplets,
                    # TODO: not sure about this
                    gt_phase=triplets[0]['phase'] if triplets else None
                ))
        
        return samples
    
    def print_summary(self, samples: List[MultiFrameSample]):
        """Print summary of selected samples"""
        print("\n" + "="*80)
        print("SELECTED MULTI-FRAME SAMPLES")
        print("="*80)
        print(f"\nTotal samples: {len(samples)}")
        print(f"Frames per sample: {samples[0].num_frames if samples else 0}")
        print()
        
        # for i, sample in enumerate(samples, 1):
        #     print(f"{i}. {sample.sample_id}")
        #     print(f"   Video: {sample.video_id}, Frames: {sample.start_frame}-{sample.end_frame}")
        #     print(f"   Triplets: {[t['triplet_name'] for t in sample.gt_triplets]}")
        #     print(f"   Graph: {'Yes' if sample.graph_path else 'No'}")
        
        print("="*80)

