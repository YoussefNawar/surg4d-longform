#!/usr/bin/env python3
"""
Sample selector for multi-frame evaluation.

Selects continuous sequences of frames with consistent triplet configurations.
"""

import sys
import os
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))
from benchmark.benchmark_config import BenchmarkConfig
from benchmark.cholect50_utils import CholecT50Loader

# Import from local module
try:
    from benchmark.multiframe_evaluator import MultiFrameSample
except ImportError:
    # Fallback if running as script
    from frame_evaluators import MultiFrameSample


class TripletsFrameSelector:
    """Select multi-frame samples for evaluation"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.loader = CholecT50Loader(config.cholect50_root)
        self.available_graph = self._find_available_graph(config.video_dir)
        # TODO: hack, fix later
        video_id, clip_start, wrong_graph_dir = self.available_graph
        print(f"wrong_graph_dir: {wrong_graph_dir}")
        graph_dir = config.graph_dir
        self.available_graph = (video_id, clip_start, graph_dir)
        print(f"correct graph_dir: {graph_dir}")
    
    # TODO: video and clip start should not be called anything related to graph, they come from a different place
    def _find_available_graph(self, video_dir) -> List[tuple]:
        """Find all preprocessed clips with graphs"""        
        video_id = int(video_dir.name.replace("video", ""))
            
        for clip_dir in video_dir.glob("video*_*"):
            clip_name = clip_dir.name
            print(f"clip_name: {clip_name}")
            clip_start = int(clip_name.split("_")[1])
            
            # Check if essential graph files exist
            if (clip_dir / "adjacency_matrices.npy").exists():
                return video_id, clip_start, clip_dir
            else:
                raise FileNotFoundError(f"No graph found for video {video_id} at clip start {clip_start}")
            
    # TODO: investigate the args here, some of them make no sense anymore
    def select_sequences(
        self, 
        num_sequences: int = 5,
        frames_per_sequence: int = 5,
        min_config_length: int = 20
    ) -> List[MultiFrameSample]:
        """
        Select multi-frame sequences for evaluation.
        
        Args:
            num_sequences: Number of sequences to select
            frames_per_sequence: Number of frames in each sequence
            min_config_length: Minimum length of consistent triplet configuration
        """

        video_id, clip_start, graph_path = self.available_graph
        print(f"Found an available graph for video {video_id} at clip start {clip_start} with path {graph_path}")

        # TODO: to do this properly later, we will need duration of the video and framerate
        
        video_data = self.loader.load_video_annotations(video_id)
        print(f"Loaded video data for video {video_id}")
            
        # Find image paths
        video_dir = self.config.preprocessed_root / f"video{video_id:02d}"
        clip_dirs = list(video_dir.glob(f"video{video_id:02d}_{clip_start:05d}"))
        clip_dir = clip_dirs[0]
            
            
        # Get image paths for selected frames
        image_paths = []
        # image paths are all images in the clip directory
        image_paths = list(sorted(Path(os.path.join(clip_dir, "images")).glob("*.jpg")))
        print(f"first 10 image_paths: {image_paths[:10]}")
            
        # Get ground truth triplets
        # Currently, the frame ids in the cholecseg8k dataset are relative to the clip start -> have to add and then divide by the framerate
        # TODO: get this later from some config, hardcoded for now
        FRAMERATE = 25
        NUM_FRAMES = 80
        TEMP_CONTEXT_FR_MULTIPLIER = 1

        samples = []

        for i in range(clip_start, clip_start + NUM_FRAMES):
            # TODO: hardcoded hack, fix later
            if i % FRAMERATE == 0:
                frame_triplet = i // FRAMERATE
                triplets = self.loader.get_frame_triplets(video_data, frame_triplet)
                print(f"triplets for frame {i}/{frame_triplet}: {triplets}")

                # TODO: truncate images, graph, etc.
                # TODO: fix issue in table for verb
                # TODO. start and end frame do not seem to matter?
                # TODO: currently we hack start_frame = end_frame = index into current dataset relative to clip start
                end_frame = (i - clip_start)
                start_frame = end_frame - TEMP_CONTEXT_FR_MULTIPLIER * FRAMERATE
                if start_frame < 0:
                    start_frame = 0
                samples.append(MultiFrameSample(
                    video_id=video_id,
                    # TODO: this logic won't always work! fix
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

