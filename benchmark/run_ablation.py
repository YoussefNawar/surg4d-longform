#!/usr/bin/env python3
# TODO: udpate this
"""
Main script to run ablation study for multi-frame triplet recognition.

Three conditions:
1. Single Frame: Baseline using middle frame
2. Multi-Frame: Temporal reasoning with multiple frames
3. Multi-Frame + Graph: Spatiotemporal reasoning with scene graph

Usage:
    python run_ablation.py --num_sequences 5 --frames_per_sequence 5
    python run_ablation.py --conditions single_frame multiframe
"""

import argparse
from pathlib import Path
from datetime import datetime

from benchmark_config import BenchmarkConfig
from frame_selectors import TripletsFrameSelector
from frame_evaluators import TripletsFrameEvaluator

import yaml


def evaluate_triplets(config):
    # Select samples
    print("Selecting samples...")
    selector = TripletsFrameSelector(config)
    samples = selector.select_sequences()
    if not samples:
        print("ERROR: No samples selected! Check if preprocessed data is available.")
        return 1
    selector.print_summary(samples)
    # Run ablation study
    print("\nInitializing evaluator...")
    evaluator = TripletsFrameEvaluator(config)
    results = evaluator.run_ablation_study(samples, ablations=config.triplets_config['ablations'])
    return evaluator, results


def save_triplet_results(evaluator, results, output_path, results_dir):
    # Print comparison
    print("\n" + "="*80)
    print("ABLATION STUDY RESULTS")
    print("="*80)
    print()
    print(f"{'Condition':<25} {'Instrument':<12} {'Verb':<12} {'Target':<12} {'Full Triplet':<12}")
    print("-"*80)
    for condition, data in results['conditions'].items():
        metrics = data['metrics']
        print(f"{condition:<25} "
              f"{metrics['instrument_acc']:>10.1%}  "
              f"{metrics['verb_acc']:>10.1%}  "
              f"{metrics['target_acc']:>10.1%}  "
              f"{metrics['triplet_acc']:>10.1%}")
    print("="*80)
    # Compute improvements
    if 'single_frame' in results['conditions'] and 'multiframe' in results['conditions']:
        single = results['conditions']['single_frame']['metrics']['triplet_acc']
        multi = results['conditions']['multiframe']['metrics']['triplet_acc']
        improvement = (multi - single) / single * 100 if single > 0 else 0
        print(f"\nTemporal improvement: {improvement:+.1f}%")
    if 'multiframe' in results['conditions'] and 'multiframe_graph' in results['conditions']:
        multi = results['conditions']['multiframe']['metrics']['triplet_acc']
        graph = results['conditions']['multiframe_graph']['metrics']['triplet_acc']
        improvement = (graph - multi) / multi * 100 if multi > 0 else 0
        print(f"Graph improvement: {improvement:+.1f}%")
    # Save results
    if output_path:
        output_path = Path(output_path)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = results_dir / f"ablation_{timestamp}.json"
    evaluator.save_results(results, output_path)
    print(f"\n✓ Ablation study complete!")
    print(f"✓ Results saved to: {output_path}")


def evaluate_temporal(config):
    return None

def evaluate_spatial(config):
    return None

def evaluate_spatiotemporal(config):
    return None



def main():
    parser = argparse.ArgumentParser(
        description='Run ablation study for several tasks.'
    )
    parser.add_argument(
        '--triplets_config',
        default=None,
        help='Config for triplets (default: None)'
    )
    parser.add_argument(
        '--temporal_config',
        default=None,
        help='Config for temporal (default: None)'
    )
    parser.add_argument(
        '--spatial_config',
        default=None,
        help='Config for spatial (default: None)'
    )
    parser.add_argument(
        '--spatiotemporal_config',
        default=None,
        help='Config for spatiotemporal (default: None)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path (default: auto-generated in results/)'
    )
    parser.add_argument(
        '--use_4bit',
        action='store_true',
        help='Use 4-bit quantization for model'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    # Load configs with yaml for each task if provided
    triplets_config = None
    temporal_config = None
    spatial_config = None
    spatiotemporal_config = None
    if args.triplets_config:
        with open(args.triplets_config, 'r') as f:
            triplets_config = yaml.load(f, Loader=yaml.FullLoader)
            print(f"triplets_config: {triplets_config}")
    if args.temporal_config:
        with open(args.temporal_config, 'r') as f:
            temporal_config = yaml.load(f, Loader=yaml.FullLoader)
    if args.spatial_config:
        with open(args.spatial_config, 'r') as f:
            spatial_config = yaml.load(f, Loader=yaml.FullLoader)
    if args.spatiotemporal_config:
        with open(args.spatiotemporal_config, 'r') as f:
            spatiotemporal_config = yaml.load(f, Loader=yaml.FullLoader)

    # Configuration
    config = BenchmarkConfig(
        triplets_config=triplets_config,
        temporal_config=temporal_config,
        spatial_config=spatial_config,
        spatiotemporal_config=spatiotemporal_config,
        model_name="qwen",
        # TODO: maybe make conditional but probably fixing to 3 once we have merged current remote and everything works
        qwen_version="qwen2.5",
        use_4bit_quantization=args.use_4bit
    )

    print(f"Evaluating triplets: {'✅' if config.triplets_config is not None else '❌'}")
    print(f"Evaluating temporal: {'✅' if config.temporal_config is not None else '❌'}")
    print(f"Evaluating spatial: {'✅' if config.spatial_config is not None else '❌'}")
    print(f"Evaluating spatiotemporal: {'✅' if config.spatiotemporal_config is not None else '❌'}")
    
    # Evaluate each task individually
    # TODO: do we want a base config and a task specific config?
    if triplets_config is not None:
        evaluator, results = evaluate_triplets(config)
        save_triplet_results(evaluator, results, args.output, config.results_dir)
    if temporal_config is not None:
        evaluate_temporal(config)
    if spatial_config is not None:
        evaluate_spatial(config)
    if spatiotemporal_config is not None:
        evaluate_spatiotemporal(config)
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())

