#!/usr/bin/env python3
"""
Command-line testing script for ARC algorithm
Usage: python cli_test.py --env env1 --model models/arc_best.pth
"""

import argparse
import sys
import os
import runpy
from test import ARCTester, test_all_environments as test_all_func
from config import ENVIRONMENTS, MODEL_CONFIG, VIS_CONFIG

def main():
    parser = argparse.ArgumentParser(
        description='Test trained ARC agent',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run standalone APF dynamic Run 1 (GIF + Excel)
    python cli_test.py --apf-run1

  # Test on single environment
  python cli_test.py --env env1
  
  # Test on all environments
  python cli_test.py --all
  
  # Test with specific model
  python cli_test.py --env env1 --model models/arc_checkpoint_ep200.pth
  
  # Test without creating animation (faster)
  python cli_test.py --env env1 --no-animation
  
  # Evaluate with 100 runs
  python cli_test.py --env env7 --evaluate --runs 100
        """
    )
    
    parser.add_argument('--env', type=str, default=None,
                       choices=list(ENVIRONMENTS.keys()),
                       help='Environment name to test')
    parser.add_argument('--all', action='store_true',
                       help='Test on all main environments (env1-env6)')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to model file (default: best model)')
    parser.add_argument('--no-animation', action='store_true',
                       help='Skip animation creation (faster)')
    parser.add_argument('--evaluate', action='store_true',
                       help='Run multiple tests for evaluation')
    parser.add_argument('--runs', type=int, default=100,
                       help='Number of evaluation runs (default: 100)')
    parser.add_argument('--apf-run1', action='store_true',
                       help='Run standalone APF Dynamic Run 1 pipeline (outputs to results/)')
    
    args = parser.parse_args()

    if args.apf_run1:
        if args.env or args.all or args.model or args.no_animation or args.evaluate:
            parser.error("--apf-run1 cannot be combined with ARC testing flags")
        script_path = os.path.join(os.path.dirname(__file__), 'apf_dynamic_run1.py')
        if not os.path.exists(script_path):
            print(f"\nError: APF script not found at {script_path}")
            return 1
        print("\n" + "="*70)
        print(" APF Dynamic Run 1")
        print("="*70 + "\n")
        runpy.run_path(script_path, run_name='__main__')
        return 0
    
    # Validate arguments
    if not args.all and not args.env:
        parser.error("Either --env or --all must be specified")
    
    if args.all and args.env:
        parser.error("Cannot specify both --env and --all")
    
    # Set model path
    if args.model is None:
        args.model = os.path.join(MODEL_CONFIG['save_dir'], MODEL_CONFIG['best_model_name'])
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"\nError: Model not found at {args.model}")
        print("\nAvailable models:")
        model_dir = MODEL_CONFIG['save_dir']
        if os.path.exists(model_dir):
            models = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
            if models:
                for m in models:
                    print(f"  - {os.path.join(model_dir, m)}")
            else:
                print("  No models found. Train a model first using cli_train.py")
        else:
            print(f"  Model directory does not exist: {model_dir}")
        return 1
    
    print("\n" + "="*70)
    print(" ARC Testing")
    print("="*70)
    print(f"\nModel: {args.model}")
    
    # Test all environments
    if args.all:
        print("Testing on all environments (env1-env6)...")
        print("="*70 + "\n")
        test_all_func(MODEL_CONFIG['save_dir'], os.path.basename(args.model))
        return 0
    
    # Test single environment
    print(f"Environment: {args.env} - {ENVIRONMENTS[args.env]['name']}")
    print("="*70 + "\n")
    
    # Create tester
    tester = ARCTester(args.model, args.env)
    
    # Evaluation mode
    if args.evaluate:
        print(f"Running {args.runs} evaluation episodes...")
        results = tester.test_multiple_runs(num_runs=args.runs)
        
        print("\n" + "="*70)
        print(" Evaluation Results")
        print("="*70)
        print(f"\nSuccess Rate: {results['success_rate']:.2%}")
        print(f"Collision Rate: {results['collision_rate']:.2%}")
        print(f"Average Reward: {results['avg_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"Average Steps: {results['avg_steps']:.1f} ± {results['std_steps']:.1f}")
        print("\n" + "="*70 + "\n")
        return 0
    
    # Single test mode
    print("Running test episode...")
    trajectory, stats = tester.test_episode()
    
    print("\n" + "="*70)
    print(" Test Results")
    print("="*70)
    print(f"\nSteps: {stats['steps']}")
    print(f"Total Reward: {stats['total_reward']:.2f}")
    print(f"Success: {stats['success']}")
    print(f"Collision: {stats['collision']}")
    print(f"Final Distance to Target: {stats['distance_to_target']:.2f}m")
    
    # Create visualizations
    vis_dir = VIS_CONFIG['save_dir']
    os.makedirs(vis_dir, exist_ok=True)
    
    print(f"\nCreating visualizations...")
    
    # Save trajectory plot
    img_path = os.path.join(vis_dir, f'{args.env}_trajectory.png')
    tester.visualize_trajectory(trajectory, save_path=img_path)
    
    # Create animation if requested
    if not args.no_animation:
        anim_path = os.path.join(vis_dir, f'{args.env}_animation.gif')
        tester.create_animation(trajectory, save_path=anim_path)
    
    print("\n" + "="*70)
    print(" Output Files")
    print("="*70)
    print(f"\nVisualization directory: {vis_dir}/")
    print(f"  - Trajectory plot: {args.env}_trajectory.png")
    if not args.no_animation:
        print(f"  - Animation: {args.env}_animation.gif")
    print("\n" + "="*70 + "\n")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())