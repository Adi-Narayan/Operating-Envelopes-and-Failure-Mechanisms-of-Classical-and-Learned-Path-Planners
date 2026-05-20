"""
Command-line visualization script for training results
Usage: python cli_visualize.py --logdir runs/ARC_env7
"""

import argparse
import sys
import os
import pickle
from visualize_training import plot_training_results, plot_comparison_results
from config import VIS_CONFIG, MODEL_CONFIG
from test import ARCTester

def load_trainer_from_tensorboard(log_dir):
    """Load training data from TensorBoard logs"""
    from tensorboard.backend.event_processing import event_accumulator
    
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    
    # Extract data
    data = {
        'episode_rewards': [],
        'episode_steps': [],
        'success_rate': [],
        'collision_rate': [],
    }
    
    tags = {
        'Training/Episode_Reward': 'episode_rewards',
        'Training/Episode_Steps': 'episode_steps',
        'Training/Success_Rate': 'success_rate',
        'Training/Collision_Rate': 'collision_rate',
    }
    
    for tb_tag, data_key in tags.items():
        if tb_tag in ea.Tags()['scalars']:
            events = ea.Scalars(tb_tag)
            data[data_key] = [e.value for e in events]
    
    return data

def main():
    parser = argparse.ArgumentParser(
        description='Visualize ARC training results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize from TensorBoard logs
  python cli_visualize.py --logdir runs/ARC_env7
  
  # Create comparison between env7 and env8
  python cli_visualize.py --compare --model models/arc_best.pth
  
  # Just plot training curves (no evaluation)
  python cli_visualize.py --logdir runs/ARC_env7 --no-eval
        """
    )
    
    parser.add_argument('--logdir', type=str, default=None,
                       help='TensorBoard log directory')
    parser.add_argument('--compare', action='store_true',
                       help='Create comparison plot between env7 and env8')
    parser.add_argument('--model', type=str, default=None,
                       help='Model path for evaluation (default: best model)')
    parser.add_argument('--no-eval', action='store_true',
                       help='Skip evaluation on test environments')
    parser.add_argument('--runs', type=int, default=100,
                       help='Number of evaluation runs (default: 100)')
    
    args = parser.parse_args()
    
    if not args.logdir and not args.compare:
        parser.error("Either --logdir or --compare must be specified")
    
    print("\n" + "="*70)
    print(" ARC Visualization")
    print("="*70 + "\n")
    
    vis_dir = VIS_CONFIG['save_dir']
    os.makedirs(vis_dir, exist_ok=True)
    
    # Set model path
    if args.model is None:
        args.model = os.path.join(MODEL_CONFIG['save_dir'], MODEL_CONFIG['best_model_name'])
    
    # Plot training results
    if args.logdir:
        if not os.path.exists(args.logdir):
            print(f"Error: Log directory not found: {args.logdir}")
            return 1
        
        print(f"Loading training data from: {args.logdir}")
        
        try:
            training_data = load_trainer_from_tensorboard(args.logdir)
            
            if not training_data['episode_rewards']:
                print("Error: No training data found in log directory")
                return 1
            
            print(f"Found {len(training_data['episode_rewards'])} episodes")
            
            # Create training plot
            output_path = os.path.join(vis_dir, 'training_results.png')
            plot_training_results(training_data, save_path=output_path)
            
            print(f"\nCreated: {output_path}")
            
        except Exception as e:
            print(f"Error loading TensorBoard logs: {e}")
            print("\nNote: You may need to install tensorboard:")
            print("  pip install tensorboard")
            return 1
    
    # Create comparison plot
    if args.compare:
        if not os.path.exists(args.model):
            print(f"Error: Model not found: {args.model}")
            return 1
        
        if args.no_eval:
            print("Note: --no-eval ignored when using --compare")
        
        print(f"\nEvaluating model on env7 and env8...")
        print(f"Model: {args.model}")
        print(f"Runs per environment: {args.runs}\n")
        
        # Test on env7
        print("Testing on Environment 7 (sparse obstacles)...")
        tester7 = ARCTester(args.model, 'env7')
        results_env7 = tester7.test_multiple_runs(args.runs)
        
        # Test on env8
        print("\nTesting on Environment 8 (dense obstacles)...")
        tester8 = ARCTester(args.model, 'env8')
        results_env8 = tester8.test_multiple_runs(args.runs)
        
        # Create comparison plot
        output_path = os.path.join(vis_dir, 'comparison_results.png')
        plot_comparison_results(results_env7, results_env8, save_path=output_path)
        
        print(f"\nCreated: {output_path}")
        
        # Print comparison
        print("\n" + "="*70)
        print(" Performance Comparison")
        print("="*70)
        print(f"\n{'Metric':<20} {'Env7 (Sparse)':<20} {'Env8 (Dense)':<20}")
        print("-" * 70)
        print(f"{'Success Rate':<20} {results_env7['success_rate']:>18.2%} {results_env8['success_rate']:>18.2%}")
        print(f"{'Collision Rate':<20} {results_env7['collision_rate']:>18.2%} {results_env8['collision_rate']:>18.2%}")
        print(f"{'Avg Reward':<20} {results_env7['avg_reward']:>18.2f} {results_env8['avg_reward']:>18.2f}")
        print(f"{'Avg Steps':<20} {results_env7['avg_steps']:>18.1f} {results_env8['avg_steps']:>18.1f}")
        print("")
    
    print("="*70)
    print(f"\nAll visualizations saved in: {vis_dir}/")
    print("="*70 + "\n")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())