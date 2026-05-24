"""
Main execution script for ARC UAV Path Planning
Run commands line by line from terminal
"""

import os
import runpy
import torch
from config import MODEL_CONFIG, ENVIRONMENTS


def print_menu():
    """Print main menu"""
    print("\n" + "="*70)
    print(" ARC UAV Path Planning - Main Menu")
    print("="*70)
    print("\n Available Commands:")
    print("  1. train_arc(env_name, num_episodes)")
    print("  2. test_arc(model_path, env_name)")
    print("  3. test_all_environments(model_path)")
    print("  4. evaluate_performance(env_name, num_runs)")
    print("  5. visualize_training(trainer)")
    print("  6. run_apf_dynamic_run1()")
    print("\n Available Environments:")
    for env_name, env_def in ENVIRONMENTS.items():
        print(f"  - {env_name}: {env_def['name']}")
    print("\n" + "="*70)


def train_arc(env_name='env7', num_episodes=300):
    """
    Train ARC agent
    
    Example usage:
        from main import train_arc
        trainer = train_arc('env7', 300)
    
    Args:
        env_name: Environment name from config
        num_episodes: Number of training episodes
        
    Returns:
        trainer: ARCTrainer instance
    """
    from train import ARCTrainer
    
    print(f"\n{'='*70}")
    print(f" Training ARC on {env_name}")
    print(f"{'='*70}")
    
    trainer = ARCTrainer(env_name=env_name)
    trainer.train(num_episodes=num_episodes)
    
    print(f"\nTraining completed!")
    print(f"Model saved in: {MODEL_CONFIG['save_dir']}")
    print(f"TensorBoard logs in: runs/ARC_{env_name}")
    print(f"\nTo view training logs, run:")
    print(f"  tensorboard --logdir=runs/ARC_{env_name}")
    
    return trainer


def test_arc(model_path=None, env_name='env1', create_animation=True):
    """
    Test trained ARC agent on a single environment
    
    Example usage:
        from main import test_arc
        trajectory, stats = test_arc('models/arc_best.pth', 'env1')
    
    Args:
        model_path: Path to trained model (uses best model if None)
        env_name: Environment name
        create_animation: Whether to create animation
        
    Returns:
        trajectory, stats: Test results
    """
    from test import ARCTester
    from config import VIS_CONFIG
    
    if model_path is None:
        model_path = os.path.join(MODEL_CONFIG['save_dir'], MODEL_CONFIG['best_model_name'])
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return None, None
    
    print(f"\n{'='*70}")
    print(f" Testing ARC on {env_name}")
    print(f"{'='*70}")
    
    tester = ARCTester(model_path, env_name)
    trajectory, stats = tester.test_episode()
    
    print(f"\nTest Results:")
    print(f"  Steps: {stats['steps']}")
    print(f"  Total Reward: {stats['total_reward']:.2f}")
    print(f"  Success: {stats['success']}")
    print(f"  Collision: {stats['collision']}")
    print(f"  Final Distance to Target: {stats['distance_to_target']:.2f}m")
    
    # Create visualization directory
    vis_dir = VIS_CONFIG['save_dir']
    os.makedirs(vis_dir, exist_ok=True)
    
    # Save trajectory visualization
    img_path = os.path.join(vis_dir, f'{env_name}_trajectory.png')
    tester.visualize_trajectory(trajectory, save_path=img_path)
    
    # Create animation if requested
    if create_animation:
        anim_path = os.path.join(vis_dir, f'{env_name}_animation.gif')
        tester.create_animation(trajectory, save_path=anim_path)
    
    print(f"\nVisualizations saved in: {vis_dir}")
    
    return trajectory, stats


def test_all_environments(model_path=None):
    """
    Test model on all 6 main environments (env1-env6)
    
    Example usage:
        from main import test_all_environments
        test_all_environments('models/arc_best.pth')
    
    Args:
        model_path: Path to trained model (uses best model if None)
    """
    from test import test_all_environments as test_all
    
    if model_path is None:
        model_path = os.path.join(MODEL_CONFIG['save_dir'], MODEL_CONFIG['best_model_name'])
    
    test_all(MODEL_CONFIG['save_dir'], os.path.basename(model_path))


def evaluate_performance(env_name='env7', num_runs=100, model_path=None):
    """
    Evaluate performance with multiple test runs
    
    Example usage:
        from main import evaluate_performance
        results = evaluate_performance('env7', 100)
    
    Args:
        env_name: Environment name
        num_runs: Number of test runs
        model_path: Path to model (uses best if None)
        
    Returns:
        results: Dictionary with performance statistics
    """
    from test import ARCTester
    
    if model_path is None:
        model_path = os.path.join(MODEL_CONFIG['save_dir'], MODEL_CONFIG['best_model_name'])
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return None
    
    print(f"\n{'='*70}")
    print(f" Evaluating Performance on {env_name}")
    print(f"{'='*70}")
    
    tester = ARCTester(model_path, env_name)
    results = tester.test_multiple_runs(num_runs)
    
    return results


def visualize_training(trainer, test_env7=True, test_env8=True):
    """
    Create training visualization plots
    
    Example usage:
        from main import train_arc, visualize_training
        trainer = train_arc('env7', 300)
        visualize_training(trainer)
    
    Args:
        trainer: ARCTrainer instance
        test_env7: Whether to test on env7
        test_env8: Whether to test on env8
    """
    from visualize_training import visualize_training_results
    from test import ARCTester
    
    test_results_env7 = None
    test_results_env8 = None
    
    model_path = os.path.join(MODEL_CONFIG['save_dir'], MODEL_CONFIG['best_model_name'])
    
    if test_env7 and os.path.exists(model_path):
        print("\nEvaluating on Environment 7...")
        tester7 = ARCTester(model_path, 'env7')
        test_results_env7 = tester7.test_multiple_runs(100)
    
    if test_env8 and os.path.exists(model_path):
        print("\nEvaluating on Environment 8...")
        tester8 = ARCTester(model_path, 'env8')
        test_results_env8 = tester8.test_multiple_runs(100)
    
    visualize_training_results(trainer, test_results_env7, test_results_env8)
    print("\nTraining visualizations created!")


def run_apf_dynamic_run1():
    """Run the standalone APF dynamic run-1 script (GIF + Excel outputs)."""
    script_path = os.path.join(os.path.dirname(__file__), 'apf_dynamic_run1.py')
    if not os.path.exists(script_path):
        print(f"Error: APF script not found at {script_path}")
        return
    print(f"\n{'='*70}")
    print(" Running APF Dynamic Run 1")
    print(f"{'='*70}")
    runpy.run_path(script_path, run_name='__main__')


def complete_workflow():
    """
    Complete workflow: Train, test, and visualize
    
    Example usage:
        from main import complete_workflow
        complete_workflow()
    """
    print("\n" + "="*70)
    print(" ARC Complete Workflow")
    print("="*70)
    
    # Step 1: Train on env7
    print("\nStep 1/4: Training on Environment 7 (Sparse Obstacles)...")
    trainer = train_arc('env7', num_episodes=300)
    
    # Step 2: Create training visualizations
    print("\nStep 2/4: Creating training visualizations...")
    visualize_training(trainer)
    
    # Step 3: Test on all environments
    print("\nStep 3/4: Testing on all environments...")
    test_all_environments()
    
    # Step 4: Performance evaluation
    print("\nStep 4/4: Performance evaluation...")
    results_env7 = evaluate_performance('env7', 100)
    results_env8 = evaluate_performance('env8', 100)
    
    print("\n" + "="*70)
    print(" Workflow Complete!")
    print("="*70)
    print("\nResults Summary:")
    print(f"  Environment 7 Success Rate: {results_env7['success_rate']:.2%}")
    print(f"  Environment 8 Success Rate: {results_env8['success_rate']:.2%}")
    print(f"\nAll outputs saved in:")
    print(f"  Models: {MODEL_CONFIG['save_dir']}/")
    print(f"  Visualizations: {VIS_CONFIG['save_dir']}/")
    print(f"  TensorBoard logs: runs/")


# Example commands to run line by line in terminal
if __name__ == "__main__":
    print_menu()
    
    print("\n" + "="*70)
    print(" Quick Start Guide")
    print("="*70)
    print("\n1. Import this module:")
    print("   >>> from main import *")
    print("\n2. Train the model:")
    print("   >>> trainer = train_arc('env7', 300)")
    print("\n3. Test on a single environment:")
    print("   >>> trajectory, stats = test_arc(env_name='env1')")
    print("\n4. Test on all environments:")
    print("   >>> test_all_environments()")
    print("\n5. Create training visualizations:")
    print("   >>> visualize_training(trainer)")
    print("\n6. Evaluate performance:")
    print("   >>> results = evaluate_performance('env7', 100)")
    print("\n7. Run APF dynamic Run 1:")
    print("   >>> run_apf_dynamic_run1()")
    print("\n8. Or run complete workflow:")
    print("   >>> complete_workflow()")
    print("\n" + "="*70)