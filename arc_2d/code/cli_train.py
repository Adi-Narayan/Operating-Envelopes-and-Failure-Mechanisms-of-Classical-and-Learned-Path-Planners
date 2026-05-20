"""
Command-line training script for ARC algorithm
Usage: python cli_train.py --env env7 --episodes 300
"""

import argparse
import sys
from train import ARCTrainer
from config import ENVIRONMENTS, MODEL_CONFIG
import time

def main():
    parser = argparse.ArgumentParser(
        description='Train ARC agent for UAV path planning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli_train.py --env env7 --episodes 300
  python cli_train.py --env env8 --episodes 500
  
Available environments:
  env1: Barrier-free with stationary target
  env2: Simple obstacle with stationary target
  env3: Complex obstacles with stationary target
  env4: Barrier-free with moving target
  env5: Simple obstacle with moving target
  env6: Complex obstacles with moving target
  env7: Sparse obstacles (for testing)
  env8: Dense obstacles (for testing)
        """
    )
    
    parser.add_argument('--env', type=str, default='env7',
                       choices=list(ENVIRONMENTS.keys()),
                       help='Environment name (default: env7)')
    parser.add_argument('--episodes', type=int, default=300,
                       help='Number of training episodes (default: 300)')
    parser.add_argument('--device', type=str, default=None,
                       choices=['cpu', 'cuda'],
                       help='Device to use (default: auto-detect)')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print(f" ARC Training")
    print("="*70)
    print(f"\nEnvironment: {args.env} - {ENVIRONMENTS[args.env]['name']}")
    print(f"Episodes: {args.episodes}")
    print(f"Device: {args.device if args.device else 'auto-detect'}")
    print("\n" + "="*70 + "\n")
    
    # Create trainer
    import torch
    device = None
    if args.device:
        device = torch.device(args.device)
    
    trainer = ARCTrainer(env_name=args.env, device=device)
    
    # Train
    trainer.train(num_episodes=args.episodes)
    
    print("\n" + "="*70)
    print(" Training Complete!")
    print("="*70)
    print(f"\nModel saved in: {MODEL_CONFIG['save_dir']}/")
    print(f"  - Best model: {MODEL_CONFIG['best_model_name']}")
    print(f"  - Checkpoints: {MODEL_CONFIG['checkpoint_prefix']}_ep*.pth")
    print(f"\nTensorBoard logs: runs/ARC_{args.env}/")
    print("\nTo view training progress:")
    print(f"  tensorboard --logdir=runs/ARC_{args.env}")
    print("\nTo test the trained model:")
    print(f"  python cli_test.py --env {args.env}")
    print("\n" + "="*70 + "\n")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())