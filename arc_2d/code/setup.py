#!/usr/bin/env python3
"""
Setup helper for ARC UAV Path Planning
Checks environment and creates necessary directories
"""

import os
import sys
import subprocess

def print_header(text):
    print("\n" + "="*70)
    print(f" {text}")
    print("="*70 + "\n")

def check_python_version():
    """Check Python version"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print(f"❌ Python {version.major}.{version.minor} detected")
        print("   Required: Python 3.7 or higher")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_dependencies():
    """Check if required packages are installed"""
    print("\nChecking dependencies...")
    
    required = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
        'shapely': 'Shapely',
        'tqdm': 'tqdm',
        'tensorboard': 'TensorBoard'
    }
    
    missing = []
    
    for module, name in required.items():
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"❌ {name}")
            missing.append(module)
    
    if missing:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing)}")
        print("\nTo install:")
        print("  pip install -r requirements.txt")
        return False
    
    return True

def create_directories():
    """Create necessary directories"""
    print("\nCreating directories...")
    
    dirs = ['models', 'runs', 'visualization']
    
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"✓ Created {d}/")
        else:
            print(f"✓ {d}/ exists")
    
    return True

def make_executable():
    """Make shell scripts executable (Linux/Mac)"""
    if os.name != 'posix':
        return True
    
    print("\nMaking scripts executable...")
    
    scripts = ['run_complete_workflow.sh']
    
    for script in scripts:
        if os.path.exists(script):
            try:
                os.chmod(script, 0o755)
                print(f"✓ {script}")
            except Exception as e:
                print(f"❌ {script}: {e}")
                return False
        else:
            print(f"⚠️  {script} not found")
    
    return True

def test_gpu():
    """Test GPU availability"""
    print("\nChecking GPU availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
            return True
        else:
            print("ℹ️  No GPU detected - will use CPU")
            print("   Training will be slower but still functional")
            return True
    except Exception as e:
        print(f"⚠️  Could not check GPU: {e}")
        return True

def print_next_steps():
    """Print next steps"""
    print_header("Setup Complete!")
    
    print("You can now:")
    print("")
    print("1. Run complete workflow:")
    if os.name == 'posix':
        print("   ./run_complete_workflow.sh")
    else:
        print("   run_complete_workflow.bat")
    print("")
    print("2. Train manually:")
    print("   python cli_train.py --env env7 --episodes 300")
    print("")
    print("3. Test:")
    print("   python cli_test.py --all")
    print("")
    print("4. Interactive mode:")
    print("   python")
    print("   >>> from main import *")
    print("   >>> trainer = train_arc('env7', 300)")
    print("")
    print("For quick reference:")
    print("   See QUICKSTART.md")
    print("")
    print("For detailed documentation:")
    print("   See README.md")
    print("")

def main():
    """Main setup function"""
    print_header("ARC UAV Path Planning - Setup")
    
    all_good = True
    
    # Check Python version
    if not check_python_version():
        all_good = False
    
    # Check dependencies
    if not check_dependencies():
        all_good = False
    
    # Create directories
    if not create_directories():
        all_good = False
    
    # Make scripts executable
    if not make_executable():
        all_good = False
    
    # Test GPU
    test_gpu()
    
    # Final message
    print("")
    if all_good:
        print_next_steps()
        return 0
    else:
        print_header("Setup Incomplete")
        print("⚠️  Please resolve the issues above and run setup again:")
        print("   python setup.py")
        print("")
        return 1

if __name__ == '__main__':
    sys.exit(main())