"""
Quick test to verify the environment and imports work correctly.
Run this before starting the full training pipeline.
"""
import sys

def test_imports():
    """Test that all required packages are installed."""
    print("Testing imports...")
    
    errors = []
    
    # Test NumPy
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError as e:
        errors.append(f"✗ NumPy: {e}")
    
    # Test Matplotlib
    try:
        import matplotlib
        print(f"✓ Matplotlib {matplotlib.__version__}")
    except ImportError as e:
        errors.append(f"✗ Matplotlib: {e}")
    
    # Test PyTorch
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            print(f"  → CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print(f"  → CUDA not available (will use CPU)")
    except ImportError as e:
        errors.append(f"✗ PyTorch: {e}")
    
    # Test Gymnasium
    try:
        import gymnasium as gym
        print(f"✓ Gymnasium {gym.__version__}")
    except ImportError as e:
        errors.append(f"✗ Gymnasium: {e}")
    
    # Test Stable-Baselines3
    try:
        import stable_baselines3 as sb3
        print(f"✓ Stable-Baselines3 {sb3.__version__}")
    except ImportError as e:
        errors.append(f"✗ Stable-Baselines3: {e}")
    
    # Test tqdm
    try:
        import tqdm
        print(f"✓ tqdm {tqdm.__version__}")
    except ImportError as e:
        errors.append(f"✗ tqdm: {e}")
    
    print()
    
    if errors:
        print("ERRORS FOUND:")
        for error in errors:
            print(f"  {error}")
        print("\nPlease run: pip install -r requirements.txt")
        return False
    else:
        print("✓ All imports successful!")
        return True


def test_environment():
    """Test that the racing environment can be created."""
    print("\nTesting racing environment...")
    
    try:
        from racing_env import RacingEnv
        
        env = RacingEnv(track_width=4.0, max_steps=100)
        obs, info = env.reset()
        
        print(f"✓ Environment created successfully")
        print(f"  → Observation shape: {obs.shape}")
        print(f"  → Action space: {env.action_space}")
        print(f"  → Track length: {env.track_length:.1f}")
        
        # Test a few steps
        for i in range(3):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"✓ Environment stepping works")
        env.close()
        
        return True
        
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mariokart():
    """Test that mariokart.py functions are accessible."""
    print("\nTesting mariokart.py functions...")
    
    try:
        from mariokart import (
            make_centerline_from_custom_track,
            bicycle_dynamics,
            time_optimal_controller
        )
        
        # Test track generation
        track_x, track_y = make_centerline_from_custom_track()
        print(f"✓ Track generation works ({len(track_x)} points)")
        
        # Test dynamics
        import numpy as np
        state = np.array([0, 0, 0, 1, 0, 0])
        params = {"m": 150.0, "Iz": 20.0, "lf": 0.7, "lr": 0.7, "Cf": 800.0, "Cr": 800.0}
        state_dot = bicycle_dynamics(state, 0.0, 0.0, params)
        print(f"✓ Vehicle dynamics work")
        
        return True
        
    except Exception as e:
        print(f"✗ mariokart.py test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("RL Racing Controller - System Test")
    print("="*60)
    print()
    
    tests_passed = 0
    tests_total = 3
    
    # Test imports
    if test_imports():
        tests_passed += 1
    
    # Test mariokart.py
    if test_mariokart():
        tests_passed += 1
    
    # Test environment
    if test_environment():
        tests_passed += 1
    
    # Summary
    print()
    print("="*60)
    print(f"Tests passed: {tests_passed}/{tests_total}")
    
    if tests_passed == tests_total:
        print("✓ All systems ready!")
        print()
        print("You can now start the training pipeline:")
        print("  1. python collect_demonstrations.py")
        print("  2. python train_bc.py")
        print("  3. python train_rl.py")
        print()
        print("Or use the quick start script:")
        print("  ./quickstart.sh")
    else:
        print("✗ Some tests failed. Please fix the issues above.")
        print()
        print("If packages are missing, run:")
        print("  source rl_env/bin/activate")
        print("  pip install -r requirements.txt")
    
    print("="*60)
    
    return tests_passed == tests_total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
