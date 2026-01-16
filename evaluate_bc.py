"""
Evaluate behavior cloning policy against baseline controller.
"""
import numpy as np
import torch
from train_bc import PolicyNetwork
from racing_env import RacingEnv
from mariokart import time_optimal_controller
import argparse
from pathlib import Path


def evaluate_bc_policy(model, env, n_episodes=10):
    """
    Evaluate BC policy on the environment.
    
    Returns:
        Dictionary with episode statistics
    """
    episode_rewards = []
    episode_lengths = []
    lap_times = []
    success_count = 0
    lateral_errors = []
    
    model.eval()
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        step_count = 0
        episode_lateral_errors = []
        
        while not done and step_count < env.max_steps:
            # Get action from BC policy
            with torch.no_grad():
                action = model(torch.FloatTensor(obs)).numpy()
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            step_count += 1
            
            # Track lateral error
            if hasattr(env, 'lateral_error'):
                episode_lateral_errors.append(abs(env.lateral_error))
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(step_count)
        
        if episode_lateral_errors:
            lateral_errors.append(np.mean(episode_lateral_errors))
        
        # Check if completed track successfully
        if hasattr(env, 'track_position'):
            if env.track_position >= len(env.track_x) - 2:
                success_count += 1
                lap_time = step_count * env.dt
                lap_times.append(lap_time)
        
        print(f"Episode {episode + 1}/{n_episodes}: "
              f"Reward={episode_reward:.2f}, Steps={step_count}, "
              f"Avg Lateral Error={lateral_errors[-1] if episode_lateral_errors else 'N/A'}")
    
    # Compute statistics
    results = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'success_rate': success_count / n_episodes,
        'mean_lateral_error': np.mean(lateral_errors) if lateral_errors else None,
    }
    
    if lap_times:
        results['mean_lap_time'] = np.mean(lap_times)
        results['std_lap_time'] = np.std(lap_times)
        results['best_lap_time'] = np.min(lap_times)
    
    return results


def evaluate_baseline(env, n_episodes=10):
    """
    Evaluate baseline time_optimal_controller.
    """
    episode_rewards = []
    episode_lengths = []
    lap_times = []
    success_count = 0
    lateral_errors = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        step_count = 0
        episode_lateral_errors = []
        
        while not done and step_count < env.max_steps:
            # Get action from baseline controller
            # Extract state from environment
            state = env.state
            ax_cmd, delta = time_optimal_controller(
                state, env.track_x, env.track_y, env.tangent, 
                env.normal, env.track_width
            )
            action = np.array([ax_cmd, delta])
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            step_count += 1
            
            if hasattr(env, 'lateral_error'):
                episode_lateral_errors.append(abs(env.lateral_error))
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(step_count)
        
        if episode_lateral_errors:
            lateral_errors.append(np.mean(episode_lateral_errors))
        
        if hasattr(env, 'track_position'):
            if env.track_position >= len(env.track_x) - 2:
                success_count += 1
                lap_time = step_count * env.dt
                lap_times.append(lap_time)
        
        print(f"Baseline Episode {episode + 1}/{n_episodes}: "
              f"Reward={episode_reward:.2f}, Steps={step_count}")
    
    results = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'success_rate': success_count / n_episodes,
        'mean_lateral_error': np.mean(lateral_errors) if lateral_errors else None,
    }
    
    if lap_times:
        results['mean_lap_time'] = np.mean(lap_times)
        results['std_lap_time'] = np.std(lap_times)
        results['best_lap_time'] = np.min(lap_times)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate BC policy')
    parser.add_argument('--model', type=str, default='bc_policy.pth',
                       help='Path to trained BC model')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of evaluation episodes')
    parser.add_argument('--compare-baseline', action='store_true',
                       help='Also evaluate baseline controller')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not Path(args.model).exists():
        print(f"Error: Model file {args.model} not found!")
        print("Train a BC model first with: python train_bc.py")
        return
    
    # Load BC model
    print(f"Loading BC model from {args.model}...")
    model = PolicyNetwork()
    checkpoint = torch.load(args.model)
    # Handle both checkpoint format and direct state_dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Create environment
    env = RacingEnv(track_width=4.0, max_steps=10000, track_variations=False)
    
    # Evaluate BC policy
    print("\n" + "="*60)
    print("Evaluating BC Policy")
    print("="*60)
    bc_results = evaluate_bc_policy(model, env, n_episodes=args.episodes)
    
    print("\n" + "="*60)
    print("BC Policy Results:")
    print("="*60)
    print(f"Mean Reward: {bc_results['mean_reward']:.2f} ± {bc_results['std_reward']:.2f}")
    print(f"Mean Episode Length: {bc_results['mean_length']:.1f} ± {bc_results['std_length']:.1f} steps")
    print(f"Success Rate: {bc_results['success_rate']*100:.1f}%")
    
    if bc_results['mean_lateral_error'] is not None:
        print(f"Mean Lateral Error: {bc_results['mean_lateral_error']:.3f} m")
    
    if 'mean_lap_time' in bc_results:
        print(f"Mean Lap Time: {bc_results['mean_lap_time']:.2f} ± {bc_results['std_lap_time']:.2f} s")
        print(f"Best Lap Time: {bc_results['best_lap_time']:.2f} s")
    
    # Optionally compare with baseline
    if args.compare_baseline:
        print("\n" + "="*60)
        print("Evaluating Baseline Controller")
        print("="*60)
        baseline_results = evaluate_baseline(env, n_episodes=args.episodes)
        
        print("\n" + "="*60)
        print("Baseline Controller Results:")
        print("="*60)
        print(f"Mean Reward: {baseline_results['mean_reward']:.2f} ± {baseline_results['std_reward']:.2f}")
        print(f"Mean Episode Length: {baseline_results['mean_length']:.1f} ± {baseline_results['std_length']:.1f} steps")
        print(f"Success Rate: {baseline_results['success_rate']*100:.1f}%")
        
        if baseline_results['mean_lateral_error'] is not None:
            print(f"Mean Lateral Error: {baseline_results['mean_lateral_error']:.3f} m")
        
        if 'mean_lap_time' in baseline_results:
            print(f"Mean Lap Time: {baseline_results['mean_lap_time']:.2f} ± {baseline_results['std_lap_time']:.2f} s")
            print(f"Best Lap Time: {baseline_results['best_lap_time']:.2f} s")
        
        # Comparison
        print("\n" + "="*60)
        print("Comparison (BC vs Baseline):")
        print("="*60)
        reward_diff = bc_results['mean_reward'] - baseline_results['mean_reward']
        print(f"Reward Difference: {reward_diff:+.2f} ({reward_diff/abs(baseline_results['mean_reward'])*100:+.1f}%)")
        
        if 'mean_lap_time' in bc_results and 'mean_lap_time' in baseline_results:
            time_diff = bc_results['mean_lap_time'] - baseline_results['mean_lap_time']
            print(f"Lap Time Difference: {time_diff:+.2f} s ({time_diff/baseline_results['mean_lap_time']*100:+.1f}%)")


if __name__ == '__main__':
    main()
