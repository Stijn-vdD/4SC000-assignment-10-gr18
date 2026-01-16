"""
Test and evaluate trained RL policy against baseline controller.
"""
import numpy as np
import torch
from stable_baselines3 import SAC
from racing_env import RacingEnv
from mariokart import time_optimal_controller
import matplotlib.pyplot as plt
from pathlib import Path


def evaluate_policy(model, env, n_episodes=10, deterministic=True, render=False):
    """
    Evaluate a policy on the environment.
    
    Returns:
        Dictionary with episode statistics
    """
    episode_rewards = []
    episode_lengths = []
    lap_times = []
    penalties = []
    success_count = 0
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        step_count = 0
        
        while not done:
            # Get action from policy
            if isinstance(model, SAC):
                action, _ = model.predict(obs, deterministic=deterministic)
            else:
                # For baseline controller
                action = model(env, obs)
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            step_count += 1
            
            if render:
                env.render()
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(step_count)
        lap_times.append(info['lap_time'])
        penalties.append(info['penalty_offtrack'])
        
        # Success = completed lap without major crashes
        if truncated and info['penalty_offtrack'] < 5.0:
            success_count += 1
    
    results = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'mean_lap_time': np.mean(lap_times),
        'std_lap_time': np.std(lap_times),
        'mean_penalty': np.mean(penalties),
        'success_rate': success_count / n_episodes,
        'all_rewards': episode_rewards,
        'all_lap_times': lap_times,
    }
    
    return results


def baseline_controller_policy(env, obs):
    """Wrapper for time_optimal_controller to match policy interface."""
    state = env.state
    s_progress = env.s_progress
    
    params = env.params
    vx_ref = 12.0
    max_accel = env.max_accel
    max_brake = env.max_brake
    look_ahead = 6.0
    k_delta = 2.0
    max_steer = env.max_steer
    a_lat_max = 6.0
    nsamples = 20
    
    delta, ax_cmd, vx_target = time_optimal_controller(
        state, s_progress, env.s_dist, env.waypts,
        params, vx_ref, max_accel, max_brake,
        look_ahead, k_delta=k_delta, max_steer=max_steer,
        a_lat_max=a_lat_max, nsamples=nsamples
    )
    
    return np.array([ax_cmd, delta], dtype=np.float32)


def compare_controllers(rl_model_path, n_episodes=10):
    """
    Compare RL policy against baseline time_optimal_controller.
    
    Args:
        rl_model_path: Path to trained SAC model
        n_episodes: Number of episodes for evaluation
    """
    print(f"Loading RL model from {rl_model_path}...")
    rl_model = SAC.load(rl_model_path)
    
    # Create evaluation environment
    env = RacingEnv(track_width=4.0, max_steps=5000, track_variations=False)
    
    print(f"\nEvaluating RL policy over {n_episodes} episodes...")
    rl_results = evaluate_policy(rl_model, env, n_episodes=n_episodes, deterministic=True)
    
    print(f"\nEvaluating baseline controller over {n_episodes} episodes...")
    baseline_results = evaluate_policy(baseline_controller_policy, env, n_episodes=n_episodes)
    
    # Print comparison
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON")
    print("="*70)
    
    print(f"\n{'Metric':<30} {'RL Policy':<20} {'Baseline':<20}")
    print("-"*70)
    
    print(f"{'Mean Lap Time (s)':<30} "
          f"{rl_results['mean_lap_time']:>8.2f} ± {rl_results['std_lap_time']:>5.2f}    "
          f"{baseline_results['mean_lap_time']:>8.2f} ± {baseline_results['std_lap_time']:>5.2f}")
    
    print(f"{'Mean Episode Reward':<30} "
          f"{rl_results['mean_reward']:>8.2f} ± {rl_results['std_reward']:>5.2f}    "
          f"{baseline_results['mean_reward']:>8.2f} ± {baseline_results['std_reward']:>5.2f}")
    
    print(f"{'Mean Off-track Penalty':<30} "
          f"{rl_results['mean_penalty']:>8.2f}               "
          f"{baseline_results['mean_penalty']:>8.2f}")
    
    print(f"{'Success Rate':<30} "
          f"{rl_results['success_rate']*100:>8.1f}%              "
          f"{baseline_results['success_rate']*100:>8.1f}%")
    
    print(f"{'Mean Episode Length':<30} "
          f"{rl_results['mean_length']:>8.1f}               "
          f"{baseline_results['mean_length']:>8.1f}")
    
    # Compute improvement
    if baseline_results['mean_lap_time'] > 0:
        lap_time_improvement = (
            (baseline_results['mean_lap_time'] - rl_results['mean_lap_time']) / 
            baseline_results['mean_lap_time'] * 100
        )
        print(f"\nLap Time Improvement: {lap_time_improvement:+.1f}%")
    
    print("="*70)
    
    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Lap times
    axes[0].boxplot([rl_results['all_lap_times'], baseline_results['all_lap_times']],
                    labels=['RL Policy', 'Baseline'])
    axes[0].set_ylabel('Lap Time (s)')
    axes[0].set_title('Lap Time Comparison')
    axes[0].grid(True, alpha=0.3)
    
    # Episode rewards
    axes[1].boxplot([rl_results['all_rewards'], baseline_results['all_rewards']],
                    labels=['RL Policy', 'Baseline'])
    axes[1].set_ylabel('Episode Reward')
    axes[1].set_title('Episode Reward Comparison')
    axes[1].grid(True, alpha=0.3)
    
    # Bar chart of success rate
    axes[2].bar(['RL Policy', 'Baseline'], 
                [rl_results['success_rate']*100, baseline_results['success_rate']*100])
    axes[2].set_ylabel('Success Rate (%)')
    axes[2].set_title('Success Rate Comparison')
    axes[2].set_ylim([0, 105])
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('controller_comparison.png', dpi=150)
    print(f"\nComparison plot saved to controller_comparison.png")
    
    env.close()
    
    return rl_results, baseline_results


def test_on_track_variations(rl_model_path, n_episodes=5):
    """
    Test RL policy on track variations to verify generalization.
    
    Args:
        rl_model_path: Path to trained SAC model
        n_episodes: Number of variation episodes to test
    """
    print(f"\nTesting generalization on track variations...")
    rl_model = SAC.load(rl_model_path)
    
    success_count = 0
    lap_times = []
    penalties = []
    
    for episode in range(n_episodes):
        env = RacingEnv(track_width=4.0, max_steps=5000, track_variations=True)
        obs, info = env.reset()
        done = False
        
        while not done:
            action, _ = rl_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
        lap_times.append(info['lap_time'])
        penalties.append(info['penalty_offtrack'])
        
        if truncated and info['penalty_offtrack'] < 5.0:
            success_count += 1
        
        env.close()
    
    print(f"\nGeneralization Test Results:")
    print(f"  Success Rate: {success_count}/{n_episodes} ({success_count/n_episodes*100:.1f}%)")
    print(f"  Mean Lap Time: {np.mean(lap_times):.2f} ± {np.std(lap_times):.2f} s")
    print(f"  Mean Penalty: {np.mean(penalties):.2f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate trained RL policy')
    parser.add_argument('--model', type=str, default='sac_logs/final_model.zip',
                        help='Path to trained model')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of evaluation episodes')
    parser.add_argument('--test-variations', action='store_true',
                        help='Test on track variations')
    
    args = parser.parse_args()
    
    if not Path(args.model).exists():
        print(f"Error: Model file {args.model} not found!")
        print("Available models:")
        for p in Path('.').rglob('*.zip'):
            print(f"  {p}")
    else:
        # Compare controllers
        rl_results, baseline_results = compare_controllers(
            args.model, 
            n_episodes=args.episodes
        )
        
        # Test generalization if requested
        if args.test_variations:
            test_on_track_variations(args.model, n_episodes=5)
