"""
Collect expert demonstrations from the time_optimal_controller for behavior cloning.
Generates state-action pairs by running simulations with varied starting positions.
Uses multiprocessing for parallel episode collection.
"""
import numpy as np
import pickle
from tqdm import tqdm
from racing_env import RacingEnv
from mariokart import time_optimal_controller
from multiprocessing import Pool, cpu_count
import os


def collect_single_episode(args):
    """
    Collect a single episode of expert demonstrations.
    Designed to be called in parallel processes.
    
    Args:
        args: Tuple of (episode_id, start_s, seed)
    
    Returns:
        Dictionary with observations, actions, episode stats
    """
    episode_id, start_s, seed = args
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Create environment
    env = RacingEnv(track_width=4.0, max_steps=5000, track_variations=False)
    
    # Reset with specific starting position
    obs, info = env.reset(options={'start_s': start_s})
    
    episode_observations = []
    episode_actions = []
    episode_reward = 0.0
    episode_length = 0
    
    done = False
    
    while not done:
        # Get state for expert controller
        state = env.state
        s_progress = env.s_progress
        
        # Expert controller parameters
        params = env.params
        vx_ref = 12.0
        max_accel = env.max_accel
        max_brake = env.max_brake
        look_ahead = 6.0
        k_delta = 2.0
        max_steer = env.max_steer
        a_lat_max = 6.0
        nsamples = 20
        
        # Get expert action
        delta, ax_cmd, vx_target = time_optimal_controller(
            state, s_progress, env.s_dist, env.waypts,
            params, vx_ref, max_accel, max_brake,
            look_ahead, k_delta=k_delta, max_steer=max_steer,
            a_lat_max=a_lat_max, nsamples=nsamples
        )
        
        # Store observation and action
        expert_action = np.array([ax_cmd, delta], dtype=np.float32)
        
        episode_observations.append(obs)
        episode_actions.append(expert_action)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(expert_action)
        done = terminated or truncated
        
        episode_reward += reward
        episode_length += 1
        
        # Prevent infinite loops
        if episode_length > 5000:
            break
    
    env.close()
    
    return {
        'observations': np.array(episode_observations, dtype=np.float32),
        'actions': np.array(episode_actions, dtype=np.float32),
        'reward': episode_reward,
        'length': episode_length,
    }


def collect_demonstrations(num_episodes=100, save_path='demonstrations.pkl', n_workers=None):
    """
    Collect expert demonstrations using the time_optimal_controller.
    Uses multiprocessing to parallelize episode collection.
    
    Args:
        num_episodes: Number of episodes to collect
        save_path: Path to save collected data
        n_workers: Number of parallel workers (None = cpu_count())
    
    Returns:
        Dictionary with 'observations' and 'actions' arrays
    """
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)  # Leave one CPU free
    
    print(f"Collecting {num_episodes} expert demonstrations using {n_workers} parallel workers...")
    
    # Create temporary environment to get track length for random starting positions
    temp_env = RacingEnv(track_width=4.0, max_steps=5000, track_variations=False)
    track_length = temp_env.track_length
    temp_env.close()
    
    # Prepare arguments for parallel processing
    # Each episode gets: (episode_id, random_start_position, random_seed)
    episode_args = [
        (i, np.random.uniform(0, track_length), np.random.randint(0, 1000000))
        for i in range(num_episodes)
    ]
    
    # Collect episodes in parallel
    if n_workers > 1:
        with Pool(processes=n_workers) as pool:
            results = list(tqdm(
                pool.imap(collect_single_episode, episode_args),
                total=num_episodes,
                desc="Collecting episodes"
            ))
    else:
        # Sequential fallback
        results = [collect_single_episode(args) for args in tqdm(episode_args, desc="Collecting episodes")]
    
    # Aggregate results
    all_observations = []
    all_actions = []
    episode_returns = []
    episode_lengths = []
    
    for result in results:
        all_observations.append(result['observations'])
        all_actions.append(result['actions'])
        episode_returns.append(result['reward'])
        episode_lengths.append(result['length'])
    
    # Concatenate all episodes
    observations = np.vstack(all_observations)
    actions = np.vstack(all_actions)
    
    # Statistics
    print(f"\nCollection complete!")
    print(f"Total transitions: {len(observations)}")
    print(f"Average episode return: {np.mean(episode_returns):.2f} ± {np.std(episode_returns):.2f}")
    print(f"Average episode length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Action ranges:")
    print(f"  ax_cmd: [{actions[:, 0].min():.2f}, {actions[:, 0].max():.2f}]")
    print(f"  delta:  [{actions[:, 1].min():.2f}, {actions[:, 1].max():.2f}]")
    
    # Save to file
    data = {
        'observations': observations,
        'actions': actions,
        'episode_returns': episode_returns,
        'episode_lengths': episode_lengths,
    }
    
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"\nData saved to {save_path}")
    
    return data


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect expert demonstrations')
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes')
    parser.add_argument('--workers', type=int, default=None, help='Number of parallel workers')
    parser.add_argument('--output', type=str, default='demonstrations.pkl', help='Output file')
    
    args = parser.parse_args()
    
    # Collect demonstrations
    data = collect_demonstrations(
        num_episodes=args.episodes,
        save_path=args.output,
        n_workers=args.workers
    )
    
    print("\nDemonstration statistics:")
    print(f"Observation shape: {data['observations'].shape}")
    print(f"Action shape: {data['actions'].shape}")
