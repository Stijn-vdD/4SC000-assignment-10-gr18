"""
Visualize a trained RL policy racing on the track.
Saves output to visualization.png
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from stable_baselines3 import SAC
from racing_env import RacingEnv
import sys
from pathlib import Path


def visualize_rl_policy(model_path, num_laps=1, deterministic=True):
    """
    Visualize trained RL policy racing on track.
    
    Args:
        model_path: Path to trained SAC model (with or without .zip)
        num_laps: Number of laps to complete
        deterministic: Use deterministic policy
    """
    # Convert to Path object for easier handling
    model_path_obj = Path(model_path)
    
    # Handle both with and without .zip extension
    if str(model_path_obj).endswith('.zip'):
        model_load_path = str(model_path_obj)[:-4]  # Remove .zip for loading
        model_file = model_path_obj
    else:
        model_load_path = str(model_path_obj)
        model_file = model_path_obj.with_suffix('.zip')
    
    print(f"Loading model from {model_file}...")
    
    # Verify file exists
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file.absolute()}")
    
    try:
        model = SAC.load(model_load_path)
    except Exception as e:
        raise FileNotFoundError(f"Failed to load model from {model_load_path}: {str(e)}")
    
    # Create environment
    env = RacingEnv(track_width=4.0, max_steps=10000, track_variations=False)
    
    # Extract track info for visualization
    track_x = env.track_x
    track_y = env.track_y
    track_width = env.track_width
    normal = env.normal
    waypts = env.waypts
    
    # Compute track boundaries
    left_x = track_x + track_width * normal[:, 0]
    left_y = track_y + track_width * normal[:, 1]
    right_x = track_x - track_width * normal[:, 0]
    right_y = track_y - track_width * normal[:, 1]
    
    # Setup visualization
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.set_facecolor("white")
    ax.set_aspect("equal", adjustable="box")
    
    # Asphalt ribbon
    road_poly_x = np.concatenate([left_x, right_x[::-1]])
    road_poly_y = np.concatenate([left_y, right_y[::-1]])
    road_patch = Polygon(
        np.column_stack([road_poly_x, road_poly_y]),
        closed=True,
        facecolor=(0.2, 0.2, 0.2),
        edgecolor="none",
        alpha=0.95
    )
    ax.add_patch(road_patch)
    
    # Centerline
    ax.plot(track_x, track_y, 'w--', linewidth=1.0, alpha=0.5)
    
    # Start/finish line
    sf_n = normal[0, :]
    sf_pt = waypts[0, :]
    ax.plot([sf_pt[0] - sf_n[0] * track_width, sf_pt[0] + sf_n[0] * track_width],
            [sf_pt[1] - sf_n[1] * track_width, sf_pt[1] + sf_n[1] * track_width],
            color='yellow', linewidth=4, label='Start/Finish')
    
    # Walls
    ax.plot(left_x, left_y, 'k', linewidth=2)
    ax.plot(right_x, right_y, 'k', linewidth=2)
    
    # Trajectory line
    traj_line, = ax.plot([], [], 'cyan', linewidth=2, label='Trajectory', alpha=0.7)
    
    # Kart visualization
    kart_size = 1.0
    kart_scale = 1.1
    body_local = np.array([
        [1.6, 0.0],
        [-1.0, 0.7],
        [-1.0, -0.7]
    ]) * kart_size * kart_scale
    
    body_patch = Polygon(body_local, closed=True, facecolor=(0.2, 0.8, 0.2),
                         edgecolor='k', linewidth=1.4, zorder=10)
    ax.add_patch(body_patch)
    
    canopy_local = np.array([[0.45, 0.18], [-0.05, 0.45], [-0.35, 0.05]]) * kart_size * kart_scale
    canopy_patch = Polygon(canopy_local, closed=True, facecolor=(0.2, 0.6, 1.0),
                           edgecolor='k', linewidth=0.9, zorder=12)
    ax.add_patch(canopy_patch)
    
    # Wheels
    wheel_offsets_local = np.array([[0.9, -0.6], [-0.9, -0.6], [0.9, 0.6], [-0.9, 0.6]]) * kart_size * kart_scale
    wheel_radius = 0.28 * kart_size * kart_scale
    wheel_patches = []
    for _ in range(4):
        c = plt.Circle((0, 0), wheel_radius, facecolor='black', edgecolor='k', linewidth=0.6, zorder=9)
        ax.add_patch(c)
        wheel_patches.append(c)
    
    # Set limits
    margin = 10
    ax.set_xlim(np.min(track_x) - margin, np.max(track_x) + margin)
    ax.set_ylim(np.min(track_y) - margin, np.max(track_y) + margin)
    
    # Info text
    info_txt = ax.text(
        np.min(track_x) - margin + 5,
        np.max(track_y) + margin - 5,
        "",
        fontsize=10,
        fontweight='bold',
        color='black',
        bbox=dict(facecolor=(1, 1, 1, 0.8), edgecolor='k', boxstyle='round,pad=0.5')
    )
    
    ax.set_title("RL Policy Racing (Trained with SAC)", fontsize=14, fontweight='bold')
    ax.set_xlabel("x [scaled]")
    ax.set_ylabel("y [scaled]")
    ax.legend(loc='upper right', fontsize=9)
    
    # Run episodes with video saving
    obs, info = env.reset()
    
    traj_x = []
    traj_y = []
    laps_completed = 0
    last_s = env.s_progress
    episode_time = 0.0
    frame_count = 0
    
    print(f"\nStarting visualization...")
    print(f"Target laps: {num_laps}\n")
    
    try:
        while laps_completed < num_laps:
                # Get action from policy
                action, _ = model.predict(obs, deterministic=deterministic)
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Store trajectory
                traj_x.append(env.state[0])
                traj_y.append(env.state[1])
                
                # Update visualization
                x, y, psi = env.state[0], env.state[1], env.state[2]
                vx = env.state[3]
                
                # Rotate and translate kart
                c = np.cos(psi)
                s = np.sin(psi)
                Rmat = np.array([[c, -s], [s, c]])
                
                kart_world = (Rmat @ body_local.T).T + np.array([x, y])
                body_patch.set_xy(kart_world)
                
                canopy_world = (Rmat @ canopy_local.T).T + np.array([x, y])
                canopy_patch.set_xy(canopy_world)
                
                for i, off in enumerate(wheel_offsets_local):
                    wp = (Rmat @ off) + np.array([x, y])
                    wheel_patches[i].center = (wp[0], wp[1])
                
                # Update trajectory
                traj_line.set_xdata(traj_x[-500:])  # Keep last 500 points
                traj_line.set_ydata(traj_y[-500:])
                
                # Check for lap completion
                if last_s > env.track_length * 0.9 and env.s_progress < env.track_length * 0.1:
                    laps_completed += 1
                    print(f"Lap {laps_completed} completed! Time: {episode_time:.2f}s")
                    
                    if laps_completed >= num_laps:
                        body_patch.set_facecolor((1.0, 0.84, 0.0))  # Gold color
                
                last_s = env.s_progress
                episode_time += env.dt
                
                # Update info text
                info_txt.set_text(
                    f"Time: {episode_time:.1f}s\n"
                    f"Speed: {vx:.1f} m/s\n"
                    f"Lap: {laps_completed}/{num_laps}\n"
                    f"Progress: {env.s_progress/env.track_length*100:.1f}%\n"
                    f"Penalty: {env.penalty_offtrack:.2f}"
                )
                
                frame_count += 1
                
                # Check termination
                if terminated:
                    print(f"Episode terminated (crash or off-track)")
                    break
                
                if truncated and laps_completed >= num_laps:
                    break
    
    except KeyboardInterrupt:
        print("\nVisualization interrupted by user")
    
    print(f"\nVisualization complete!")
    print(f"Total time: {episode_time:.2f}s")
    print(f"Laps completed: {laps_completed}/{num_laps}")
    print(f"Off-track penalty: {env.penalty_offtrack:.2f}")
    
    # Save final plot
    plot_path = 'visualization.png'
    plt.tight_layout()
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    print(f"Visualization saved to {plot_path}")
    
    plt.close()
    
    env.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize trained RL policy')
    parser.add_argument('--model', type=str, default='./sac_logs/final_model.zip',
                        help='Path to trained model')
    parser.add_argument('--laps', type=int, default=1,
                        help='Number of laps to complete')
    parser.add_argument('--stochastic', action='store_true',
                        help='Use stochastic policy (default: deterministic)')
    
    args = parser.parse_args()
    
    # try:
    visualize_rl_policy(
        args.model,
        num_laps=args.laps,
        deterministic=not args.stochastic
    )
    # except FileNotFoundError:
    #     print(f"\nError: Model file '{args.model}' not found!")
    #     print("\nPlease train a model first:")
    #     print("  python train_rl.py")
    #     print("\nOr specify a different model path:")
    #     print("  python visualize_policy.py --model <path_to_model.zip>")
    #     sys.exit(1)
