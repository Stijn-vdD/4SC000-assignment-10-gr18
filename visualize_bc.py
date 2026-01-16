"""
Visualize behavior cloning policy racing on the track.
Saves plots to file since interactive display may not be available.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend (works in headless environments)
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
from train_bc import PolicyNetwork
from racing_env import RacingEnv
import argparse
from pathlib import Path


def visualize_bc_policy(model_path, num_laps=1):
    """
    Visualize trained BC policy racing on track.
    
    Args:
        model_path: Path to trained BC model
        num_laps: Number of laps to complete
    """
    # Check if model exists
    if not Path(model_path).exists():
        print(f"Error: Model file {model_path} not found!")
        print("Train a BC model first with: python train_bc.py")
        return
    
    # Load model
    print(f"Loading BC model from {model_path}...")
    model = PolicyNetwork()
    checkpoint = torch.load(model_path)
    # Handle both checkpoint format and direct state_dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
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
    
    print("\nRunning BC policy visualization...")
    print("Saving plots to bc_visualization.png")
    
    # Setup visualization (no interactive display needed)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.set_facecolor("white")
    ax1.set_aspect("equal", adjustable="box")
    
    # Plot track on ax1
    road_poly_x = np.concatenate([left_x, right_x[::-1]])
    road_poly_y = np.concatenate([left_y, right_y[::-1]])
    ax1.add_patch(Polygon(np.column_stack([road_poly_x, road_poly_y]), 
                          closed=True, facecolor="#555555", edgecolor="white", linewidth=2))
    
    # Dashed centerline
    ax1.plot(track_x, track_y, "y--", linewidth=1, alpha=0.6, label="Centerline")
    
    # Waypoints
    ax1.plot(waypts[:, 0], waypts[:, 1], "o", color="white", markersize=4, alpha=0.5)
    
    # Car representation
    car_plot, = ax1.plot([], [], "ro", markersize=10, label="BC Policy")
    trajectory_plot, = ax1.plot([], [], "r-", linewidth=1, alpha=0.3)
    
    ax1.set_xlabel("X [m]", fontsize=12)
    ax1.set_ylabel("Y [m]", fontsize=12)
    ax1.set_title("BC Policy - Track View", fontsize=14, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Setup telemetry plot (ax2)
    time_data = []
    velocity_data = []
    lateral_error_data = []
    steering_data = []
    
    ax2_1 = ax2
    ax2_2 = ax2.twinx()
    
    vel_line, = ax2_1.plot([], [], "b-", linewidth=2, label="Velocity")
    lat_line, = ax2_2.plot([], [], "r-", linewidth=2, label="Lateral Error")
    steer_line, = ax2_2.plot([], [], "g-", linewidth=2, label="Steering")
    
    ax2_1.set_xlabel("Time [s]", fontsize=12)
    ax2_1.set_ylabel("Velocity [m/s]", fontsize=12, color="b")
    ax2_2.set_ylabel("Lateral Error [m] / Steering [rad]", fontsize=12, color="r")
    ax2_1.tick_params(axis='y', labelcolor='b')
    ax2_2.tick_params(axis='y', labelcolor='r')
    ax2.set_title("Telemetry", fontsize=14, fontweight="bold")
    ax2_1.grid(True, alpha=0.3)
    
    # Create legend
    lines = [vel_line, lat_line, steer_line]
    labels = [l.get_label() for l in lines]
    ax2_1.legend(lines, labels, loc='upper left')
    
    # Run episode
    obs, info = env.reset()
    done = False
    step_count = 0
    trajectory_x = []
    trajectory_y = []
    
    laps_completed = 0
    last_position = 0
    
    print("\nRunning BC policy visualization...")
    print("Close the plot window to exit.")
    
    while not done and laps_completed < num_laps:
        # Get action from BC policy
        with torch.no_grad():
            action = model(torch.FloatTensor(obs)).numpy()
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Get current state
        x, y, psi, vx, vy, r = env.state
        
        # Track trajectory
        trajectory_x.append(x)
        trajectory_y.append(y)
        
        # Update telemetry
        current_time = step_count * env.dt
        velocity = np.sqrt(vx**2 + vy**2)
        time_data.append(current_time)
        velocity_data.append(velocity)
        lateral_error_data.append(env.lateral_error if hasattr(env, 'lateral_error') else 0)
        steering_data.append(action[1])
        
        # Update plots every 5 steps for performance
        if step_count % 5 == 0:
            # Update car position
            car_plot.set_data([x], [y])
            trajectory_plot.set_data(trajectory_x, trajectory_y)
            
            # Update telemetry
            vel_line.set_data(time_data, velocity_data)
            lat_line.set_data(time_data, lateral_error_data)
            steer_line.set_data(time_data, steering_data)
            
            # Auto-scale telemetry
            if time_data:
                ax2_1.set_xlim(0, max(time_data) + 1)
                ax2_1.set_ylim(0, max(velocity_data) + 2)
                
                lat_range = max(abs(min(lateral_error_data)), abs(max(lateral_error_data))) + 0.5
                steer_range = max(abs(min(steering_data)), abs(max(steering_data))) + 0.2
                ax2_2.set_ylim(-max(lat_range, steer_range), max(lat_range, steer_range))
        
        step_count += 1
        
        # Check lap completion
        if hasattr(env, 'track_position'):
            current_position = env.track_position
            if current_position < last_position - len(env.track_x) / 2:
                laps_completed += 1
                lap_time = current_time
                print(f"Lap {laps_completed} completed in {lap_time:.2f}s")
            last_position = current_position
        
        # Prevent infinite loop
        if step_count >= env.max_steps:
            print(f"Reached max steps ({env.max_steps})")
            break
    
    # Save final plot
    plt.tight_layout()
    plt.savefig('bc_visualization.png', dpi=100, bbox_inches='tight')
    print("Plot saved to bc_visualization.png")
    
    # Final statistics
    print("\n" + "="*60)
    print("Run Statistics:")
    print("="*60)
    print(f"Total steps: {step_count}")
    print(f"Total time: {current_time:.2f} s")
    print(f"Laps completed: {laps_completed}")
    if velocity_data:
        print(f"Average velocity: {np.mean(velocity_data):.2f} m/s")
        print(f"Max velocity: {np.max(velocity_data):.2f} m/s")
    if lateral_error_data:
        print(f"Average lateral error: {np.mean(np.abs(lateral_error_data)):.3f} m")
        print(f"Max lateral error: {np.max(np.abs(lateral_error_data)):.3f} m")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize BC policy')
    parser.add_argument('--model', type=str, default='bc_policy.pth',
                       help='Path to trained BC model')
    parser.add_argument('--laps', type=int, default=1,
                       help='Number of laps to complete')
    
    args = parser.parse_args()
    
    visualize_bc_policy(args.model, num_laps=args.laps)


if __name__ == '__main__':
    main()
