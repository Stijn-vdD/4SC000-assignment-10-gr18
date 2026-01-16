"""
Gymnasium environment for the Mario Kart racing simulation.
Wraps the bicycle dynamics and track from mariokart.py.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from mariokart import (
    bicycle_dynamics, make_centerline_from_custom_track,
    project_to_track, interp_track, curvature_at_s, angle_wrap
)


class RacingEnv(gym.Env):
    """
    Custom Gymnasium environment for centerline following (tracking problem).
    
    Goal: Follow the centerline as closely as possible.
    
    Observation space (8D) - Simplified for faster learning:
        - lateral_error: distance from centerline [-track_width, track_width]
        - heading_error: angle difference from track tangent [-π, π]
        - vx, vy: vehicle velocities
        - tangent_x, tangent_y: local track direction
        - cos_psi, sin_psi: heading representation
    
    Action space (2D continuous):
        - ax_cmd: longitudinal acceleration [-6.0, 4.0] m/s²
        - delta: steering angle [-0.4363, 0.4363] rad (±25°)
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 50}
    
    def __init__(self, track_width=4.0, max_steps=5000, speed_limit=None,
                 track_variations=False, render_mode=None):
        super().__init__()
        
        self.track_width = track_width
        self.max_steps = max_steps
        self.speed_limit = speed_limit  # For curriculum learning
        self.track_variations = track_variations
        self.render_mode = render_mode
        
        # Vehicle parameters (from mariokart.py)
        self.params = {
            "m": 150.0,
            "Iz": 20.0,
            "lf": 0.7,
            "lr": 0.7,
            "Cf": 800.0,
            "Cr": 800.0,
        }
        
        # Control constraints
        self.max_steer = np.deg2rad(25.0)
        self.max_accel = 4.0
        self.max_brake = -6.0
        
        # Simulation timestep
        self.dt = 0.02
        
        # Build track
        self._build_track()
        
        # Define observation space (8D) - Simplified for centerline following
        obs_high = np.array([
            self.track_width * 2,  # lateral_error
            np.pi,                  # heading_error
            20.0,                   # vx
            10.0,                   # vy
            1.0,                    # tangent_x
            1.0,                    # tangent_y
            1.0,                    # cos_psi
            1.0,                    # sin_psi
        ], dtype=np.float32)
        
        self.observation_space = spaces.Box(
            low=-obs_high,
            high=obs_high,
            dtype=np.float32
        )
        
        # Define action space (2D continuous)
        self.action_space = spaces.Box(
            low=np.array([self.max_brake, -self.max_steer], dtype=np.float32),
            high=np.array([self.max_accel, self.max_steer], dtype=np.float32),
            dtype=np.float32
        )
        
        # Episode tracking
        self.state = None
        self.s_progress = 0.0
        self.steps = 0
        self.last_s_progress = 0.0
        self.total_reward = 0.0
        self.penalty_offtrack = 0.0
        self.episode_start_time = 0.0
        
    def _build_track(self):
        """Build track centerline and compute normals."""
        track_x, track_y = make_centerline_from_custom_track()
        
        # Apply variations if enabled
        if self.track_variations:
            # Randomly perturb control points (±10%)
            noise_scale = 0.1
            track_x += np.random.randn(len(track_x)) * noise_scale * np.std(track_x)
            track_y += np.random.randn(len(track_y)) * noise_scale * np.std(track_y)
            
            # Optionally vary track width (±12.5%)
            width_variation = np.random.uniform(0.875, 1.125)
            self.track_width = 4.0 * width_variation
        
        # Close loop
        track_x[-1] = track_x[0]
        track_y[-1] = track_y[0]
        
        # Compute tangents and normals
        dx = np.gradient(track_x)
        dy = np.gradient(track_y)
        tang = np.stack([dx, dy], axis=1)
        tang_norm = np.linalg.norm(tang, axis=1).reshape(-1, 1)
        tang_norm[tang_norm < 1e-12] = 1e-12
        tang = tang / tang_norm
        normal = np.stack([-tang[:, 1], tang[:, 0]], axis=1)
        
        self.track_x = track_x
        self.track_y = track_y
        self.tangent = tang
        self.normal = normal
        
        # Waypoints and arc-length
        self.waypts = np.column_stack((track_x, track_y))
        diffs = np.diff(self.waypts, axis=0)
        seglen = np.sqrt(np.sum(diffs**2, axis=1))
        self.s_dist = np.concatenate(([0.0], np.cumsum(seglen)))
        self.track_length = self.s_dist[-1]
        
    def _get_obs(self):
        """Construct observation vector from current state."""
        x, y, psi, vx, vy, r = self.state
        
        # Project to track
        s_progress, closest_pt = project_to_track(
            np.array([x, y]), self.s_dist, self.waypts
        )
        
        # Lateral error (signed distance from centerline)
        lateral_error = np.linalg.norm(np.array([x, y]) - closest_pt)
        # Determine sign (left/right of track)
        to_vehicle = np.array([x, y]) - closest_pt
        idx = np.searchsorted(self.s_dist, s_progress, side='right') - 1
        idx = np.clip(idx, 0, len(self.normal) - 1)
        normal_at_s = self.normal[idx]
        lateral_sign = np.sign(np.dot(to_vehicle, normal_at_s))
        lateral_error *= lateral_sign
        
        # Track tangent direction at current position
        tangent_at_s = self.tangent[idx]
        track_psi = np.arctan2(tangent_at_s[1], tangent_at_s[0])
        
        # Heading error
        heading_error = angle_wrap(psi - track_psi)
        
        # Store for reward computation
        self.lateral_error = lateral_error
        self.heading_error = heading_error
        
        # Simplified observation: only what's needed for centerline tracking
        obs = np.array([
            lateral_error,
            heading_error,
            vx,
            vy,
            tangent_at_s[0],
            tangent_at_s[1],
            np.cos(psi),
            np.sin(psi),
        ], dtype=np.float32)
        
        return obs
    
    def _compute_reward(self, obs, action, terminated):
        """
        Simplified reward for centerline following:
        - Tracking reward: exponential in lateral error (primary objective)
        - Heading alignment: penalize heading error
        - Forward progress: encourage forward velocity
        - Smoothness: penalize aggressive control inputs
        - Off-track penalty: large penalty for leaving track
        """
        lateral_error = obs[0]
        heading_error = obs[1]
        vx = obs[2]
        
        # Primary: Track centerline with exponential penalty on deviation
        # Reward = 1.0 when on centerline, drops exponentially with error
        r_tracking = np.exp(-2.0 * abs(lateral_error))
        
        # Secondary: Align heading with track direction
        r_heading = np.exp(-1.0 * abs(heading_error))
        
        # Tertiary: Maintain forward velocity (encourage progress)
        target_speed = 8.0  # Moderate target speed for good tracking
        r_speed = -0.1 * (vx - target_speed)**2 / target_speed**2
        
        # Smoothness penalty (discourage jerky inputs)
        ax_cmd, delta = action[0], action[1]
        r_smooth = -0.05 * (abs(delta) + 0.3 * abs(ax_cmd) / self.max_accel)
        
        # Off-track penalty (goes off centerline by more than track width)
        if abs(lateral_error) > self.track_width:
            r_offtrack = -10.0
            self.penalty_offtrack += abs(lateral_error) - self.track_width
        else:
            r_offtrack = 0.0
        
        # Large penalty for termination (crash)
        r_crash = -20.0 if terminated else 0.0
        
        # Total reward (weighted combination)
        reward = 10.0 * r_tracking + 2.0 * r_heading + r_speed + r_smooth + r_offtrack + r_crash
        
        return reward
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        # Rebuild track with variations if enabled
        if self.track_variations:
            self._build_track()
        
        # Random initial position along track
        if options is not None and 'start_s' in options:
            s_start = options['start_s']
        else:
            s_start = np.random.uniform(0, self.track_length)
        
        # Get position and heading at s_start
        p_start = interp_track(s_start, self.s_dist, self.waypts)
        
        # Estimate heading from nearby points
        p_ahead = interp_track(s_start + 1.0, self.s_dist, self.waypts)
        psi_start = np.arctan2(p_ahead[1] - p_start[1], p_ahead[0] - p_start[0])
        
        # Initial velocities
        vx_start = 0.1  # Start nearly stopped
        vy_start = 0.0
        r_start = 0.0
        
        self.state = np.array([
            p_start[0], p_start[1], psi_start,
            vx_start, vy_start, r_start
        ])
        
        self.s_progress = s_start
        self.last_s_progress = s_start
        self.steps = 0
        self.total_reward = 0.0
        self.penalty_offtrack = 0.0
        self.episode_start_time = 0.0
        
        obs = self._get_obs()
        info = {
            's_progress': self.s_progress,
            'lap_time': 0.0,
            'penalty_offtrack': self.penalty_offtrack,
        }
        
        return obs, info
    
    def step(self, action):
        """Execute one simulation step."""
        # Unpack action
        ax_cmd, delta = action[0], action[1]
        
        # Apply speed limit if in curriculum mode
        if self.speed_limit is not None:
            vx_current = self.state[3]
            if vx_current > self.speed_limit:
                # Apply braking to enforce limit
                ax_cmd = min(ax_cmd, -2.0)
        
        # Clip actions to valid ranges
        ax_cmd = np.clip(ax_cmd, self.max_brake, self.max_accel)
        delta = np.clip(delta, -self.max_steer, self.max_steer)
        
        # Euler integration
        state_dot = bicycle_dynamics(self.state, delta, ax_cmd, self.params)
        self.state = self.state + state_dot * self.dt
        
        # Update progress
        self.last_s_progress = self.s_progress
        self.s_progress, _ = project_to_track(
            self.state[0:2], self.s_dist, self.waypts
        )
        
        self.steps += 1
        self.episode_start_time += self.dt
        
        # Get observation
        obs = self._get_obs()
        
        # Check termination conditions - simplified for centerline following
        lateral_error_abs = abs(obs[0])
        terminated = False
        truncated = False
        
        # Crash: significantly off track (more tolerant than before)
        if lateral_error_abs > self.track_width * 2.0:
            terminated = True
        
        # Backwards driving
        if obs[2] < -1.0:  # vx < -1 m/s
            terminated = True
        
        # Lap completion: detect wrap from high s to low s
        if (self.last_s_progress > self.track_length * 0.9 and 
            self.s_progress < self.track_length * 0.1 and 
            self.episode_start_time > 1.0):
            truncated = True  # Successfully completed lap
        
        # Max steps reached
        if self.steps >= self.max_steps:
            truncated = True
        
        # Compute reward
        reward = self._compute_reward(obs, action, terminated)
        self.total_reward += reward
        
        info = {
            's_progress': self.s_progress,
            'lap_time': self.episode_start_time,
            'penalty_offtrack': self.penalty_offtrack,
            'vx': self.state[3],
            'lateral_error': obs[0],
        }
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment (optional, for debugging)."""
        if self.render_mode == "human":
            # Could implement matplotlib visualization here
            pass
    
    def close(self):
        """Clean up resources."""
        pass


def make_racing_env(speed_limit=None, track_variations=False):
    """Factory function to create racing environment."""
    return RacingEnv(
        track_width=4.0,
        max_steps=5000,
        speed_limit=speed_limit,
        track_variations=track_variations
    )
