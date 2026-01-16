"""
Train SAC agent for centerline following.
Uses Stable-Baselines3 and initializes from behavior cloning weights.
"""
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
from pathlib import Path
import json

from racing_env import RacingEnv
from train_bc import PolicyNetwork


class CurriculumCallback(BaseCallback):
    """
    Callback to implement curriculum learning by adjusting speed limits.
    
    Phase 1 (0-50k steps): speed_limit = 8 m/s
    Phase 2 (50k-150k steps): speed_limit = 11 m/s
    Phase 3 (150k+ steps): no speed limit
    """
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.phase = 1
    
    def _on_step(self):
        n_calls = self.num_timesteps
        
        # Update curriculum phase
        if n_calls >= 150000 and self.phase < 3:
            self.phase = 3
            if self.verbose > 0:
                print(f"\n=== CURRICULUM PHASE 3: Unconstrained speed ===")
            # Remove speed limits from all environments
            for env in self.training_env.envs:
                env.speed_limit = None
        elif n_calls >= 50000 and self.phase < 2:
            self.phase = 2
            if self.verbose > 0:
                print(f"\n=== CURRICULUM PHASE 2: Speed limit 11 m/s ===")
            # Update speed limits
            for env in self.training_env.envs:
                env.speed_limit = 11.0
        
        return True


class TrackDistributionCallback(BaseCallback):
    """
    Callback to adjust track distribution during training.
    
    Initially: 85% target track, 15% variations
    After 300k steps: 95% target track, 5% variations (fine-tuning)
    """
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.fine_tuning = False
    
    def _on_step(self):
        if not self.fine_tuning and self.num_timesteps >= 300000:
            self.fine_tuning = True
            if self.verbose > 0:
                print(f"\n=== TRACK-SPECIFIC FINE-TUNING: 95% target track ===")
        
        return True


class ProgressLoggingCallback(BaseCallback):
    """Log detailed training progress."""
    
    def __init__(self, log_freq=1000, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.lap_times = []
    
    def _on_step(self):
        # Collect episode statistics
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                ep_info = info['episode']
                self.episode_rewards.append(ep_info['r'])
                self.episode_lengths.append(ep_info['l'])
        
        # Log every log_freq steps
        if self.num_timesteps % self.log_freq == 0:
            if len(self.episode_rewards) > 0:
                mean_reward = np.mean(self.episode_rewards[-100:])
                mean_length = np.mean(self.episode_lengths[-100:])
                
                if self.verbose > 0:
                    print(f"Steps: {self.num_timesteps} | "
                          f"Avg Reward (100 eps): {mean_reward:.2f} | "
                          f"Avg Length: {mean_length:.1f}")
        
        return True


def make_env(env_id, rank, speed_limit=None, track_variations_prob=0.15):
    """
    Create a single racing environment.
    
    Args:
        env_id: Environment identifier
        rank: Process rank for seeding
        speed_limit: Speed limit for curriculum (None = no limit)
        track_variations_prob: Probability of using track variations
    """
    def _init():
        # Decide whether to use track variations
        use_variations = np.random.random() < track_variations_prob
        
        env = RacingEnv(
            track_width=4.0,
            max_steps=5000,
            speed_limit=speed_limit,
            track_variations=use_variations
        )
        env = Monitor(env)
        return env
    
    return _init


def load_bc_weights(sac_model, bc_checkpoint_path):
    """
    Load behavior cloning weights into SAC actor network.
    
    Args:
        sac_model: SAC model instance
        bc_checkpoint_path: Path to BC checkpoint
    """
    print(f"\nLoading BC weights from {bc_checkpoint_path}...")
    
    # Load BC checkpoint
    bc_checkpoint = torch.load(bc_checkpoint_path, map_location='cpu')
    bc_state_dict = bc_checkpoint['model_state_dict']
    
    # Get SAC actor network
    sac_actor = sac_model.actor
    
    # Map BC weights to SAC actor
    # BC network structure: input -> hidden layers -> output
    # SAC actor structure: features_extractor -> mu (mean), log_std
    
    try:
        # Extract shared layers (features)
        actor_state_dict = sac_actor.state_dict()
        
        # Map BC weights to SAC latent_pi (shared features)
        bc_layer_idx = 0
        for name, param in actor_state_dict.items():
            if 'latent_pi' in name:
                # Map to corresponding BC layer
                bc_key = f'network.{bc_layer_idx}.weight' if 'weight' in name else f'network.{bc_layer_idx}.bias'
                if bc_key in bc_state_dict:
                    actor_state_dict[name] = bc_state_dict[bc_key]
                    print(f"  Mapped {bc_key} -> {name}")
                    if 'weight' in name:
                        bc_layer_idx += 2  # Skip ReLU layer
        
        # Load updated weights
        sac_actor.load_state_dict(actor_state_dict, strict=False)
        print("BC weights loaded successfully!")
        
    except Exception as e:
        print(f"Warning: Could not load BC weights: {e}")
        print("Starting with random initialization.")


def train_sac(
    total_timesteps=500000,
    n_envs=None,  # Auto-detect based on CPU cores
    bc_checkpoint=None,
    learning_rate=3e-4,
    buffer_size=100000,
    batch_size=256,
    gamma=0.99,
    tau=0.005,
    target_entropy='auto',
    log_dir='./sac_logs',
    save_freq=50000,
):
    """
    Train SAC agent with curriculum learning.
    
    Args:
        total_timesteps: Total training steps
        n_envs: Number of parallel environments (None = auto-detect)
        bc_checkpoint: Path to BC checkpoint for warm start
        learning_rate: Learning rate
        buffer_size: Replay buffer size
        batch_size: Batch size
        gamma: Discount factor
        tau: Soft update coefficient
        target_entropy: Target entropy (auto = -dim(A))
        log_dir: Directory for logs
        save_freq: Checkpoint save frequency
    """
    # Auto-detect number of environments based on CPU cores
    if n_envs is None:
        import multiprocessing
        n_cpus = multiprocessing.cpu_count()
        n_envs = max(2, min(8, n_cpus - 1))  # Use 2-8 envs, leave 1 CPU free
        print(f"Auto-detected {n_cpus} CPU cores, using {n_envs} parallel environments")
    
    # Create output directory
    Path(log_dir).mkdir(exist_ok=True)
    
    # Create vectorized environments (no curriculum needed for centerline tracking)
    print(f"Creating {n_envs} parallel environments...")
    
    env_fns = [make_env(f"racing-{i}", i, speed_limit=None) for i in range(n_envs)]
    
    if n_envs == 1:
        env = DummyVecEnv(env_fns)
    else:
        env = SubprocVecEnv(env_fns)
    
    # Create evaluation environment (no variations)
    eval_env = Monitor(RacingEnv(track_width=4.0, max_steps=5000, track_variations=False))
    
    # Create SAC model
    print("\nCreating SAC model...")
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        gamma=gamma,
        tau=tau,
        ent_coef='auto',
        target_entropy=target_entropy,
        policy_kwargs={
            'net_arch': [128, 64],  # Simplified for centerline following
            'activation_fn': nn.ReLU,
        },
        verbose=1,
        tensorboard_log=log_dir,
    )
    
    # Load BC weights if provided
    if bc_checkpoint is not None and Path(bc_checkpoint).exists():
        load_bc_weights(model, bc_checkpoint)
    
    print(f"\nModel created:")
    print(f"  Policy: {model.policy}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Buffer size: {buffer_size}")
    print(f"  Batch size: {batch_size}")
    
    # Create callbacks (simplified - no curriculum needed for centerline tracking)
    progress_cb = ProgressLoggingCallback(log_freq=5000, verbose=1)
    
    checkpoint_cb = CheckpointCallback(
        save_freq=save_freq // n_envs,
        save_path=f"{log_dir}/checkpoints",
        name_prefix="sac_racing",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=f"{log_dir}/best_model",
        log_path=f"{log_dir}/eval",
        eval_freq=10000 // n_envs,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )
    
    callbacks = [progress_cb, checkpoint_cb, eval_cb]
    
    # Train
    print(f"\n{'='*60}")
    print(f"Starting SAC training for centerline following: {total_timesteps} steps")
    print(f"{'='*60}\n")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        log_interval=10,
        progress_bar=True,
    )
    
    # Save final model
    final_model_path = f"{log_dir}/final_model"
    model.save(final_model_path)
    print(f"\nFinal model saved to {final_model_path}")
    
    # Save training config
    config = {
        'total_timesteps': total_timesteps,
        'n_envs': n_envs,
        'learning_rate': learning_rate,
        'buffer_size': buffer_size,
        'batch_size': batch_size,
        'gamma': gamma,
        'tau': tau,
    }
    
    with open(f"{log_dir}/config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    env.close()
    eval_env.close()
    
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train SAC racing agent')
    parser.add_argument('--timesteps', type=int, default=500000, help='Total training timesteps')
    parser.add_argument('--envs', type=int, default=None, help='Number of parallel environments (auto-detect if not specified)')
    parser.add_argument('--bc-checkpoint', type=str, default='bc_policy.pth', help='BC checkpoint path')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--buffer-size', type=int, default=100000, help='Replay buffer size')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--log-dir', type=str, default='./sac_logs', help='Log directory')
    parser.add_argument('--no-bc', action='store_true', help='Skip BC warm start')
    
    args = parser.parse_args()
    
    bc_checkpoint = None if args.no_bc else args.bc_checkpoint
    
    # Train SAC with behavior cloning warm start
    model = train_sac(
        total_timesteps=args.timesteps,
        n_envs=args.envs,
        bc_checkpoint=bc_checkpoint,
        learning_rate=args.lr,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        gamma=0.99,
        log_dir=args.log_dir,
        save_freq=50000,
    )
    
    print("\nTraining complete!")
