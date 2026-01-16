"""
Behavior cloning: Pre-train policy network on expert demonstrations.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
from pathlib import Path
import matplotlib.pyplot as plt


class DemonstrationDataset(Dataset):
    """PyTorch dataset for behavior cloning."""
    
    def __init__(self, observations, actions):
        self.observations = torch.FloatTensor(observations)
        self.actions = torch.FloatTensor(actions)
    
    def __len__(self):
        return len(self.observations)
    
    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx]


class PolicyNetwork(nn.Module):
    """
    Simplified policy network for centerline following.
    2-layer MLP: [128, 64] -> 2 outputs (ax_cmd, delta)
    Smaller network for faster learning on simplified 8D observations.
    """
    
    def __init__(self, obs_dim=8, action_dim=2, hidden_sizes=[128, 64]):
        super().__init__()
        
        layers = []
        input_dim = obs_dim
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.ReLU())
            input_dim = hidden_size
        
        # Output layer
        layers.append(nn.Linear(input_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, obs):
        return self.network(obs)


def train_bc(data_path='demonstrations.pkl', 
             save_path='bc_policy.pth',
             epochs=50,
             batch_size=256,
             learning_rate=3e-4,
             validation_split=0.1,
             num_workers=4):
    """
    Train policy network via behavior cloning.
    
    Args:
        data_path: Path to demonstrations pickle file
        save_path: Path to save trained model
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for Adam optimizer
        validation_split: Fraction of data for validation
        num_workers: Number of data loading workers (0 = main thread)
    """
    # Load demonstrations
    print(f"Loading demonstrations from {data_path}...")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    observations = data['observations']
    actions = data['actions']
    
    print(f"Loaded {len(observations)} transitions")
    print(f"Observation shape: {observations.shape}")
    print(f"Action shape: {actions.shape}")
    
    # Split into train/validation
    n_samples = len(observations)
    n_val = int(n_samples * validation_split)
    n_train = n_samples - n_val
    
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    train_dataset = DemonstrationDataset(
        observations[train_indices], 
        actions[train_indices]
    )
    val_dataset = DemonstrationDataset(
        observations[val_indices],
        actions[val_indices]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True)
    
    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    obs_dim = observations.shape[1]
    action_dim = actions.shape[1]
    model = PolicyNetwork(obs_dim=obs_dim, action_dim=action_dim).to(device)
    
    print(f"\nModel architecture:")
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print(f"\nStarting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for obs_batch, action_batch in train_loader:
            obs_batch = obs_batch.to(device)
            action_batch = action_batch.to(device)
            
            # Forward pass
            pred_actions = model(obs_batch)
            loss = criterion(pred_actions, action_batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for obs_batch, action_batch in val_loader:
                obs_batch = obs_batch.to(device)
                action_batch = action_batch.to(device)
                
                pred_actions = model(obs_batch)
                loss = criterion(pred_actions, action_batch)
                
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, save_path)
            print(f"  -> Saved new best model (val_loss: {val_loss:.6f})")
    
    print(f"\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Model saved to {save_path}")
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Behavior Cloning Training')
    plt.legend()
    plt.grid(True)
    plt.savefig('bc_training_curves.png')
    print(f"Training curves saved to bc_training_curves.png")
    
    # Compute per-action MSE on validation set
    model.eval()
    with torch.no_grad():
        all_preds = []
        all_targets = []
        
        for obs_batch, action_batch in val_loader:
            obs_batch = obs_batch.to(device)
            pred_actions = model(obs_batch)
            all_preds.append(pred_actions.cpu().numpy())
            all_targets.append(action_batch.numpy())
        
        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)
        
        ax_mse = np.mean((all_preds[:, 0] - all_targets[:, 0])**2)
        delta_mse = np.mean((all_preds[:, 1] - all_targets[:, 1])**2)
        
        print(f"\nPer-action validation MSE:")
        print(f"  ax_cmd: {ax_mse:.6f} (RMSE: {np.sqrt(ax_mse):.4f} m/s²)")
        print(f"  delta:  {delta_mse:.6f} (RMSE: {np.sqrt(delta_mse):.4f} rad = {np.rad2deg(np.sqrt(delta_mse)):.2f}°)")
    
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train behavior cloning policy')
    parser.add_argument('--data', type=str, default='demonstrations.pkl', help='Path to demonstrations')
    parser.add_argument('--output', type=str, default='bc_policy.pth', help='Output model path')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--workers', type=int, default=4, help='Number of data loading workers')
    
    args = parser.parse_args()
    
    # Train behavior cloning model
    model = train_bc(
        data_path=args.data,
        save_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        validation_split=0.1,
        num_workers=args.workers
    )
