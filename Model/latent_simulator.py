import torch
import torch.nn as nn
import torch.nn.functional as F

class LatentWorldSimulator(nn.Module):
    """
    Lightweight simulator operating in the AutoE2E latent space.
    Uses AutoSplat logic to evaluate trajectory safety without full 3D rendering.
    """
    def __init__(self, latent_channels=1440, grid_size=8):
        super(LatentWorldSimulator, self).__init__()
        self.latent_channels = latent_channels
        self.grid_size = grid_size
        
        # Project trajectory to latent space
        # Maps (Batch, 64, 2) -> (Batch, 4, grid_size, grid_size)
        # to compare with the 4 FutureState outputs
        self.traj_to_latent = nn.Sequential(
            nn.Linear(128, 512),
            nn.GELU(),
            nn.Linear(512, 4 * grid_size * grid_size)
        )

    def evaluate_trajectory_safety(self, predicted_future_states, proposed_trajectory):
        """
        Calculates a safety reward by comparing the trajectory with latent occupancy.
        
        Args:
            predicted_future_states: Tuple of 4 tensors (Batch, 1440, 8, 8) from FutureState.
            proposed_trajectory: Tensor (Batch, 128) with 64 steps of (accel, curv).
            
        Returns:
            safety_reward: Negative reward if there is a latent collision.
        """
        if predicted_future_states is None:
            return torch.zeros(proposed_trajectory.shape[0]).to(proposed_trajectory.device)

        batch_size = proposed_trajectory.shape[0]
        
        # 1. Map trajectory to a "planned occupancy mask" on the 8x8 grid
        # This simulates where the car will be in the 4 time intervals (1.6s each)
        traj_mask = self.traj_to_latent(proposed_trajectory)
        traj_mask = traj_mask.view(batch_size, 4, self.grid_size, self.grid_size)
        traj_mask = torch.sigmoid(traj_mask) # Ego-car occupancy probability
        
        collision_penalty = 0
        
        for i, future_state in enumerate(predicted_future_states):
            # 2. Estimate environment occupancy from latent embedding
            # Collapse channels to get an "obstacle density map"
            # High activations in certain channels indicate object presence
            obstacle_density = torch.mean(torch.abs(future_state), dim=1) # (Batch, 8, 8)
            obstacle_density = F.normalize(obstacle_density, dim=(1,2))
            
            # 3. Intersection between planned trajectory and obstacle density
            # Penalize if the ego-car plans to be where the World Model predicts obstacles
            intersection = traj_mask[:, i, :, :] * obstacle_density
            collision_penalty += torch.sum(intersection, dim=(1, 2))
            
        return -collision_penalty # Negative reward

if __name__ == "__main__":
    # Quick simulator test
    sim = LatentWorldSimulator()
    
    # Mock of 4 JEPA future states (Batch=1)
    future_states = tuple(torch.randn(1, 1440, 8, 8) for _ in range(4))
    # Test trajectory
    trajectory = torch.randn(1, 128)
    
    reward = sim.evaluate_trajectory_safety(future_states, trajectory)
    print(f"Safety Reward (Latent): {reward.item():.4f}")
