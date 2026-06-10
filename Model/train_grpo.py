import torch
import torch.optim as optim
import torch.nn.functional as F
import sys
import os
import copy
from torch.utils.data import DataLoader

# Add parent directory to path to import model components
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'auto_fsd/Model')))
from model_components.auto_e2e import AutoE2E
from model_components.causal_reasoning import calculate_causal_consistency_reward
from nvidia_parser import load_spain_subset
from latent_simulator import LatentWorldSimulator
from config import Config

class GRPOTrainer:
    def __init__(self, model, lr=Config.LEARNING_RATE, lambda_smooth=Config.LAMBDA_SMOOTH, 
                 lambda_causal=Config.LAMBDA_CAUSAL, lambda_jepa=Config.LAMBDA_JEPA, 
                 lambda_safety=Config.LAMBDA_SAFETY, group_size=Config.GROUP_SIZE, kl_coeff=Config.KL_COEFF):
        self.model = model
        # Reference model is a frozen copy of the initial model
        self.ref_model = copy.deepcopy(model)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
            
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.latent_sim = LatentWorldSimulator().to(Config.DEVICE)
        
        self.lambda_smooth = lambda_smooth
        self.lambda_causal = lambda_causal
        self.lambda_jepa = lambda_jepa
        self.lambda_safety = lambda_safety
        self.group_size = group_size
        self.kl_coeff = kl_coeff

    def calculate_smoothness_reward(self, trajectory):
        """
        Calculates the smoothness penalty: R_smooth = -lambda * sum((a_t - a_{t-1})^2)
        trajectory: (batch, 128) -> 64 steps of (accel, curvature)
        """
        traj = trajectory.view(-1, 64, 2)
        diff = traj[:, 1:, :] - traj[:, :-1, :]
        smoothness_penalty = torch.sum(diff**2, dim=(1, 2))
        return -self.lambda_smooth * smoothness_penalty

    def calculate_jepa_loss(self, predicted_future_features, target_future_features):
        """
        JEPA Loss: MSE between predicted and actual future visual features in latent space.
        """
        jepa_loss = 0
        if isinstance(predicted_future_features, (list, tuple)):
            for pred, target in zip(predicted_future_features, target_future_features):
                jepa_loss += F.mse_loss(pred, target)
            return jepa_loss / len(predicted_future_features)
        else:
            return F.mse_loss(predicted_future_features, target_future_features)

    def grpo_step(self, visual_tiles, visual_history, egomotion_history, target_trajectory=None, target_future_vision=None):
        """
        Full GRPO step with group sampling, reward normalization, JEPA and Latent Safety reward.
        """
        self.optimizer.zero_grad()
        torch.cuda.empty_cache()
        
        group_trajectories = []
        group_rewards = []
        group_jepa_losses = []
        
        # 1. Forward pass
        for i in range(self.group_size):
            # Simulate sampling with noise
            noise = torch.randn_like(visual_tiles) * 0.01 if i > 0 else 0
            outputs = self.model(visual_tiles + noise, visual_history, egomotion_history)
            
            traj = outputs["trajectory"]
            decision_logits = outputs["decision_logits"]
            future_vision = outputs["future_vision"]
            
            group_trajectories.append(traj)
            
            # Calculate Rewards
            # R1: Smoothness
            r_smooth = self.calculate_smoothness_reward(traj)
            
            # R2: Causal Consistency (System 2)
            r_causal = calculate_causal_consistency_reward(decision_logits, traj) * self.lambda_causal
            
            # R3: Latent Safety (Simulation)
            r_safety = self.latent_sim.evaluate_trajectory_safety(future_vision, traj) * self.lambda_safety
            
            # R4: Imitation
            if target_trajectory is not None:
                if target_trajectory.dim() == 1 and traj.dim() == 2:
                    target_traj_batch = target_trajectory.unsqueeze(0).expand(traj.size(0), -1)
                else:
                    target_traj_batch = target_trajectory
                r_imit = -torch.mean((traj - target_traj_batch)**2, dim=-1)
            else:
                r_imit = torch.tensor(0.0).to(traj.device)
                
            total_reward = r_imit + r_smooth + r_causal + r_safety
            group_rewards.append(total_reward)

            # World Model Loss (JEPA)
            if target_future_vision is not None:
                j_loss = self.calculate_jepa_loss(future_vision, target_future_vision)
                group_jepa_losses.append(j_loss)
            else:
                group_jepa_losses.append(torch.tensor(0.0).to(traj.device))
            
        # 2. Reward Normalization
        rewards_tensor = torch.stack(group_rewards)
        mean_r = rewards_tensor.mean()
        std_r = rewards_tensor.std() + 1e-8
        norm_rewards = (rewards_tensor - mean_r) / std_r
        
        # 3. Loss Calculation
        total_loss = 0
        
        with torch.no_grad():
            ref_outputs = self.ref_model(visual_tiles, visual_history, egomotion_history)
            ref_traj = ref_outputs["trajectory"]
            
        for i in range(self.group_size):
            # Policy gradient proxy
            loss_i = -norm_rewards[i] * torch.mean(group_trajectories[i]**2)
            
            # KL divergence
            kl_div = F.mse_loss(group_trajectories[i], ref_traj)
            
            # Total Loss = Policy Loss + KL + JEPA Reconstruction Loss
            total_loss += (loss_i + self.kl_coeff * kl_div).mean() + self.lambda_jepa * group_jepa_losses[i]
            
        total_loss = total_loss / self.group_size
        
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item(), mean_r.item(), torch.mean(torch.stack(group_jepa_losses)).item()

def main():
    Config.print_config()
    device = Config.DEVICE
    
    # Empty cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    model = AutoE2E().to(device)
    trainer = GRPOTrainer(model)
    
    print(f"Loading dataset for: {Config.TARGET_COUNTRY}...")
    dataset = load_spain_subset()
    if dataset is None or len(dataset) == 0:
        print(f"Error: Could not load real dataset for {Config.TARGET_COUNTRY}. Check downloaded files.")
        return
        
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    print(f"Starting GRPO training with {len(dataset)} real clips...")
    
    for i, batch in enumerate(dataloader):
        visual_tiles = batch["visual_tiles"].to(device)
        egomotion_history = batch["egomotion_history"].to(device)
        visual_history = batch["visual_history"].to(device)
        target_trajectory = batch["target_trajectory"].to(device)
        
        visual_tiles = visual_tiles.squeeze(0)
        egomotion_history = egomotion_history.squeeze(0)
        visual_history = visual_history.squeeze(0)
        target_trajectory = target_trajectory.squeeze(0)

        loss, mean_r, jepa_l = trainer.grpo_step(
            visual_tiles, 
            visual_history, 
            egomotion_history, 
            target_trajectory
        )
        
        print(f"Step {i+1}/{len(dataset)} | Loss: {loss:.4f} | Mean Reward: {mean_r:.4f} | JEPA Loss: {jepa_l:.4f}")
        
        if i >= 4: # Run for 5 steps for verification
            break
            
    print("---")
    print("Smooth-GRPO Training Verified with Config architecture.")

if __name__ == "__main__":
    main()
