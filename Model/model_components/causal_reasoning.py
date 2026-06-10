import torch
import torch.nn as nn

class CausalReasoningModule(nn.Module):
    """
    System 2: Causal reasoning module for Robotaxi.
    Generates a latent representation of the maneuver justification
    and (optionally) a sequence of text tokens.
    """
    def __init__(self, input_dim=1344, hidden_dim=512, vocab_size=1000):
        super(CausalReasoningModule, self).__init__()
        
        # Reasoning encoder
        self.reasoning_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Decision Grounding classification head
        # 0: Intersection, 1: Pedestrian, 2: Traffic Light, 3: Obstacle, 4: Clear
        self.decision_grounding = nn.Linear(hidden_dim, 5)
        
        # Simplified text head (projects to vocabulary space)
        # In a real version, this would feed a Transformer decoder.
        self.text_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, feature_vector):
        # Generate reasoning embedding
        reasoning_latent = self.reasoning_encoder(feature_vector)
        
        # Classify primary cause (Grounding)
        decision_logits = self.decision_grounding(reasoning_latent)
        
        # Generate latent "thought"
        text_logits = self.text_head(reasoning_latent)
        
        return reasoning_latent, decision_logits, text_logits

def calculate_causal_consistency_reward(decision_logits, predicted_trajectory):
    """
    R_consistency: Penalizes if reasoning does not match physical action.
    Example: If cause is 'Pedestrian' (Grounding), acceleration should be low/negative.
    """
    decision = torch.argmax(decision_logits, dim=-1)
    accel = predicted_trajectory.view(-1, 64, 2)[:, :, 0] # Take acceleration
    mean_accel = torch.mean(accel, dim=1)
    
    reward = torch.zeros_like(mean_accel)
    
    # Rule: If obstacle/pedestrian (decisions 1, 2, 3), penalize high positive acceleration
    mask_hazard = (decision == 1) | (decision == 2) | (decision == 3)
    reward[mask_hazard] = -torch.clamp(mean_accel[mask_hazard], min=0.0)
    
    return reward
