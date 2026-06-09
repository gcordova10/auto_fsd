import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import Config

class FutureState(nn.Module):
    def __init__(self):
        super(FutureState, self).__init__()

        # Action projection to a smaller feature map size to save memory
        # Trajectory (128) -> (ACTION_CONDITION_CHANNELS * GRID_SIZE * GRID_SIZE)
        self.action_proj = nn.Linear(128, Config.ACTION_CONDITION_CHANNELS * Config.GRID_SIZE * Config.GRID_SIZE)

        # Compress features
        # Input: LATENT_CHANNELS (visual) + ACTION_CONDITION_CHANNELS (action)
        self.predict_future_1 = nn.Conv2d(Config.LATENT_CHANNELS + Config.ACTION_CONDITION_CHANNELS, 2880, 3, 1, 1)
        self.predict_future_2 = nn.Conv2d(2880, 5760, 3, 1, 1)

        # Activation
        self.activation = nn.GELU()

    def forward(self, fused_features, trajectory):
        """
        Predict future states conditioned on the proposed trajectory (Action-Conditioned World Model).
        """
        # Project action and reshape to match visual feature maps
        action_features = self.action_proj(trajectory)
        
        # Reshape to (Batch, ACTION_CONDITION_CHANNELS, GRID_SIZE, GRID_SIZE)
        action_features = action_features.view(-1, Config.ACTION_CONDITION_CHANNELS, Config.GRID_SIZE, Config.GRID_SIZE)
        
        if action_features.size(0) == 1 and fused_features.size(0) > 1:
            action_features = action_features.expand(fused_features.size(0), -1, -1, -1)
        
        # Condition visual features by concatenating the action features
        conditioned_features = torch.cat((fused_features, action_features), dim=1)
        # Predicting 4 future visual feature vectors over a 6.4s horizon
        future_features = self.predict_future_1(conditioned_features)
        future_features = self.activation(future_features)
        future_features = self.predict_future_2(future_features)

        # Future feature vectors
        future_visual_features = torch.chunk(future_features, chunks=4, dim=1)

        return future_visual_features
   