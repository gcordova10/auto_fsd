import torch
import torch.nn as nn


class FutureState(nn.Module):
    def __init__(self, embed_dim=256, action_dim=128, grid_size=8):
        super(FutureState, self).__init__()

        # Action projection to match visual feature map size
        # Trajectory (128) -> (embed_dim * grid_size * grid_size)
        self.action_proj = nn.Linear(action_dim, embed_dim * grid_size * grid_size)
        self.grid_size = grid_size
        self.embed_dim = embed_dim

        # Predict future visual features (4 timesteps × C channels = 4C)
        # Input: 2 * embed_dim (fused_features + projected_action)
        self.predict_future_1 = nn.Conv2d(2 * embed_dim, 2 * embed_dim, 3, 1, 1)
        self.predict_future_2 = nn.Conv2d(2 * embed_dim, 4 * embed_dim, 3, 1, 1)

        # Activation
        self.activation = nn.GELU()

    def forward(self, fused_features, trajectory):
        # fused_features: [B, C, 8, 8]
        # trajectory: [B, 128]
        
        # Project and reshape action to match visual features
        action_features = self.action_proj(trajectory)
        action_features = action_features.view(-1, self.embed_dim, self.grid_size, self.grid_size)
        
        # Condition visual features by concatenating action features
        conditioned_features = torch.cat((fused_features, action_features), dim=1)

        # Predicting 4 future visual feature vectors over a 6.4s horizon
        future_features = self.predict_future_1(conditioned_features)
        future_features = self.activation(future_features)
        future_features = self.predict_future_2(future_features)

        # Split into 4 future feature vectors: each [B, C, 8, 8]
        future_visual_features = torch.chunk(future_features, chunks=4, dim=1)

        return future_visual_features
