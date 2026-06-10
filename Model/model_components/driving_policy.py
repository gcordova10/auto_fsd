import torch
import torch.nn as nn


class DrivingPolicy(nn.Module):
    def __init__(self, embed_dim=256, visual_history_dim=896, egomotion_dim=256):
        super(DrivingPolicy, self).__init__()

        # Dimensions for the compressed visual feature vector
        compressed_dim = 14
        visual_flat_dim = 3 * 8 * 8  # After channel reduction (192)

        # 2D Conv layer to reduce channels
        self.reduce_channels = nn.Conv2d(embed_dim, 3, 3, 1, 1)

        # Temporal Memory (GRU)
        # Input: 14 (compressed visual) + 4 (egomotion) = 18 features per step
        self.temporal_gru = nn.GRU(input_size=18, hidden_size=512, batch_first=True)

        # Total input dimension to MLP: 192 (current vision) + 512 (GRU hidden state)
        mlp_input_dim = visual_flat_dim + 512

        # Linear layers to process reduced features
        self.fc1 = nn.Linear(mlp_input_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        
        # Trajectory output - Predict 5 Bézier control points (x, y) = 10 outputs
        self.fc3 = nn.Linear(512, 10)

        # Visual history compression layer
        self.compress_vision = nn.Linear(visual_flat_dim, compressed_dim)

        # Dropout
        self.dropout = nn.Dropout(0.25)

        # Activation
        self.activation = nn.GELU()

        # Precompute Bézier basis functions for 64 points
        self.register_buffer("bezier_matrix", self._precompute_bezier_matrix(num_points=64, num_controls=5))

    def _precompute_bezier_matrix(self, num_points, num_controls):
        import numpy as np
        from scipy.special import comb
        
        def bernstein_poly(i, n, t):
            return comb(n, i) * (t**(n-i)) * (1-t)**i

        t = np.linspace(0, 1, num_points)
        n = num_controls - 1
        matrix = np.zeros((num_points, num_controls))
        for i in range(num_controls):
            matrix[:, i] = bernstein_poly(i, n, t)
        return torch.from_numpy(matrix).float()

    def forward(self, fused_features, visual_history, egomotion_history):
        # fused_features: [B, C, 8, 8]
        B = fused_features.shape[0]

        # 1. Process Current Vision: [B, 3, 8, 8]
        feature_map = self.reduce_channels(fused_features)
        visual_feature_vector = torch.flatten(feature_map, start_dim=1) # [B, 192]

        # 2. Process Temporal Memory
        # visual_history [B, 896] -> [B, 64, 14]
        # egomotion_history [B, 256] -> [B, 64, 4]
        v_hist = visual_history.view(B, 64, 14)
        e_hist = egomotion_history.view(B, 64, 4)
        
        # Combine visual and egomotion sequences
        temporal_sequence = torch.cat((v_hist, e_hist), dim=2) # [B, 64, 18]
        
        # Forward pass through GRU
        _, h_n = self.temporal_gru(temporal_sequence)
        temporal_context = h_n.squeeze(0) # [B, 512]

        # 3. Concatenate Current and Temporal
        feature_vector = torch.cat((visual_feature_vector, temporal_context), dim=1)

        # Multi-layer perceptron
        f1 = self.fc1(feature_vector)
        f1 = self.activation(f1)
        f1 = self.dropout(f1)

        f2 = self.fc2(f1)
        f2 = self.activation(f2)
        f2 = self.dropout(f2)

        # Predict Bézier control points: [B, 5, 2]
        control_points = self.fc3(f2).view(-1, 5, 2)
        
        # Reconstruct trajectory using precomputed Bézier matrix: [B, 64, 2]
        trajectory = torch.matmul(self.bezier_matrix, control_points)
        trajectory = trajectory.reshape(B, 128) # Flatten for compatibility

        # Compressed visual feature vector of length 14 to form visual history
        compressed_visual_feature_vector = self.compress_vision(visual_feature_vector)

        return trajectory, compressed_visual_feature_vector
