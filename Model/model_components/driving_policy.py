import torch
import torch.nn as nn

class DrivingPolicy(nn.Module):
    def __init__(self):
        super(DrivingPolicy, self).__init__()

        # 2D Conv layer to reduce channels
        self.reduce_channels = nn.Conv2d(1440, 3, 3, 1, 1)

        # Temporal Memory (GRU)
        # Input: 14 (compressed visual) + 4 (egomotion) = 18 features per step
        self.temporal_gru = nn.GRU(input_size=18, hidden_size=512, batch_first=True)

        # Linear layers to process current features + temporal memory
        # 1176 (current vision) + 512 (GRU hidden state) = 1688
        self.fc1 = nn.Linear(1688, 1688)
        self.fc2 = nn.Linear(1688, 844)
        self.fc3 = nn.Linear(844, 128)

        # Visual history compression layer
        self.compress_vision = nn.Linear(1176, 14)

        # Dropout
        self.dropout = nn.Dropout(0.25)

        # Activation
        self.activation = nn.GELU()
 
    def forward(self, fused_features, visual_history, egomotion_history):
        # 1. Process Current Vision
        # fused_features shape: (8, 1440, 7, 7) assuming batch 1
        feature_map = self.reduce_channels(fused_features)
        # Flatten and keep tiles in batch dimension or combine them
        current_vision = torch.flatten(feature_map, start_dim=1) # (8, 147)
        current_vision_vector = torch.flatten(current_vision) # (1176)
        
        # 2. Process Temporal Memory
        # visual_history (896) -> (1, 64, 14)
        # egomotion_history (256) -> (1, 64, 4)
        v_hist = visual_history.view(1, 64, 14)
        e_hist = egomotion_history.view(1, 64, 4)
        
        # Combine visual and egomotion sequences
        temporal_sequence = torch.cat((v_hist, e_hist), dim=2) # (1, 64, 18)
        
        # Forward pass through GRU
        # we only need the last hidden state
        _, h_n = self.temporal_gru(temporal_sequence)
        temporal_context = h_n.squeeze(0).squeeze(0) # (512)
        
        # 3. Concatenate Current and Temporal
        feature_vector = torch.cat((current_vision_vector, temporal_context), dim=0)
        
        # Multi-layer perceptron
        f1 = self.fc1(feature_vector)
        f1 = self.activation(f1)
        f1 = self.dropout(f1)

        f2 = self.fc2(f1)
        f2 = self.activation(f2)
        f2 = self.dropout(f2)

        # Trajectory output
        trajectory = self.fc3(f2)

        # Compressed visual feature vector for future history
        compressed_visual_feature_vector = self.compress_vision(current_vision_vector)

        return trajectory, compressed_visual_feature_vector