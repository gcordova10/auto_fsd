import torch
import torch.nn as nn
from .backbone import Backbone
from .feature_fusion import FeatureFusion
from .driving_policy import DrivingPolicy
from .future_state import FutureState
from .causal_reasoning import CausalReasoningModule


class AutoFSD(nn.Module):
    def __init__(self):
        super(AutoFSD, self).__init__()

        # Backbone feature extractor
        self.Backbone = Backbone()

        # Multi-scale feature fusion
        self.FeatureFusion = FeatureFusion()

        # Driving policy prediction
        self.DrivingPolicy = DrivingPolicy()

        # Future visual state prediction
        self.FutureState = FutureState()

        # Causal reasoning (System 2)
        self.CausalReasoning = CausalReasoningModule()


    def forward(self, image, visual_history, egomotion_history):

        features = self.Backbone(image)
        fused_features = self.FeatureFusion(features)

        # Ensure history tensors have batch dimension (simulated as 1 for now)
        if visual_history.dim() == 1:
            visual_history = visual_history.unsqueeze(0)
        if egomotion_history.dim() == 1:
            egomotion_history = egomotion_history.unsqueeze(0)

        # Driving Policy with Temporal Memory (GRU)
        # It handles the tile flattening internally
        driving_policy, compressed_visual_feature_vector = \
            self.DrivingPolicy(fused_features, visual_history[0], egomotion_history[0])

        # Prepare feature vector for Reasoning (System 2)
        # Using the same flattened visual features as DrivingPolicy
        feature_map = self.DrivingPolicy.reduce_channels(fused_features)
        visual_feature_vector = torch.flatten(feature_map, start_dim=1) # (8, 147)

        # Expand histories to match visual tiles (8)
        v_hist_exp = visual_history.expand(visual_feature_vector.size(0), -1)
        e_hist_exp = egomotion_history.expand(visual_feature_vector.size(0), -1)

        feature_vector_reasoning = torch.cat((visual_feature_vector, 
                                             v_hist_exp, e_hist_exp), dim=1)

        # System 2 Reasoning
        reasoning_latent, decision_logits, text_logits = self.CausalReasoning(feature_vector_reasoning)

        # Future Visual State prediction conditioned on trajectory
        # Expand driving_policy to match fused_features batch size (usually 1, but fused_features has tiles)
        # However, FutureState expects (B, 1440, 7, 7). fused_features is (B, 1440, 7, 7).
        # We use the raw trajectory from DrivingPolicy (before tile expansion if any)
        
        # Ensure driving_policy has a batch dimension for FutureState
        traj_for_future = driving_policy.unsqueeze(0) if driving_policy.dim() == 1 else driving_policy
        future_visual_features = self.FutureState(fused_features, traj_for_future)

        # Ensure the trajectory is expanded to match the batch size of reasoning (8)
        if driving_policy.dim() == 1:
            driving_policy = driving_policy.unsqueeze(0).expand(visual_feature_vector.size(0), -1)

        return {
            "trajectory": driving_policy,
            "visual_context": compressed_visual_feature_vector,
            "future_vision": future_visual_features,
            "reasoning_latent": reasoning_latent,
            "decision_logits": decision_logits,
            "text_logits": text_logits
        }