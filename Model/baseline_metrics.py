import torch
import numpy as np
import sys
import os

# Add parent directory to path to import model components
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'auto_fsd/Model')))
from model_components.auto_fsd import AutoFSD
from nvidia_parser import NvidiaDatasetParser

def calculate_temporal_variance(trajectory):
    """
    Calculates the temporal variance of actions: sigma^2_{delta a}
    trajectory: (batch, 128) -> 64 steps of (accel, curvature)
    """
    # Reshape to (batch, 64, 2)
    traj = trajectory.view(-1, 64, 2)
    # Difference between consecutive steps: (batch, 63, 2)
    diff = traj[:, 1:, :] - traj[:, :-1, :]
    
    # Calculate variance of differences
    # We can take the mean of variances across the batch
    var_accel = torch.var(diff[:, :, 0], dim=1)
    var_curv = torch.var(diff[:, :, 1], dim=1)
    
    return torch.mean(var_accel).item(), torch.mean(var_curv).item()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running baseline on {device}")
    
    model = AutoFSD().to(device)
    model.eval()
    
    parser = NvidiaDatasetParser()
    
    # Generate mock data for evaluation
    # In a real scenario, this would loop through the downloaded dataset
    dummy_cameras = {name: np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8) for name in parser.camera_names}
    dummy_egomotion = np.random.randn(256)
    
    visual_tiles, egomotion_history, visual_history = parser.format_input(dummy_cameras, dummy_egomotion)
    
    # Add batch dimension
    visual_tiles = visual_tiles.to(device)
    egomotion_history = egomotion_history.to(device)
    visual_history = visual_history.to(device)
    
    print("Executing forward pass...")
    with torch.no_grad():
        trajectory, _, _ = model(visual_tiles, visual_history, egomotion_history)
    
    var_a, var_c = calculate_temporal_variance(trajectory)
    
    print("\n--- Baseline Metrics ---")
    print(f"Temporal Variance (Acceleration): {var_a:.6f}")
    print(f"Temporal Variance (Curvature): {var_c:.6f}")
    print("------------------------\n")
    
    # Save results to a file for later comparison
    with open('baseline_results.txt', 'w') as f:
        f.write(f"Acceleration Variance: {var_a}\n")
        f.write(f"Curvature Variance: {var_c}\n")

if __name__ == "__main__":
    main()
