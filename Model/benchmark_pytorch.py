import torch
import sys
import os
import time

# Add model path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'auto_fsd/Model')))
from model_components.auto_fsd import AutoFSD

def benchmark_pytorch():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model = AutoFSD().to(device)
    model.eval()

    # Create dummy inputs
    visual_tiles = torch.randn(8, 3, 224, 224).to(device)
    visual_history = torch.randn(1, 896).to(device)
    egomotion_history = torch.randn(1, 256).to(device)

    print("--- Vanilla PyTorch ---")
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(visual_tiles, visual_history, egomotion_history)

    num_iters = 50
    start_time = time.time()
    for _ in range(num_iters):
        with torch.no_grad():
            _ = model(visual_tiles, visual_history, egomotion_history)
    end_time = time.time()
    print(f"Latency: {(end_time - start_time)/num_iters*1000:.2f} ms | FPS: {num_iters/(end_time - start_time):.2f}")

    print("\n--- Torch Compiled (Default) ---")
    compiled_model = torch.compile(model)
    # Warmup (Compiling happens here)
    print("Compiling (this might take a minute)...")
    for _ in range(5):
        with torch.no_grad():
            _ = compiled_model(visual_tiles, visual_history, egomotion_history)

    start_time = time.time()
    for _ in range(num_iters):
        with torch.no_grad():
            _ = compiled_model(visual_tiles, visual_history, egomotion_history)
    end_time = time.time()
    print(f"Latency: {(end_time - start_time)/num_iters*1000:.2f} ms | FPS: {num_iters/(end_time - start_time):.2f}")

if __name__ == "__main__":
    benchmark_pytorch()
