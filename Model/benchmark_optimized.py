import torch
import sys
import os
import time
import tensorrt

# Add model path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'auto_fsd/Model')))
from model_components.auto_fsd import AutoFSD

def benchmark_optimized():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Enable TF32 for better performance on Ampere GPUs (RTX 3060)
    torch.set_float32_matmul_precision('high')
    
    model = AutoFSD().to(device)
    model.eval()

    # Create dummy inputs
    visual_tiles = torch.randn(8, 3, 224, 224).to(device)
    visual_history = torch.randn(1, 896).to(device)
    egomotion_history = torch.randn(1, 256).to(device)

    print("\n--- Torch Compiled (max-autotune) ---")
    # max-autotune can give better results than default
    compiled_model = torch.compile(model, mode="max-autotune")
    
    print("Compiling...")
    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = compiled_model(visual_tiles, visual_history, egomotion_history)

    num_iters = 100
    start_time = time.time()
    for _ in range(num_iters):
        with torch.no_grad():
            _ = compiled_model(visual_tiles, visual_history, egomotion_history)
    end_time = time.time()
    print(f"Latency: {(end_time - start_time)/num_iters*1000:.2f} ms | FPS: {num_iters/(end_time - start_time):.2f}")

    print("\n--- Half Precision (FP16) ---")
    model_fp16 = model.half()
    # Need to convert inputs too
    vt_h = visual_tiles.half()
    vh_h = visual_history.half()
    eh_h = egomotion_history.half()
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model_fp16(vt_h, vh_h, eh_h)

    start_time = time.time()
    for _ in range(num_iters):
        with torch.no_grad():
            _ = model_fp16(vt_h, vh_h, eh_h)
    end_time = time.time()
    print(f"Latency: {(end_time - start_time)/num_iters*1000:.2f} ms | FPS: {num_iters/(end_time - start_time):.2f}")

    print("\n--- Compiled + FP16 ---")
    compiled_fp16 = torch.compile(model_fp16, mode="max-autotune")
    print("Compiling...")
    for _ in range(5):
        with torch.no_grad():
            _ = compiled_fp16(vt_h, vh_h, eh_h)
            
    start_time = time.time()
    for _ in range(num_iters):
        with torch.no_grad():
            _ = compiled_fp16(vt_h, vh_h, eh_h)
    end_time = time.time()
    print(f"Latency: {(end_time - start_time)/num_iters*1000:.2f} ms | FPS: {num_iters/(end_time - start_time):.2f}")

if __name__ == "__main__":
    benchmark_optimized()
