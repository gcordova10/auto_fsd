import onnxruntime as ort
import numpy as np
import torch
import time

def benchmark_onnx():
    onnx_path = "auto_fsd/Model/autofsd_optimized.onnx"
    if not os.path.exists(onnx_path):
        print("ONNX model not found.")
        return

    # Initialize ORT session
    # Try GPU first, then CPU
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    try:
        session = ort.InferenceSession(onnx_path, providers=providers)
        print(f"Session initialized with providers: {session.get_providers()}")
    except Exception as e:
        print(f"Error initializing session: {e}")
        return

    # Prepare dummy inputs
    visual_tiles = np.random.randn(8, 3, 224, 224).astype(np.float32)
    visual_history = np.random.randn(1, 896).astype(np.float32)
    egomotion_history = np.random.randn(1, 256).astype(np.float32)

    input_dict = {
        'visual_tiles': visual_tiles,
        'visual_history': visual_history,
        'egomotion_history': egomotion_history
    }

    # Warmup
    print("Warmup...")
    for _ in range(10):
        _ = session.run(None, input_dict)

    # Benchmark
    print("Benchmarking...")
    num_iters = 50
    start_time = time.time()
    for _ in range(num_iters):
        _ = session.run(None, input_dict)
    end_time = time.time()

    avg_latency = (end_time - start_time) / num_iters
    fps = 1.0 / avg_latency

    print(f"Average Latency: {avg_latency*1000:.2f} ms")
    print(f"Throughput: {fps:.2f} FPS")

if __name__ == "__main__":
    import os
    benchmark_onnx()
