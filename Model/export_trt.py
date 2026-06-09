import torch
import torch.onnx
import sys
import os

# Add model path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'auto_fsd/Model')))
from model_components.auto_fsd import AutoFSD

def export_autofsd_to_onnx():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoFSD().to(device)
    model.eval()

    # Create dummy inputs
    visual_tiles = torch.randn(8, 3, 224, 224).to(device)
    visual_history = torch.randn(1, 896).to(device)
    egomotion_history = torch.randn(1, 256).to(device)

    # Output path
    export_path = "auto_fsd/Model/autofsd_optimized.onnx"
    
    print(f"Exporting AutoFSD to ONNX at {export_path}...")

    # Export the model with static shapes for better TRT performance
    torch.onnx.export(
        model,
        (visual_tiles, visual_history, egomotion_history),
        export_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['visual_tiles', 'visual_history', 'egomotion_history'],
        output_names=['trajectory', 'decision_logits', 'text_logits', 'future_vision']
    )
    
    if os.path.exists(export_path):
        print(f"Success! ONNX model saved at {export_path}")
        print(f"File size: {os.path.getsize(export_path) / 1e6:.2f} MB")
    else:
        print("Export failed.")

if __name__ == "__main__":
    export_autofsd_to_onnx()
