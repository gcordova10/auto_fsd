import torch
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # --- GPU & Hardware ---
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Adjust based on VRAM: 6GB -> 1, 12GB -> 4, 24GB+ -> 8
    GROUP_SIZE = int(os.getenv("GROUP_SIZE", 1)) 
    # Action condition channels (128 for small GPUs, 1440 for large GPUs)
    ACTION_CONDITION_CHANNELS = int(os.getenv("ACTION_CONDITION_CHANNELS", 128))
    
    # --- Dataset ---
    HF_TOKEN = os.getenv("HF_TOKEN")
    REPO_ID = "nvidia/PhysicalAI-Autonomous-Vehicles"
    SNAPSHOT_PATH = os.path.join(os.path.expanduser("~"), ".cache/huggingface/hub/datasets--nvidia--PhysicalAI-Autonomous-Vehicles/snapshots/b719eea7f0a63619ef51ec7f54178af0937ef050")
    
    # Ingestion Configuration
    TARGET_COUNTRY = os.getenv("TARGET_COUNTRY", "Spain")
    LIMIT_CLIPS = int(os.getenv("LIMIT_CLIPS", 100))
    
    # --- AutoE2E Model ---
    IMAGE_SIZE = (256, 256)
    LATENT_CHANNELS = 1440
    GRID_SIZE = 8
    
    # --- GRPO Training ---
    LEARNING_RATE = 1e-5
    LAMBDA_SMOOTH = 0.1
    LAMBDA_CAUSAL = 0.1
    LAMBDA_JEPA = 0.5
    LAMBDA_SAFETY = 0.3
    KL_COEFF = 0.01

    @classmethod
    def print_config(cls):
        print("--- CURRENT AUTOFSD CONFIGURATION ---")
        print(f"Device: {cls.DEVICE}")
        print(f"Group Size (GRPO): {cls.GROUP_SIZE}")
        print(f"Action Channels: {cls.ACTION_CONDITION_CHANNELS}")
        print(f"Target Country: {cls.TARGET_COUNTRY}")
        print(f"Dataset Path: {cls.SNAPSHOT_PATH}")
        print("---------------------------------------")
