"""Checks that the local Stable Diffusion model files exist."""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_PATH = os.path.join(BASE_DIR, "models", "model_assets")

REQUIRED_FILES = [
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/model.safetensors",
    "unet/diffusion_pytorch_model.safetensors",
    "vae/diffusion_pytorch_model.safetensors",
]


def missing_files():
    return [f for f in REQUIRED_FILES
            if not os.path.exists(os.path.join(SAVE_PATH, f))]


if __name__ == "__main__":
    missing = missing_files()
    if missing:
        print("Missing model files:")
        for f in missing:
            print("  -", f)
    else:
        print("All model files found in", SAVE_PATH)