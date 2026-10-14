"""Run once:  python convert_openvino.py
Converts your Stable Diffusion model (models/model_assets) to OpenVINO format (models/openvino_model)."""
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BASE, "models", "model_assets")
DST = os.path.join(BASE, "models", "openvino_model")

if not os.path.isfile(os.path.join(SRC, "model_index.json")):
    sys.exit(f"Model not found in: {SRC}\nPut this file in your ClothAI folder (next to app.py).")

try:
    from optimum.intel import OVStableDiffusionPipeline
except ImportError:
    sys.exit('OpenVINO is not installed. Run:  pip install --upgrade-strategy eager "optimum[openvino]"')

print("Converting model... this can take a few minutes, please wait.")
pipe = OVStableDiffusionPipeline.from_pretrained(SRC, export=True, compile=False)
pipe.save_pretrained(DST)
print("Done! Converted model saved in:", DST)
