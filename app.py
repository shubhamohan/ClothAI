import io
import logging
import os
import time
import warnings
from pathlib import Path

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("diffusers").setLevel(logging.ERROR)

import streamlit as st
import streamlit.components.v1 as components
import torch
from diffusers import (DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler,
                       StableDiffusionPipeline)

from export_model import SAVE_PATH as MODEL_PATH, missing_files
from viewer3d import build_html

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
OV_DIR = os.path.join(BASE_DIR, "models", "openvino_model")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
if DEVICE == "cpu":
    torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))

NEGATIVE_PROMPT = ("low quality, blurry, pixelated, distorted, deformed, bad anatomy, "
                   "extra limbs, watermark, text, logo, cropped, out of frame")
STYLE_SUFFIX = (", fashion design, full outfit, clothing on mannequin, studio lighting, "
                "highly detailed fabric, intricate details, professional fashion photography")
SCHEDULERS = {"DPM++ (recommended)": DPMSolverMultistepScheduler,
              "Euler Ancestral": EulerAncestralDiscreteScheduler}
ENGINE_PT, ENGINE_OV = "PyTorch (original)", "OpenVINO (faster)"

st.set_page_config(page_title="ClothAI", page_icon="👗", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Poppins:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
.stApp {
  background:
    radial-gradient(circle at 12% 18%, rgba(255,183,213,.55) 0, transparent 40%),
    radial-gradient(circle at 88% 12%, rgba(196,181,253,.55) 0, transparent 42%),
    radial-gradient(circle at 80% 85%, rgba(253,230,138,.45) 0, transparent 40%),
    radial-gradient(circle at 15% 88%, rgba(147,197,253,.50) 0, transparent 40%),
    linear-gradient(135deg, #fdf2f8 0%, #f5f3ff 50%, #fffbeb 100%);
  background-attachment: fixed; }
header[data-testid="stHeader"] { background: transparent; }
.title { text-align:center; font-family:'Playfair Display',serif; font-size:3rem;
  background:linear-gradient(90deg,#9d174d,#6d28d9,#b45309); -webkit-background-clip:text;
  -webkit-text-fill-color:transparent; margin-bottom:0; }
.subtitle { text-align:center; color:#6b5b7b; font-weight:300; letter-spacing:1px; margin-bottom:1.5rem; }
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { background:rgba(255,255,255,.55);
  backdrop-filter:blur(14px); border:1px solid rgba(255,255,255,.8); border-radius:22px;
  padding:1.4rem 1.6rem; box-shadow:0 10px 35px rgba(109,40,217,.12); }
section[data-testid="stSidebar"] { background:rgba(255,255,255,.55); backdrop-filter:blur(16px); }
.stButton > button[kind="primary"] { background:linear-gradient(90deg,#ec4899,#8b5cf6); border:none;
  border-radius:14px; font-weight:500; padding:.7rem 1rem; box-shadow:0 6px 18px rgba(139,92,246,.35); }
.stImage img { border-radius:16px; box-shadow:0 8px 28px rgba(0,0,0,.15); }
.footer { text-align:center; color:#7c6a8c; font-size:.9rem; }
</style>""", unsafe_allow_html=True)

@st.cache_resource
def ov_available():
    if not os.path.isfile(os.path.join(OV_DIR, "model_index.json")):
        return False
    try:
        from optimum.intel import OVStableDiffusionPipeline  # noqa: F401
        return True
    except Exception:
        return False

@st.cache_resource(show_spinner="Loading PyTorch model...")
def load_pytorch():
    missing = missing_files()
    if missing:
        raise FileNotFoundError(f"Missing model files: {', '.join(missing)}")
    pipe = StableDiffusionPipeline.from_pretrained(
        str(Path(MODEL_PATH).resolve()), torch_dtype=DTYPE, safety_checker=None,
        requires_safety_checker=False, local_files_only=True, low_cpu_mem_usage=True).to(DEVICE)
    pipe.enable_attention_slicing()
    return pipe


@st.cache_resource(show_spinner="Loading OpenVINO model (first run compiles it for your CPU)...")
def load_openvino(size):
    from optimum.intel import OVStableDiffusionPipeline
    pipe = OVStableDiffusionPipeline.from_pretrained(OV_DIR, compile=False)
    pipe.reshape(batch_size=1, height=size, width=size, num_images_per_prompt=1)  # fixed size = faster
    pipe.compile()
    return pipe


def generate_cloth(prompt, steps, guidance, size, sampler, engine, progress_cb=None):
    pipe = load_openvino(size) if engine == ENGINE_OV else load_pytorch()
    try:
        pipe.scheduler = SCHEDULERS[sampler].from_config(pipe.scheduler.config)
    except Exception:
        pass

    def on_step(_p, step, _t, kwargs):
        if progress_cb:
            progress_cb((step + 1) / steps)
        return kwargs

    args = dict(prompt=prompt.strip() + STYLE_SUFFIX, negative_prompt=NEGATIVE_PROMPT,
                num_inference_steps=steps, guidance_scale=guidance, height=size, width=size,
                callback_on_step_end=on_step)
    t0 = time.perf_counter()
    try:
        out = pipe(**args)
    except TypeError:                      # some pipelines don't support the progress callback
        args.pop("callback_on_step_end")
        out = pipe(**args)
    seconds = time.perf_counter() - t0
    image = out.images[0]
    image.save(os.path.join(OUTPUT_DIR, f"cloth_{int(time.time())}.png"))
    return image, seconds


ss = st.session_state
ss.setdefault("image", None)
ss.setdefault("prompt", "")
ss.setdefault("timings", [])


def set_example(text):
    ss.prompt = text


st.markdown('<h1 class="title">👗 ClothAI - AI Fashion Designer</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Generate unique clothing designs using AI</p>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Settings")
    engines = [ENGINE_OV, ENGINE_PT] if ov_available() else [ENGINE_PT]
    engine = st.radio("Engine", engines)
    if not ov_available():
        st.caption("OpenVINO model not found. Run `python convert_openvino.py` first.")
    steps = st.slider("Inference Steps", 10, 50, 20, 1, help="Below ~15 steps images look noisy.")
    guidance = st.slider("Guidance Scale", 1.0, 15.0, 7.5, 0.5)
    size = st.select_slider("Image Size", options=[384, 448, 512], value=384)
    sampler = st.selectbox("Sampler", list(SCHEDULERS))
    st.markdown("---")
    st.markdown("### 💡 Example Prompts")
    for i, ex in enumerate(["Elegant evening gown with gold embroidery",
                            "Casual summer dress with floral print",
                            "Traditional Indian lehenga in royal blue with zari work"]):
        st.button(ex, key=f"ex{i}", use_container_width=True, on_click=set_example, args=(ex,))

col1, col2 = st.columns(2, gap="large")
with col1:
    st.subheader("Describe your clothing design")
    prompt = st.text_area("Prompt", key="prompt", height=120, label_visibility="collapsed",
                          placeholder="Example: Traditional Indian lehenga in royal blue with gold embroidery")
    if st.button("✨ Generate Design", type="primary", use_container_width=True):
        if not prompt.strip():
            st.error("Please enter a clothing description.")
        else:
            try:
                bar = st.progress(0.0, text="✨ Generating your design...")
                img, sec = generate_cloth(prompt, steps, guidance, size, sampler, engine,
                                          lambda p: bar.progress(min(p, 1.0), text="✨ Generating your design..."))
                bar.empty()
                ss.image = img
                ss.timings.insert(0, dict(Engine=engine, Size=size, Steps=steps, Seconds=round(sec, 1)))
                st.success(f"✅ Done in {sec:.1f} s using {engine}")
            except Exception as e:
                st.error(f"Generation error: {e}")
    if ss.timings:
        with st.expander("⏱ Speed log (use for your report)"):
            st.table(ss.timings[:10])

with col2:
    st.subheader("Your Design")
    if ss.image is not None:
        components.html(build_html(ss.image), height=640)
        buf = io.BytesIO()
        ss.image.save(buf, format="PNG")
        st.download_button("⬇️ Download Design", buf.getvalue(), "cloth_design.png", "image/png",
                           use_container_width=True)
    else:
        st.info("Your generated clothing design will appear here.")

st.markdown("---")
st.markdown('<div class="footer">✨ Tip: add fabric, color, embroidery, style and lighting details.</div>',
            unsafe_allow_html=True)