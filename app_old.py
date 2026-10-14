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
import torch
from diffusers import (
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    StableDiffusionPipeline,
)

from export_model import SAVE_PATH as MODEL_PATH, missing_files

# ------------------------------------------------------------------
# Paths & device
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
if DEVICE == "cpu":
    torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))

NEGATIVE_PROMPT = (
    "low quality, blurry, pixelated, distorted, deformed, bad anatomy, "
    "extra limbs, watermark, text, logo, cropped, out of frame"
)
STYLE_SUFFIX = (
    ", fashion design, full outfit, clothing on mannequin, studio lighting, "
    "highly detailed fabric, intricate details, professional fashion photography"
)

SCHEDULERS = {
    "DPM++ (recommended)": DPMSolverMultistepScheduler,
    "Euler Ancestral": EulerAncestralDiscreteScheduler,
}

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(page_title="ClothAI", page_icon="👗", layout="wide")

# ------------------------------------------------------------------
# Aesthetic background + styling
# ------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Poppins:wght@300;400;500&display=swap');

    html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }

    /* Soft pastel gradient background */
    .stApp {
        background:
            radial-gradient(circle at 12% 18%, rgba(255, 183, 213, 0.55) 0, transparent 40%),
            radial-gradient(circle at 88% 12%, rgba(196, 181, 253, 0.55) 0, transparent 42%),
            radial-gradient(circle at 80% 85%, rgba(253, 230, 138, 0.45) 0, transparent 40%),
            radial-gradient(circle at 15% 88%, rgba(147, 197, 253, 0.50) 0, transparent 40%),
            linear-gradient(135deg, #fdf2f8 0%, #f5f3ff 50%, #fffbeb 100%);
        background-attachment: fixed;
    }

    /* Subtle fabric-weave texture over the gradient */
    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        opacity: 0.35;
        background-image:
            repeating-linear-gradient(45deg, rgba(255,255,255,0.35) 0 2px, transparent 2px 8px),
            repeating-linear-gradient(-45deg, rgba(255,255,255,0.25) 0 2px, transparent 2px 8px);
        z-index: 0;
    }

    header[data-testid="stHeader"] { background: transparent; }

    .title {
        text-align: center;
        font-family: 'Playfair Display', serif;
        font-size: 3rem;
        background: linear-gradient(90deg, #9d174d, #6d28d9, #b45309);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle {
        text-align: center;
        color: #6b5b7b;
        font-weight: 300;
        letter-spacing: 1px;
        margin-bottom: 1.5rem;
    }

    /* Glass cards */
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        background: rgba(255, 255, 255, 0.55);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(255, 255, 255, 0.8);
        border-radius: 22px;
        padding: 1.4rem 1.6rem;
        box-shadow: 0 10px 35px rgba(109, 40, 217, 0.12);
    }

    section[data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.55);
        backdrop-filter: blur(16px);
        border-right: 1px solid rgba(255, 255, 255, 0.8);
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #ec4899, #8b5cf6);
        border: none;
        border-radius: 14px;
        font-weight: 500;
        padding: 0.7rem 1rem;
        box-shadow: 0 6px 18px rgba(139, 92, 246, 0.35);
        transition: transform .15s ease, box-shadow .15s ease;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 24px rgba(139, 92, 246, 0.45);
    }
    .stButton > button[kind="secondary"] {
        border-radius: 12px;
        background: rgba(255,255,255,0.7);
        border: 1px solid rgba(139, 92, 246, 0.25);
    }
    .stImage img { border-radius: 16px; box-shadow: 0 8px 28px rgba(0,0,0,0.15); }
    textarea { border-radius: 14px !important; }
    .footer { text-align:center; color:#7c6a8c; font-size:0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Model loading
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading AI model... (first time takes a while)")
def load_model():
    model_dir = Path(MODEL_PATH).expanduser().resolve()
    missing = missing_files()
    if missing:
        raise FileNotFoundError(
            f"Missing model files in {model_dir}: {', '.join(missing)}"
        )

    pipe = StableDiffusionPipeline.from_pretrained(
        str(model_dir),
        torch_dtype=DTYPE,
        safety_checker=None,
        requires_safety_checker=False,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    pipe = pipe.to(DEVICE)
    pipe.enable_attention_slicing()
    return pipe


def generate_cloth(prompt, steps, guidance_scale, size, scheduler_name, progress_cb=None):
    pipe = load_model()
    pipe.scheduler = SCHEDULERS[scheduler_name].from_config(pipe.scheduler.config)

    def on_step_end(_pipe, step, _timestep, kwargs):
        if progress_cb:
            progress_cb((step + 1) / steps)
        return kwargs

    with torch.inference_mode():
        image = pipe(
            prompt=prompt.strip() + STYLE_SUFFIX,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            height=size,
            width=size,
            callback_on_step_end=on_step_end,
        ).images[0]

    path = os.path.join(OUTPUT_DIR, f"cloth_{int(time.time())}.png")
    image.save(path)
    return image, path


# ------------------------------------------------------------------
# Session state
# ------------------------------------------------------------------
st.session_state.setdefault("image", None)
st.session_state.setdefault("prompt", "")


def set_example(text):
    st.session_state.prompt = text


# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.markdown('<h1 class="title">👗 ClothAI - AI Fashion Designer</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Generate unique clothing designs using AI</p>', unsafe_allow_html=True)

# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    steps = st.slider("Inference Steps", 10, 50, 25, 1,
                      help="25+ gives clean results. Below ~15 the image can look like noise.")
    guidance_scale = st.slider("Guidance Scale", 1.0, 15.0, 7.5, 0.5)
    size = st.select_slider("Image Size", options=[384, 448, 512], value=512,
                            help="Smaller is faster on CPU.")
    scheduler_name = st.selectbox("Sampler", list(SCHEDULERS.keys()))

    st.caption(f"Running on: **{DEVICE.upper()}**")
    st.markdown("---")
    st.markdown("### 💡 Example Prompts")

    examples = [
        "Elegant evening gown with gold embroidery",
        "Casual summer dress with floral print",
        "Traditional Indian lehenga in royal blue with zari work",
    ]
    for i, ex in enumerate(examples):
        st.button(ex, key=f"ex{i}", use_container_width=True,
                  on_click=set_example, args=(ex,))

# ------------------------------------------------------------------
# Main area
# ------------------------------------------------------------------
col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("Describe your clothing design")
    prompt = st.text_area(
        "Prompt",
        key="prompt",
        height=120,
        placeholder="Example: Traditional Indian lehenga in royal blue with gold embroidery",
        label_visibility="collapsed",
    )

    if st.button("✨ Generate Design", type="primary", use_container_width=True):
        if not prompt.strip():
            st.error("Please enter a clothing description.")
        else:
            try:
                bar = st.progress(0.0, text="✨ Generating your design...")
                image, _ = generate_cloth(
                    prompt, steps, guidance_scale, size, scheduler_name,
                    progress_cb=lambda p: bar.progress(min(p, 1.0), text="✨ Generating your design..."),
                )
                bar.empty()
                st.session_state.image = image
                st.success("✅ Design generated successfully!")
            except Exception as e:
                st.error(f"Generation error: {e}")

with col2:
    st.subheader("Your Design")
    if st.session_state.image is not None:
        st.image(st.session_state.image, use_container_width=True)
        buf = io.BytesIO()
        st.session_state.image.save(buf, format="PNG")
        st.download_button("⬇️ Download Design", data=buf.getvalue(),
                           file_name="cloth_design.png", mime="image/png",
                           use_container_width=True)
    else:
        st.info("Your generated clothing design will appear here.")

# ------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------
st.markdown("---")
st.markdown(
    '<div class="footer">✨ Tip: Add fabric, color, embroidery, style and lighting '
    'details to get better designs.</div>',
    unsafe_allow_html=True,
)