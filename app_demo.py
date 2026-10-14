"""Light 'demo mode' of ClothAI for Streamlit Community Cloud (no AI model needed).
Shows pre-generated designs from the samples/ folder in the rotating viewer.
Live generation: run app.py locally (or on a GPU machine)."""
import io
import json
import os

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from viewer3d import build_html

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.path.join(BASE_DIR, "samples")
EXT = (".png", ".jpg", ".jpeg", ".webp")

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
.footer { text-align:center; color:#7c6a8c; font-size:.9rem; }
</style>""", unsafe_allow_html=True)


def load_samples():
    if not os.path.isdir(SAMPLES):
        return [], {}
    files = sorted(f for f in os.listdir(SAMPLES) if f.lower().endswith(EXT))
    try:
        with open(os.path.join(SAMPLES, "prompts.json"), encoding="utf-8") as f:
            prompts = json.load(f)
    except Exception:
        prompts = {}
    return files, prompts


files, prompts = load_samples()

st.markdown('<h1 class="title">👗 ClothAI - AI Fashion Designer</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Generate unique clothing designs using AI</p>', unsafe_allow_html=True)

with st.sidebar:
    st.header("ℹ️ Demo mode")
    st.write("This online version shows designs that were **pre-generated** with Stable Diffusion "
             "on a local laptop. The AI model is too large to run on free hosting.")
    st.write("To generate new designs, run the full app locally with `streamlit run app.py`.")

col1, col2 = st.columns(2, gap="large")
with col1:
    st.subheader("Choose a sample design")
    if not files:
        st.warning("No sample images found. Add images to the `samples/` folder and redeploy.")
    else:
        choice = st.selectbox("Sample", files, format_func=lambda f: prompts.get(f, os.path.splitext(f)[0].replace("_", " ")),
                              label_visibility="collapsed")
        st.text_area("Prompt used", value=prompts.get(choice, os.path.splitext(choice)[0].replace("_", " ")),
                     height=120, disabled=True)
        st.caption("Generated locally with a Stable Diffusion model (about 20 steps, DPM++ sampler, 384 px).")

with col2:
    st.subheader("Your Design")
    if files:
        img = Image.open(os.path.join(SAMPLES, choice))
        components.html(build_html(img), height=640)
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="PNG")
        st.download_button("⬇️ Download Design", buf.getvalue(), "cloth_design.png", "image/png",
                           use_container_width=True)
    else:
        st.info("Your design will appear here.")

st.markdown("---")
st.markdown('<div class="footer">✨ ClothAI - final year project demo</div>', unsafe_allow_html=True)