# 👗 ClothAI - AI Fashion Designer

An AI-powered clothing design generator that creates unique fashion designs based on text prompts using Stable Diffusion.

## ✨ Features

- Generate clothing designs from text descriptions
- Customize generation parameters (steps, guidance scale)
- Example prompts for quick testing
- Download generated designs
- Modern, responsive web interface
- Runs locally on your machine

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git (for cloning the repository)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ClothAI.git
   cd ClothAI
   ```

2. **Create and activate a virtual environment**
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   
   Or install them manually:
   ```bash
   pip install streamlit torch diffusers transformers
   ```

4. **Add the local model files**
   - Place your local model files in the `models/model_assets` directory
   - The directory structure should include (at minimum):
     ```
     models/
     └── model_assets/
         ├── model_index.json
         ├── scheduler/
         ├── text_encoder/
         ├── unet/
         └── vae/
     ```

## 🖥️ Running the Application

1. **Activate the virtual environment** (if not already activated):
   ```bash
   # Windows
   .\venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

2. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

3. **Open your web browser** to the URL shown in the terminal (usually http://localhost:8501)

## 🛠️ Usage

1. **Enter a prompt** describing the clothing design you want to generate
2. **Adjust settings** in the sidebar (optional):
   - **Inference Steps**: More steps = better quality but slower generation (default: 22)
   - **Guidance Scale**: Higher values = more closely follow the prompt (default: 7.5)
3. **Click "Generate Design"** to create your design
4. **Download** your favorite designs using the download button

### Example Prompts
- "Elegant evening gown with gold embroidery, flowing silk, fashion show lighting"
- "Casual summer dress with floral print, natural linen, soft daylight"
- "Traditional Indian lehenga in royal blue with intricate zari work"

## 📁 Project Structure

```
ClothAI/
├── app.py                # Main application file
├── requirements.txt      # Project dependencies
├── models/               # AI/ML models
│   └── model_assets/      # Local model files
├── outputs/              # Generated designs
└── venv/                 # Virtual environment (ignored in version control)
```

## 📦 Dependencies

- [Streamlit](https://streamlit.io/) - Web application framework
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Transformers](https://huggingface.co/docs/transformers/index) - For text processing
- [Pillow](https://pillow.readthedocs.io/) - Image processing

## 🚧 Troubleshooting

- **CUDA not available**: The app will automatically fall back to CPU if CUDA is not available, but it will be slower.
- **Memory issues**: Try reducing the image size or using fewer inference steps.
- **Model not found**: Make sure your local model files are present in the `models/model_assets/` directory.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Hugging Face Diffusers](https://huggingface.co/docs/diffusers/index)
- [Streamlit](https://streamlit.io/) for the web interface