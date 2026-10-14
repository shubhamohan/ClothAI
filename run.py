"""Launcher: python run.py"""
import subprocess
import sys
from pathlib import Path

base = Path(__file__).parent.resolve()
print("Starting ClothAI...")
try:
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(base / "app.py")], check=True)
except KeyboardInterrupt:
    print("\nClothAI stopped.")