"""Convenience script to launch the Streamlit NDT Inspection Web App."""

import os
from pathlib import Path
import subprocess
import sys

def main():
    root = Path(__file__).resolve().parent
    app_path = root / "web_app" / "app.py"
    
    # Locate virtual environment python/streamlit if present
    venv_streamlit_win = root / ".venv" / "Scripts" / "streamlit.exe"
    venv_streamlit_unix = root / ".venv" / "bin" / "streamlit"
    
    if venv_streamlit_win.exists():
        cmd = [str(venv_streamlit_win), "run", str(app_path)]
    elif venv_streamlit_unix.exists():
        cmd = [str(venv_streamlit_unix), "run", str(app_path)]
    else:
        cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]

    print(f"[*] Starting AI-Assisted NDT Web Interface...")
    print(f"[*] Command: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
