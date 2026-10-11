"""Unified Runner for Text-to-SQL Analytics Assistant.

Launches both FastAPI backend and Streamlit frontend concurrently.
Usage:
    python run.py             # Start both backend and frontend
    python run.py --backend   # Start FastAPI backend only
    python run.py --frontend  # Start Streamlit frontend only
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

# Locate Python executable inside .venv if present
VENV_PYTHON = ROOT_DIR / ".venv" / "Scripts" / "python.exe"
PYTHON_EXE = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable


def start_backend():
    print("🚀 [Backend] Starting FastAPI server on http://127.0.0.1:8000...")
    cmd = [
        PYTHON_EXE,
        "-m",
        "uvicorn",
        "backend.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--reload",
    ]
    return subprocess.Popen(cmd, cwd=str(ROOT_DIR))


def start_frontend():
    print("💻 [Frontend] Starting Streamlit dashboard on http://localhost:8501...")
    frontend_app = str(ROOT_DIR / "frontend" / "app.py")
    cmd = [
        PYTHON_EXE,
        "-m",
        "streamlit",
        "run",
        frontend_app,
        "--server.port=8501",
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
    ]
    return subprocess.Popen(cmd, cwd=str(ROOT_DIR))


def main():
    parser = argparse.ArgumentParser(description="Text-to-SQL Analytics Assistant Runner")
    parser.add_argument("--backend", action="store_true", help="Launch backend only")
    parser.add_argument("--frontend", action="store_true", help="Launch frontend only")
    args = parser.parse_args()

    processes = []
    try:
        if args.backend:
            p_back = start_backend()
            processes.append(p_back)
            p_back.wait()
        elif args.frontend:
            p_front = start_frontend()
            processes.append(p_front)
            p_front.wait()
        else:
            # Start both
            p_back = start_backend()
            processes.append(p_back)
            time.sleep(2)  # Give backend a moment to bind
            p_front = start_frontend()
            processes.append(p_front)

            print("\n" + "=" * 60)
            print("  ⚡ Text-to-SQL Analytics Assistant is running!")
            print("  - Backend API:    http://127.0.0.1:8000")
            print("  - API Docs:       http://127.0.0.1:8000/docs")
            print("  - Streamlit UI:   http://localhost:8501")
            print("=" * 60 + "\n")
            print("Press Ctrl+C to terminate both servers.\n")

            # Wait on either process
            while True:
                time.sleep(1)
                for p in processes:
                    if p.poll() is not None:
                        raise KeyboardInterrupt

    except KeyboardInterrupt:
        print("\n🛑 Shutting down servers...")
        for p in processes:
            p.terminate()
            try:
                p.wait(timeout=3)
            except Exception:
                p.kill()
        print("Done.")


if __name__ == "__main__":
    main()
