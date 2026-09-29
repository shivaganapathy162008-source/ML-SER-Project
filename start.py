"""
start.py - Cross-platform launcher for Speech Emotion Detection System.

Usage:
    python start.py setup      # Install dependencies + check dataset
    python start.py dataset    # Check/setup dataset
    python start.py train      # Train ML model
    python start.py backend    # Start FastAPI backend
    python start.py frontend   # Open frontend in browser
    python start.py test       # Run tests
    python start.py all        # Train (if needed) + backend + frontend
"""

import sys
import os
import subprocess
import webbrowser
import time
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR      = PROJECT_ROOT / "src"
BACKEND_DIR  = PROJECT_ROOT / "backend"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
MODEL_PATH   = PROJECT_ROOT / "models" / "speech_emotion_model.pkl"

PYTHON = sys.executable


def run(cmd: list, **kwargs):
    """Run a subprocess command."""
    print(f"\n> {' '.join(cmd)}\n")
    return subprocess.run(cmd, **kwargs)


def cmd_setup():
    """Install all dependencies and check dataset."""
    print("=" * 60)
    print("  STEP 1: Installing dependencies")
    print("=" * 60)
    run([PYTHON, "-m", "pip", "install", "-r", str(PROJECT_ROOT / "requirements.txt")])

    print("\n" + "=" * 60)
    print("  STEP 2: Checking dataset")
    print("=" * 60)
    run([PYTHON, str(SRC_DIR / "download_dataset.py")])


def cmd_dataset():
    """Check dataset status."""
    run([PYTHON, str(SRC_DIR / "download_dataset.py")])


def cmd_train():
    """Run the training pipeline."""
    print("=" * 60)
    print("  Training ML Model — This may take 10-30 minutes")
    print("=" * 60)
    result = run([PYTHON, str(SRC_DIR / "train.py")])
    if result.returncode != 0:
        print("\nERROR: Training failed. Check the output above.")
        sys.exit(1)
    print("\n✅ Training complete! You can now start the backend.")


def cmd_backend():
    """Start the FastAPI backend server."""
    print("=" * 60)
    print("  Starting FastAPI Backend")
    print("  URL: http://127.0.0.1:8000")
    print("  API docs: http://127.0.0.1:8000/docs")
    print("  Press Ctrl+C to stop")
    print("=" * 60)
    os.environ["PYTHONPATH"] = str(SRC_DIR)
    run([PYTHON, str(BACKEND_DIR / "main.py")])


def cmd_frontend():
    """Open the frontend in the default browser."""
    url = "http://127.0.0.1:8000"
    print(f"Opening: {url}")
    webbrowser.open(url)
    print("\nNOTE: Make sure the backend is running!")
    print("      Run in another terminal: python start.py backend")


def cmd_test():
    """Run the test suite."""
    print("=" * 60)
    print("  Running Test Suite")
    print("=" * 60)
    result = run([
        PYTHON, "-m", "pytest",
        str(PROJECT_ROOT / "tests"),
        "-v", "--tb=short"
    ])
    sys.exit(result.returncode)


def cmd_all():
    """Full pipeline: train if needed, start backend, open frontend."""
    print("=" * 60)
    print("  Starting Complete Speech Emotion Detection System")
    print("=" * 60)

    # Train if model doesn't exist
    if not MODEL_PATH.exists():
        print("\nNo trained model found. Starting training...")
        cmd_train()

    # Start backend in background
    print("\nStarting backend server in background...")
    backend_process = subprocess.Popen(
        [PYTHON, str(BACKEND_DIR / "main.py")],
        env={**os.environ, "PYTHONPATH": str(SRC_DIR)}
    )

    # Wait for backend to initialize
    print("Waiting for backend to start (5 seconds)...")
    time.sleep(5)

    # Open frontend
    cmd_frontend()

    print("\n" + "=" * 60)
    print("  System Running!")
    print("  Frontend: http://127.0.0.1:8000")
    print("  Backend:  http://127.0.0.1:8000")
    print("  API docs: http://127.0.0.1:8000/docs")
    print("=" * 60)

    try:
        print("\nPress Ctrl+C to stop the backend...")
        backend_process.wait()
    except KeyboardInterrupt:
        print("\nStopping backend...")
        backend_process.terminate()


COMMANDS = {
    "setup":    cmd_setup,
    "dataset":  cmd_dataset,
    "train":    cmd_train,
    "backend":  cmd_backend,
    "frontend": cmd_frontend,
    "test":     cmd_test,
    "all":      cmd_all,
}


def print_help():
    print("""
🎙️  Speech Emotion Detection — Launcher

Usage:
    python start.py <command>

Commands:
    setup      Install dependencies + check dataset
    dataset    Check/setup RAVDESS dataset
    train      Train the ML model (run once)
    backend    Start FastAPI backend (keep running)
    frontend   Open web UI in browser
    test       Run test suite
    all        Full system (train if needed + backend + frontend)

Quick Start:
    1. python start.py setup
    2. python start.py train
    3. python start.py backend   ← keep this running
    4. python start.py frontend  ← in a new terminal
""")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_help()
        sys.exit(0)

    command = sys.argv[1].lower()
    if command in COMMANDS:
        COMMANDS[command]()
    elif command in ["-h", "--help", "help"]:
        print_help()
    else:
        print(f"Unknown command: {command}")
        print_help()
        sys.exit(1)
