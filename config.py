"""
config.py - Central configuration for the Speech Emotion Detection project.
All paths, hyperparameters, and constants are defined here so nothing is hard-coded
elsewhere in the codebase.
"""

import os
from pathlib import Path

# ─── Project root ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # speech-emotion-detection/

# ─── Data paths ───────────────────────────────────────────────────────────────
DATA_DIR        = PROJECT_ROOT / "data"
RAW_DIR         = DATA_DIR / "raw" / "RAVDESS"
PROCESSED_DIR   = DATA_DIR / "processed"
FEATURES_DIR    = DATA_DIR / "features"

# ─── Model paths ──────────────────────────────────────────────────────────────
MODELS_DIR      = PROJECT_ROOT / "models"
MODEL_PATH      = MODELS_DIR / "speech_emotion_model.pkl"
SCALER_PATH     = MODELS_DIR / "scaler.pkl"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
METADATA_PATH   = MODELS_DIR / "model_metadata.json"
FEATURES_CONFIG_PATH = MODELS_DIR / "features_config.json"

# ─── Results paths ────────────────────────────────────────────────────────────
RESULTS_DIR     = PROJECT_ROOT / "results"
CONFUSION_MATRIX_PATH   = RESULTS_DIR / "confusion_matrix.png"
CLASSIFICATION_REPORT_PATH = RESULTS_DIR / "classification_report.txt"
METRICS_JSON_PATH       = RESULTS_DIR / "metrics.json"
MODEL_COMPARISON_PATH   = RESULTS_DIR / "model_comparison.csv"

# ─── Audio preprocessing settings ─────────────────────────────────────────────
SAMPLE_RATE     = 22050          # Target sample rate (Hz) — librosa default
AUDIO_DURATION  = 3.0            # Fixed duration (seconds) for feature extraction
                                 # RAVDESS clips are ~3-4s; we pad/trim to 3s
N_MFCC          = 40             # Number of MFCC coefficients
N_MELS          = 128            # Mel filter banks
HOP_LENGTH      = 512            # Hop length for STFT
N_FFT           = 2048           # FFT window size
TOP_DB          = 20             # Threshold for silence trimming (dB below peak)

# ─── Feature extraction settings ──────────────────────────────────────────────
FEATURE_CONFIG = {
    "mfcc":             True,    # 40 MFCCs (mean + std = 80 values)
    "mfcc_delta":       True,    # Delta MFCCs (80 values)
    "mel_spectrogram":  False,   # Skipped: high-dim; MFCCs already capture it
    "zero_crossing_rate": True,  # ZCR mean + std (2 values)
    "rms_energy":       True,    # RMS mean + std (2 values)
    "spectral_centroid": True,   # Spectral centroid mean + std (2 values)
    "spectral_rolloff":  True,   # Spectral rolloff mean + std (2 values)
    "chroma":           True,    # Chroma features mean + std (24 values)
}
# Total feature vector size: 80 + 80 + 2 + 2 + 2 + 2 + 24 = 192 features

# ─── Train/Val/Test split ─────────────────────────────────────────────────────
TEST_SIZE       = 0.15   # 15% for final testing
VAL_SIZE        = 0.15   # 15% for validation (from training set)
RANDOM_SEED     = 42     # Fixed seed for reproducibility

# ─── RAVDESS emotion mapping ───────────────────────────────────────────────────
# RAVDESS file naming convention: 3rd segment is emotion code
# 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised
RAVDESS_EMOTION_MAP = {
    "01": "Neutral",
    "02": "Calm",
    "03": "Happy",
    "04": "Sad",
    "05": "Angry",
    "06": "Fearful",
    "07": "Disgust",
    "08": "Surprised",
}

# All 8 supported emotion classes (in consistent order)
EMOTION_CLASSES = [
    "Neutral", "Calm", "Happy", "Sad",
    "Angry", "Fearful", "Disgust", "Surprised"
]

# ─── Backend settings ─────────────────────────────────────────────────────────
BACKEND_HOST    = "127.0.0.1"
BACKEND_PORT    = 8000
MIN_AUDIO_DURATION = 0.5   # Minimum acceptable recording duration (seconds)
MAX_AUDIO_SIZE_MB  = 10    # Maximum upload file size (MB)

# ─── Ensure output directories exist ─────────────────────────────────────────
for _dir in [PROCESSED_DIR, FEATURES_DIR, MODELS_DIR, RESULTS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)
