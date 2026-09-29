"""
main.py - FastAPI Backend for Speech Emotion Detection.

Endpoints:
  GET  /health         — Health check, model status
  POST /predict        — Upload audio, get emotion prediction
  GET  /emotions       — List supported emotion classes
  GET  /model-info     — Model metadata

Run with:
  python backend/main.py
  or
  uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
"""

import sys
import json
import logging
import tempfile
import os
import io
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import numpy as np
import joblib
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# Add src/ to path so we can import our modules
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from config import (
    MODEL_PATH, SCALER_PATH, LABEL_ENCODER_PATH, METADATA_PATH,
    SAMPLE_RATE, MIN_AUDIO_DURATION, MAX_AUDIO_SIZE_MB, BACKEND_HOST, BACKEND_PORT
)
from feature_extraction import extract_features_from_bytes, extract_features_from_waveform
from preprocessing import preprocess_audio_bytes, validate_audio

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ─── FastAPI App ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Speech Emotion Detection API",
    description="Detect human emotions from spoken audio using Machine Learning.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow frontend to call the API from any origin (for local development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # In production, restrict to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Model storage (loaded once at startup) ───────────────────────────────────
class ModelStore:
    """Holds the loaded model, scaler, and label encoder."""
    model = None
    scaler = None
    label_encoder = None
    metadata: Dict = {}
    is_loaded: bool = False
    load_error: Optional[str] = None


model_store = ModelStore()


def load_model() -> bool:
    """
    Load the trained model from disk.
    Called once at startup — not on every request.
    """
    global model_store

    required_files = {
        "Model": MODEL_PATH,
        "Scaler": SCALER_PATH,
        "Label Encoder": LABEL_ENCODER_PATH,
    }

    for name, path in required_files.items():
        if not Path(path).exists():
            msg = (
                f"Missing {name}: {path}. "
                f"Please run 'python src/train.py' first to train the model."
            )
            logger.error(msg)
            model_store.load_error = msg
            model_store.is_loaded = False
            return False

    try:
        logger.info("Loading model artifacts...")
        model_store.model         = joblib.load(MODEL_PATH)
        model_store.scaler        = joblib.load(SCALER_PATH)
        model_store.label_encoder = joblib.load(LABEL_ENCODER_PATH)

        if METADATA_PATH.exists():
            with open(METADATA_PATH) as f:
                model_store.metadata = json.load(f)

        model_store.is_loaded = True
        model_store.load_error = None

        classes = list(model_store.label_encoder.classes_)
        logger.info(f"✅ Model loaded: {model_store.metadata.get('model_type', 'Unknown')}")
        logger.info(f"   Emotion classes: {classes}")
        logger.info(f"   Test accuracy: {model_store.metadata.get('test_accuracy', 'N/A')}")
        return True

    except Exception as e:
        msg = f"Failed to load model: {e}"
        logger.error(msg)
        model_store.load_error = msg
        model_store.is_loaded = False
        return False


def determine_audio_format(filename: str, content_type: str) -> str:
    """
    Determine audio format from filename/content-type.
    Browser-recorded audio is often WebM/Opus.
    """
    # Check content type first
    ct_map = {
        "audio/webm": "webm",
        "audio/ogg": "ogg",
        "audio/wav": "wav",
        "audio/x-wav": "wav",
        "audio/mpeg": "mp3",
        "audio/mp4": "mp4",
        "audio/flac": "flac",
        "video/webm": "webm",  # Some browsers send video/webm for audio
    }
    for ct, fmt in ct_map.items():
        if ct in content_type.lower():
            return fmt

    # Fall back to file extension
    ext = Path(filename).suffix.lower().lstrip('.')
    if ext in ['wav', 'mp3', 'ogg', 'flac', 'webm', 'm4a', 'mp4']:
        return ext

    return "webm"  # Default: assume browser recording


def predict_emotion(audio_bytes: bytes, audio_format: str) -> Dict:
    """
    Core prediction function.

    Pipeline:
    1. Write bytes to temp file
    2. Preprocess (mono, resample, normalize, pad/trim)
    3. Extract features (MFCC, delta, ZCR, RMS, centroid, rolloff, chroma)
    4. Scale features using saved StandardScaler
    5. Run ML model prediction
    6. Get probability estimates
    7. Return structured result

    Args:
        audio_bytes: Raw audio bytes from upload
        audio_format: Audio format string ('webm', 'wav', etc.)

    Returns:
        Dict with emotion, confidence, probabilities
    """
    if not model_store.is_loaded:
        raise ValueError("Model not loaded. Run training first.")

    # Step 1-3: Preprocess and extract features
    features = extract_features_from_bytes(audio_bytes, audio_format)
    if features is None:
        raise ValueError(
            "Feature extraction failed. The audio may be too short, silent, "
            "or in an unsupported format."
        )

    # Step 4: Scale features (MUST use the scaler fitted during training)
    features_scaled = model_store.scaler.transform(features.reshape(1, -1))

    # Step 5: Predict emotion label
    predicted_label = model_store.model.predict(features_scaled)[0]

    # Step 6: Get probabilities
    # SVM with probability=True uses Platt scaling (calibrated sigmoid)
    # Random Forest uses vote fractions
    # These are the model's confidence estimates, not perfect calibrated probabilities
    try:
        proba = model_store.model.predict_proba(features_scaled)[0]
        classes = model_store.label_encoder.classes_
        probabilities = {cls: float(round(prob, 4)) for cls, prob in zip(classes, proba)}
        confidence = float(max(proba))
    except AttributeError:
        # Fallback if model doesn't support predict_proba
        # Use a one-hot confidence (1.0 for predicted class)
        classes = model_store.label_encoder.classes_
        probabilities = {cls: 0.0 for cls in classes}
        probabilities[predicted_label] = 1.0
        confidence = 1.0
        logger.warning("Model doesn't support probability estimates; using binary confidence.")

    return {
        "emotion": predicted_label,
        "confidence": round(confidence, 4),
        "probabilities": probabilities,
        "timestamp": datetime.now().isoformat(),
    }


# ─── Pydantic response models ─────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: Optional[str] = None
    emotion_classes: Optional[List[str]] = None
    test_accuracy: Optional[float] = None
    error: Optional[str] = None


class PredictionResponse(BaseModel):
    emotion: str
    confidence: float
    probabilities: Dict[str, float]
    timestamp: str


# ─── API Endpoints ─────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Load model when the server starts."""
    logger.info("Starting Speech Emotion Detection API...")
    load_model()


@app.get("/health", response_model=HealthResponse, summary="Health check")
async def health_check():
    """
    Check if the API is running and the model is loaded.

    Returns model status, emotion classes, and accuracy info.
    """
    if model_store.is_loaded:
        return HealthResponse(
            status="ok",
            model_loaded=True,
            model_type=model_store.metadata.get("model_type"),
            emotion_classes=list(model_store.label_encoder.classes_),
            test_accuracy=model_store.metadata.get("test_accuracy"),
        )
    else:
        return HealthResponse(
            status="degraded",
            model_loaded=False,
            error=model_store.load_error or "Model not loaded",
        )


@app.post("/reload-model", summary="Hot-reload model artifacts from disk")
async def reload_model_endpoint():
    """Hot-reload model, scaler, and label encoder without restarting the server."""
    success = load_model()
    if success:
        return {
            "status": "success",
            "message": "Model reloaded successfully",
            "model_type": model_store.metadata.get("model_type"),
            "test_accuracy": model_store.metadata.get("test_accuracy"),
            "feature_dimension": model_store.metadata.get("feature_dimension", 560),
        }
    else:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": model_store.load_error or "Failed to reload model"}
        )


@app.get("/emotions", summary="List supported emotions")
async def get_emotions():
    """Return the list of emotion classes the model can detect."""
    if not model_store.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")

    return {
        "emotions": list(model_store.label_encoder.classes_),
        "count": len(model_store.label_encoder.classes_),
    }


@app.get("/model-info", summary="Model metadata")
async def get_model_info():
    """Return detailed metadata about the trained model."""
    if not model_store.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")

    return model_store.metadata


@app.post("/predict", response_model=PredictionResponse, summary="Predict emotion from audio")
async def predict(audio: UploadFile = File(..., description="Audio file (WAV, WebM, OGG, FLAC, MP3)")):
    """
    Predict the emotion in a spoken audio recording.

    Accepts audio in any format (WAV, WebM/Opus from browser, OGG, FLAC, MP3).
    
    The audio is processed through the same pipeline used during training:
    1. Format conversion (if needed)
    2. Mono conversion + resampling to 22050 Hz
    3. Silence removal + normalization
    4. Fixed-length padding/trimming to 3 seconds
    5. MFCC + additional feature extraction
    6. Feature scaling
    7. ML model prediction

    Returns:
        emotion: Predicted emotion label
        confidence: Model confidence score [0.0, 1.0]
        probabilities: Score for each emotion class (sums to ~1.0)
    """
    if not model_store.is_loaded:
        raise HTTPException(
            status_code=503,
            detail=f"Model not ready. {model_store.load_error or 'Run python src/train.py first.'}"
        )

    # Validate file size
    audio_bytes = await audio.read()
    file_size_mb = len(audio_bytes) / (1024 * 1024)

    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty audio file received.")

    if file_size_mb > MAX_AUDIO_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {file_size_mb:.1f} MB. Maximum allowed: {MAX_AUDIO_SIZE_MB} MB."
        )

    # Determine audio format
    filename = audio.filename or "recording.webm"
    content_type = audio.content_type or "audio/webm"
    audio_format = determine_audio_format(filename, content_type)

    logger.info(
        f"Received audio: {filename}, format={audio_format}, "
        f"size={file_size_mb:.2f} MB, content_type={content_type}"
    )

    # Quick duration check using librosa (without full preprocessing)
    try:
        import librosa
        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix=f".{audio_format}", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        try:
            y_check, sr_check = librosa.load(tmp_path, sr=None, mono=True)
            duration = len(y_check) / sr_check
            logger.info(f"Audio duration: {duration:.2f}s, sample rate: {sr_check} Hz")
            if duration < MIN_AUDIO_DURATION:
                raise HTTPException(
                    status_code=400,
                    detail=f"Recording too short: {duration:.1f}s. Please speak for at least {MIN_AUDIO_DURATION}s."
                )
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"Duration check failed: {e} — proceeding with prediction anyway.")

    # Run prediction
    try:
        result = predict_emotion(audio_bytes, audio_format)
        logger.info(
            f"Prediction: {result['emotion']} "
            f"(confidence: {result['confidence']:.2%})"
        )
        return PredictionResponse(**result)

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal error during emotion prediction. Please try again."
        )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all exception handler — never expose stack traces to users."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again."}
    )


# ─── Serve Frontend ──────────────────────────────────────────────────────────
FRONTEND_DIR = PROJECT_ROOT / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_index():
        return FileResponse(FRONTEND_DIR / "index.html")

    @app.get("/style.css", include_in_schema=False)
    async def serve_css():
        return FileResponse(FRONTEND_DIR / "style.css", media_type="text/css")

    @app.get("/app.js", include_in_schema=False)
    async def serve_js():
        return FileResponse(FRONTEND_DIR / "app.js", media_type="application/javascript")


# ─── Main entry point ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=BACKEND_HOST,
        port=BACKEND_PORT,
        reload=False,
        log_level="info",
    )
