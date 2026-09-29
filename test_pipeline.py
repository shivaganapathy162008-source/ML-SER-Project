"""
test_pipeline.py - Automated tests for the Speech Emotion Detection system.

Tests cover:
1. Audio preprocessing
2. Feature extraction
3. Model loading
4. Prediction pipeline
5. Backend API endpoints

Run with:
    python -m pytest tests/ -v
    or
    python tests/test_pipeline.py
"""

import sys
import os
import json
import struct
import wave
import pytest
import numpy as np
from pathlib import Path
from io import BytesIO

# Setup path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(BACKEND_DIR))

# ─── Helpers ──────────────────────────────────────────────────────────────────

def generate_sine_wave_wav(frequency=440, duration=3.0, sample_rate=22050) -> bytes:
    """Generate a WAV file containing a sine wave — used for testing without real audio."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = (np.sin(2 * np.pi * frequency * t) * 0.5 * 32767).astype(np.int16)

    buffer = BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(1)          # Mono
        wf.setsampwidth(2)          # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())
    return buffer.getvalue()


def generate_silent_wav(duration=0.1, sample_rate=22050) -> bytes:
    """Generate a nearly silent WAV file."""
    n_samples = int(sample_rate * duration)
    audio = np.zeros(n_samples, dtype=np.int16)

    buffer = BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())
    return buffer.getvalue()


def save_temp_wav(content: bytes) -> str:
    """Save bytes to a temp WAV file and return its path."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        f.write(content)
        return f.name


# ═══════════════════════════════════════════════════════════════════════════════
# TEST GROUP 1: Audio Preprocessing
# ═══════════════════════════════════════════════════════════════════════════════

class TestPreprocessing:
    """Tests for src/preprocessing.py"""

    def test_load_audio_valid(self):
        """Should load a valid WAV file successfully."""
        from preprocessing import load_audio

        wav_bytes = generate_sine_wave_wav(frequency=440, duration=2.0)
        tmp_path = save_temp_wav(wav_bytes)
        try:
            y, sr = load_audio(tmp_path, target_sr=22050)
            assert y is not None, "Audio should not be None"
            assert len(y) > 0, "Audio array should not be empty"
            assert sr == 22050, f"Sample rate should be 22050, got {sr}"
        finally:
            os.unlink(tmp_path)

    def test_load_audio_nonexistent(self):
        """Should return (None, 0) for missing file."""
        from preprocessing import load_audio

        y, sr = load_audio("/nonexistent/path/audio.wav")
        assert y is None, "Should return None for missing file"
        assert sr == 0

    def test_validate_audio_valid(self):
        """Valid audio should pass validation."""
        from preprocessing import validate_audio

        y = np.sin(2 * np.pi * 440 * np.linspace(0, 3, 66150)).astype(np.float32)
        is_valid, reason = validate_audio(y, sr=22050)
        assert is_valid, f"Valid audio failed validation: {reason}"
        assert reason == "OK"

    def test_validate_audio_too_short(self):
        """Audio below min duration should fail validation."""
        from preprocessing import validate_audio

        y = np.zeros(100, dtype=np.float32)  # ~0.005s at 22050 Hz
        y[50] = 0.5  # Not silent
        is_valid, reason = validate_audio(y, sr=22050, min_duration=0.3)
        assert not is_valid, "Very short audio should fail validation"

    def test_validate_audio_silent(self):
        """Silent audio should fail validation."""
        from preprocessing import validate_audio

        y = np.zeros(22050, dtype=np.float32)  # 1 second of silence
        is_valid, reason = validate_audio(y, sr=22050)
        assert not is_valid, "Silent audio should fail validation"

    def test_normalize_audio(self):
        """Normalized audio should have max amplitude of 1.0."""
        from preprocessing import normalize_audio

        y = np.array([0.2, 0.5, -0.3, 0.8, -0.6], dtype=np.float32)
        y_norm = normalize_audio(y)
        assert abs(np.max(np.abs(y_norm)) - 1.0) < 1e-5, "Max amplitude should be 1.0 after normalization"

    def test_pad_or_trim_padding(self):
        """Short audio should be zero-padded to target duration."""
        from preprocessing import pad_or_trim

        y = np.ones(11025, dtype=np.float32)  # 0.5s at 22050 Hz
        y_padded = pad_or_trim(y, sr=22050, duration=3.0)
        expected_len = int(22050 * 3.0)
        assert len(y_padded) == expected_len, f"Padded length should be {expected_len}, got {len(y_padded)}"
        assert np.all(y_padded[11025:] == 0), "Padded portion should be zeros"

    def test_pad_or_trim_trimming(self):
        """Long audio should be trimmed to target duration."""
        from preprocessing import pad_or_trim

        y = np.ones(100000, dtype=np.float32)  # ~4.5s at 22050 Hz
        y_trimmed = pad_or_trim(y, sr=22050, duration=3.0)
        expected_len = int(22050 * 3.0)
        assert len(y_trimmed) == expected_len, f"Trimmed length should be {expected_len}"

    def test_full_preprocess_pipeline(self):
        """Full preprocessing pipeline should return correct array dimensions."""
        from preprocessing import preprocess_audio

        wav_bytes = generate_sine_wave_wav(frequency=440, duration=4.0)
        tmp_path = save_temp_wav(wav_bytes)
        try:
            y, sr = preprocess_audio(tmp_path, target_sr=22050, duration=3.0)
            assert y is not None, "Preprocessed audio should not be None"
            assert sr == 22050
            assert len(y) == int(22050 * 3.0), f"Expected {22050*3} samples, got {len(y)}"
        finally:
            os.unlink(tmp_path)

    def test_preprocess_bytes(self):
        """Preprocessing from bytes should work correctly."""
        from preprocessing import preprocess_audio_bytes

        wav_bytes = generate_sine_wave_wav(frequency=440, duration=3.0)
        y, sr = preprocess_audio_bytes(wav_bytes, original_format='wav')
        assert y is not None
        assert sr == 22050
        assert len(y) == int(22050 * 3.0)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST GROUP 2: Feature Extraction
# ═══════════════════════════════════════════════════════════════════════════════

class TestFeatureExtraction:
    """Tests for src/feature_extraction.py"""

    def test_feature_dimension(self):
        """Feature vector should have the expected dimension."""
        from feature_extraction import get_feature_dimension

        dim = get_feature_dimension()
        assert dim > 0, "Feature dimension must be positive"
        # With current config: 80 + 80 + 2 + 2 + 2 + 2 + 24 = 192
        assert dim == 192, f"Expected 192 features, got {dim}"

    def test_extract_features_from_waveform(self):
        """Feature extraction from a waveform should return correct shape."""
        from feature_extraction import extract_features_from_waveform, get_feature_dimension

        y = np.sin(2 * np.pi * 440 * np.linspace(0, 3, 66150)).astype(np.float32)
        features = extract_features_from_waveform(y, sr=22050)

        assert features is not None, "Features should not be None"
        expected_dim = get_feature_dimension()
        assert len(features) == expected_dim, f"Expected {expected_dim} features, got {len(features)}"

    def test_features_are_finite(self):
        """Feature vector should not contain NaN or Inf values."""
        from feature_extraction import extract_features_from_waveform

        y = np.sin(2 * np.pi * 440 * np.linspace(0, 3, 66150)).astype(np.float32)
        features = extract_features_from_waveform(y, sr=22050)

        assert features is not None
        assert np.all(np.isfinite(features)), "Features contain NaN or Inf values"

    def test_extract_features_from_file(self):
        """extract_features() should work end-to-end on a WAV file."""
        from feature_extraction import extract_features, get_feature_dimension

        wav_bytes = generate_sine_wave_wav(duration=3.0)
        tmp_path = save_temp_wav(wav_bytes)
        try:
            features = extract_features(tmp_path)
            assert features is not None, "Features should not be None"
            assert len(features) == get_feature_dimension()
        finally:
            os.unlink(tmp_path)

    def test_extract_features_from_bytes(self):
        """extract_features_from_bytes() should work for API use case."""
        from feature_extraction import extract_features_from_bytes, get_feature_dimension

        wav_bytes = generate_sine_wave_wav(duration=3.0)
        features = extract_features_from_bytes(wav_bytes, 'wav')

        assert features is not None
        assert len(features) == get_feature_dimension()

    def test_feature_consistency(self):
        """Same audio → same features (deterministic)."""
        from feature_extraction import extract_features_from_waveform

        y = np.sin(2 * np.pi * 440 * np.linspace(0, 3, 66150)).astype(np.float32)
        f1 = extract_features_from_waveform(y, sr=22050)
        f2 = extract_features_from_waveform(y, sr=22050)

        assert f1 is not None and f2 is not None
        np.testing.assert_array_almost_equal(f1, f2, decimal=5,
            err_msg="Feature extraction is not deterministic")

    def test_different_emotions_different_features(self):
        """Different audio signals should produce different feature vectors."""
        from feature_extraction import extract_features_from_waveform

        t = np.linspace(0, 3, 66150)
        y_low  = (np.sin(2 * np.pi * 200 * t) * 0.3).astype(np.float32)   # Low pitch, quiet
        y_high = (np.sin(2 * np.pi * 2000 * t) * 0.9).astype(np.float32)  # High pitch, loud

        f1 = extract_features_from_waveform(y_low, sr=22050)
        f2 = extract_features_from_waveform(y_high, sr=22050)

        assert f1 is not None and f2 is not None
        assert not np.allclose(f1, f2), "Different audio should produce different features"


# ═══════════════════════════════════════════════════════════════════════════════
# TEST GROUP 3: Model
# ═══════════════════════════════════════════════════════════════════════════════

class TestModel:
    """Tests for model loading and prediction (requires trained model)."""

    MODEL_AVAILABLE = (PROJECT_ROOT / "models" / "speech_emotion_model.pkl").exists()

    def test_model_files_exist(self):
        """All required model files should exist after training."""
        if not self.MODEL_AVAILABLE:
            pytest.skip("Model not yet trained. Run: python src/train.py")

        from config import MODEL_PATH, SCALER_PATH, LABEL_ENCODER_PATH, METADATA_PATH
        assert MODEL_PATH.exists(), f"Model file missing: {MODEL_PATH}"
        assert SCALER_PATH.exists(), f"Scaler missing: {SCALER_PATH}"
        assert LABEL_ENCODER_PATH.exists(), f"Label encoder missing: {LABEL_ENCODER_PATH}"
        assert METADATA_PATH.exists(), f"Metadata missing: {METADATA_PATH}"

    def test_model_loads_correctly(self):
        """Model should load without errors."""
        if not self.MODEL_AVAILABLE:
            pytest.skip("Model not yet trained. Run: python src/train.py")

        import joblib
        from config import MODEL_PATH, SCALER_PATH, LABEL_ENCODER_PATH

        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        label_encoder = joblib.load(LABEL_ENCODER_PATH)

        assert model is not None
        assert scaler is not None
        assert label_encoder is not None
        assert len(label_encoder.classes_) >= 2, "Should have at least 2 emotion classes"

    def test_model_metadata(self):
        """Model metadata should contain required fields."""
        if not self.MODEL_AVAILABLE:
            pytest.skip("Model not yet trained. Run: python src/train.py")

        from config import METADATA_PATH

        with open(METADATA_PATH) as f:
            metadata = json.load(f)

        required_keys = ['model_type', 'dataset', 'emotion_classes', 'test_accuracy', 'feature_dimension']
        for key in required_keys:
            assert key in metadata, f"Metadata missing key: {key}"

        assert metadata['test_accuracy'] > 0, "Test accuracy should be positive"
        assert metadata['feature_dimension'] == 192, "Feature dimension should be 192"
        assert len(metadata['emotion_classes']) >= 2

    def test_prediction_on_synthetic_audio(self):
        """Model should return a valid prediction on synthetic audio."""
        if not self.MODEL_AVAILABLE:
            pytest.skip("Model not yet trained. Run: python src/train.py")

        import joblib
        from config import MODEL_PATH, SCALER_PATH, LABEL_ENCODER_PATH
        from feature_extraction import extract_features_from_waveform

        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        label_encoder = joblib.load(LABEL_ENCODER_PATH)

        # Generate test audio
        y = np.sin(2 * np.pi * 440 * np.linspace(0, 3, 66150)).astype(np.float32)
        features = extract_features_from_waveform(y, sr=22050)

        assert features is not None
        features_scaled = scaler.transform(features.reshape(1, -1))
        prediction = model.predict(features_scaled)[0]

        assert prediction in label_encoder.classes_, \
            f"Prediction '{prediction}' not in known classes {label_encoder.classes_}"

    def test_prediction_returns_probabilities(self):
        """Model should return probabilities for all emotion classes."""
        if not self.MODEL_AVAILABLE:
            pytest.skip("Model not yet trained. Run: python src/train.py")

        import joblib
        from config import MODEL_PATH, SCALER_PATH, LABEL_ENCODER_PATH
        from feature_extraction import extract_features_from_waveform

        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        label_encoder = joblib.load(LABEL_ENCODER_PATH)

        y = np.sin(2 * np.pi * 440 * np.linspace(0, 3, 66150)).astype(np.float32)
        features = extract_features_from_waveform(y, sr=22050)
        features_scaled = scaler.transform(features.reshape(1, -1))

        proba = model.predict_proba(features_scaled)[0]

        assert len(proba) == len(label_encoder.classes_), \
            "Probability array length should match number of classes"
        assert abs(sum(proba) - 1.0) < 0.01, "Probabilities should sum to ~1.0"
        assert all(0 <= p <= 1 for p in proba), "All probabilities should be in [0, 1]"


# ═══════════════════════════════════════════════════════════════════════════════
# TEST GROUP 4: Backend API
# ═══════════════════════════════════════════════════════════════════════════════

class TestBackendAPI:
    """Tests for the FastAPI backend (requires server to be running or uses TestClient)."""

    @pytest.fixture
    def client(self):
        """FastAPI test client — no server needed."""
        try:
            from fastapi.testclient import TestClient
            sys.path.insert(0, str(BACKEND_DIR))
            from main import app, load_model
            load_model()
            with TestClient(app) as test_client:
                yield test_client
        except ImportError:
            pytest.skip("httpx not installed for TestClient")

    def test_health_endpoint(self, client):
        """/health should return 200."""
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "model_loaded" in data

    def test_emotions_endpoint(self, client):
        """/emotions should return emotion list (if model loaded)."""
        resp = client.get("/emotions")
        if resp.status_code == 503:
            pytest.skip("Model not loaded — run training first")
        assert resp.status_code == 200
        data = resp.json()
        assert "emotions" in data
        assert len(data["emotions"]) >= 2

    def test_model_info_endpoint(self, client):
        """/model-info should return metadata."""
        resp = client.get("/model-info")
        if resp.status_code == 503:
            pytest.skip("Model not loaded — run training first")
        assert resp.status_code == 200

    def test_predict_valid_audio(self, client):
        """/predict with valid WAV should return emotion prediction."""
        MODEL_AVAILABLE = (PROJECT_ROOT / "models" / "speech_emotion_model.pkl").exists()
        if not MODEL_AVAILABLE:
            pytest.skip("Model not trained yet. Run: python src/train.py")

        wav_bytes = generate_sine_wave_wav(duration=3.0)
        files = {'audio': ('test.wav', wav_bytes, 'audio/wav')}
        resp = client.post("/predict", files=files)

        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        data = resp.json()
        assert "emotion" in data
        assert "confidence" in data
        assert "probabilities" in data
        assert 0.0 <= data["confidence"] <= 1.0
        assert len(data["probabilities"]) >= 2
        assert abs(sum(data["probabilities"].values()) - 1.0) < 0.05

    def test_predict_empty_audio(self, client):
        """/predict with empty file should return 400."""
        files = {'audio': ('empty.wav', b'', 'audio/wav')}
        resp = client.post("/predict", files=files)
        assert resp.status_code in [400, 422, 503], \
            f"Empty audio should return error, got {resp.status_code}"

    def test_predict_invalid_data(self, client):
        """/predict with random bytes should handle gracefully (not crash)."""
        files = {'audio': ('garbage.wav', b'not audio data at all!#@$', 'audio/wav')}
        resp = client.post("/predict", files=files)
        assert resp.status_code in [400, 422, 500, 503], \
            f"Invalid audio should return error, not {resp.status_code}"

    def test_predict_short_audio(self, client):
        """/predict with very short audio may return 400."""
        MODEL_AVAILABLE = (PROJECT_ROOT / "models" / "speech_emotion_model.pkl").exists()
        if not MODEL_AVAILABLE:
            pytest.skip("Model not trained")

        wav_bytes = generate_silent_wav(duration=0.05)  # 50ms — very short
        files = {'audio': ('short.wav', wav_bytes, 'audio/wav')}
        resp = client.post("/predict", files=files)
        # Either 400 (too short) or 422 (feature extraction failed) is acceptable
        assert resp.status_code in [200, 400, 422, 503], \
            f"Unexpected status: {resp.status_code}"


# ═══════════════════════════════════════════════════════════════════════════════
# TEST GROUP 5: Dataset
# ═══════════════════════════════════════════════════════════════════════════════

class TestDataset:
    """Tests for dataset loading (RAVDESS must be present)."""

    DATASET_AVAILABLE = (PROJECT_ROOT / "data" / "raw" / "RAVDESS").exists() and \
                        any((PROJECT_ROOT / "data" / "raw" / "RAVDESS").rglob("*.wav"))

    def test_parse_ravdess_filename_valid(self):
        """Valid RAVDESS filename should parse correctly."""
        from dataset import parse_ravdess_filename

        # 03-01-05-01-01-01-12.wav → Actor 12, Emotion 05 (Angry), speech
        path = Path("03-01-05-01-01-01-12.wav")
        result = parse_ravdess_filename(path)

        assert result is not None, "Should parse valid RAVDESS filename"
        assert result['emotion'] == 'Angry'
        assert result['actor_id'] == 12
        assert result['emotion_code'] == '05'

    def test_parse_ravdess_filename_song(self):
        """Song files (vocal_channel=02) should be skipped."""
        from dataset import parse_ravdess_filename

        # 03-02-05-01-01-01-12.wav → song (vocal_channel=02)
        path = Path("03-02-05-01-01-01-12.wav")
        result = parse_ravdess_filename(path)
        assert result is None, "Song files should return None"

    def test_parse_ravdess_all_emotions(self):
        """All 8 emotion codes should parse correctly."""
        from dataset import parse_ravdess_filename
        from config import RAVDESS_EMOTION_MAP

        for code, expected_label in RAVDESS_EMOTION_MAP.items():
            path = Path(f"03-01-{code}-01-01-01-01.wav")
            result = parse_ravdess_filename(path)
            assert result is not None, f"Failed to parse emotion code {code}"
            assert result['emotion'] == expected_label, \
                f"Expected {expected_label}, got {result['emotion']}"

    def test_dataset_not_found_returns_empty(self):
        """Loading from non-existent directory should return empty DataFrame."""
        from dataset import load_dataset_info
        from pathlib import Path

        df = load_dataset_info(Path("/nonexistent/path"))
        assert df.empty, "Should return empty DataFrame when dataset not found"

    def test_speaker_split_no_overlap(self):
        """Train, val, and test speaker sets should not overlap."""
        if not self.DATASET_AVAILABLE:
            pytest.skip("RAVDESS dataset not present")

        from dataset import load_dataset_info, get_speaker_split
        from config import RAW_DIR

        df = load_dataset_info(RAW_DIR)
        if df.empty:
            pytest.skip("Dataset empty")

        train_a, val_a, test_a = get_speaker_split(df)

        train_set = set(train_a)
        val_set   = set(val_a)
        test_set  = set(test_a)

        assert not (train_set & val_set), "Train and Val speaker sets overlap!"
        assert not (train_set & test_set), "Train and Test speaker sets overlap!"
        assert not (val_set & test_set), "Val and Test speaker sets overlap!"


# ─── Main runner ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"],
        cwd=str(PROJECT_ROOT)
    )
    sys.exit(result.returncode)
