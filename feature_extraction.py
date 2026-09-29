"""
feature_extraction.py - ENHANCED Feature Extraction for Speech Emotion Detection.

Features extracted (~560 total):
─────────────────────────────────────────────────────────────────────
Feature                  | Dim  | Why it helps
─────────────────────────────────────────────────────────────────────
MFCC (40 coeff)          | 80   | Vocal tract shape — primary SER feature
MFCC Delta               | 80   | Rate of change — speech dynamics
MFCC Delta-Delta         | 80   | Acceleration — captures emotional transitions
Zero Crossing Rate       | 2    | Voiced vs unvoiced segments
RMS Energy               | 2    | Loudness — angry vs calm/sad
Spectral Centroid        | 2    | Brightness — different per emotion
Spectral Rolloff         | 2    | Tone characteristic
Spectral Bandwidth       | 2    | Spread of spectrum
Spectral Contrast        | 14   | Peak-valley per sub-band — very discriminative
Chroma STFT              | 24   | Pitch class energy — tonal content
Tonnetz                  | 12   | Tonal centroid — harmony relationships
Fundamental Frequency F0 | 4    | Pitch mean/std/min/max — #1 emotion correlate
Mel Spectrogram Stats    | 256  | 128 mel bands × (mean+std) — rich texture
─────────────────────────────────────────────────────────────────────
Total feature vector: ~560 values (vs 192 before — 3x richer)
"""

import logging
import numpy as np
import librosa
from pathlib import Path
from typing import Optional
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    SAMPLE_RATE, N_MFCC, N_MELS, HOP_LENGTH, N_FFT, FEATURE_CONFIG
)
from preprocessing import preprocess_audio

logger = logging.getLogger(__name__)


def extract_features_from_waveform(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
) -> Optional[np.ndarray]:
    """
    Extract a rich fixed-size feature vector from a preprocessed audio waveform.

    Both training and live inference call this — identical features ensure
    training/inference consistency.

    Args:
        y:  Audio waveform (numpy array, mono, fixed length)
        sr: Sample rate in Hz

    Returns:
        1D numpy array of features (~560 values), or None on failure.
    """
    try:
        features = []

        # ── 1. MFCC mean + std (80 values) ────────────────────────────────────
        mfcc = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH
        )
        features.extend(np.mean(mfcc, axis=1))
        features.extend(np.std(mfcc, axis=1))

        # ── 2. MFCC Delta mean + std (80 values) ──────────────────────────────
        mfcc_delta = librosa.feature.delta(mfcc)
        features.extend(np.mean(mfcc_delta, axis=1))
        features.extend(np.std(mfcc_delta, axis=1))

        # ── 3. MFCC Delta-Delta (acceleration) mean + std (80 values) ─────────
        # Captures rapid emotional transitions — angry bursts vs slow sad decline
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        features.extend(np.mean(mfcc_delta2, axis=1))
        features.extend(np.std(mfcc_delta2, axis=1))

        # ── 4. Zero Crossing Rate (2 values) ──────────────────────────────────
        zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=HOP_LENGTH)
        features.append(float(np.mean(zcr)))
        features.append(float(np.std(zcr)))

        # ── 5. RMS Energy (2 values) ───────────────────────────────────────────
        rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)
        features.append(float(np.mean(rms)))
        features.append(float(np.std(rms)))

        # ── 6. Spectral Centroid (2 values) ───────────────────────────────────
        spec_centroid = librosa.feature.spectral_centroid(
            y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
        )
        features.append(float(np.mean(spec_centroid)))
        features.append(float(np.std(spec_centroid)))

        # ── 7. Spectral Rolloff (2 values) ────────────────────────────────────
        rolloff = librosa.feature.spectral_rolloff(
            y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
        )
        features.append(float(np.mean(rolloff)))
        features.append(float(np.std(rolloff)))

        # ── 8. Spectral Bandwidth (2 values) ──────────────────────────────────
        # Width of the spectral distribution — wide=noisy/excited, narrow=calm
        bandwidth = librosa.feature.spectral_bandwidth(
            y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
        )
        features.append(float(np.mean(bandwidth)))
        features.append(float(np.std(bandwidth)))

        # ── 9. Spectral Contrast (14 values: 7 bands × mean+std) ─────────────
        # Difference between peaks and valleys per frequency sub-band.
        # Angry/happy → high contrast; Sad/neutral → flatter spectrum
        spec_contrast = librosa.feature.spectral_contrast(
            y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_bands=6
        )
        features.extend(np.mean(spec_contrast, axis=1))
        features.extend(np.std(spec_contrast, axis=1))

        # ── 10. Chroma STFT (24 values) ───────────────────────────────────────
        chroma = librosa.feature.chroma_stft(
            y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
        )
        features.extend(np.mean(chroma, axis=1))
        features.extend(np.std(chroma, axis=1))

        # ── 11. Tonnetz — Tonal Centroid Features (12 values) ─────────────────
        try:
            tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
            features.extend(np.mean(tonnetz, axis=1))
            features.extend(np.std(tonnetz, axis=1))
        except Exception:
            features.extend([0.0] * 12)

        # ── 12. Fundamental Frequency / Pitch F0 (4 values) ──────────────────
        # Pitch is the #1 emotion-correlated acoustic feature in the human voice.
        try:
            f0 = librosa.yin(
                y, fmin=65, fmax=1000,
                sr=sr, hop_length=HOP_LENGTH
            )
            f0_valid = f0[np.isfinite(f0) & (f0 > 65)]
            if len(f0_valid) > 5:
                features.append(float(np.mean(f0_valid)))
                features.append(float(np.std(f0_valid)))
                features.append(float(np.min(f0_valid)))
                features.append(float(np.max(f0_valid)))
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])
        except Exception:
            features.extend([0.0, 0.0, 0.0, 0.0])

        # ── 13. Mel Spectrogram Statistics (256 values: 128 bands × mean+std) ─
        # Raw mel filterbank energy — richer than MFCCs (no DCT compression)
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        features.extend(np.mean(mel_spec_db, axis=1))
        features.extend(np.std(mel_spec_db, axis=1))

        feature_vector = np.array(features, dtype=np.float32)

        # Sanity check: replace NaN/Inf with 0
        if not np.all(np.isfinite(feature_vector)):
            n_bad = np.sum(~np.isfinite(feature_vector))
            logger.debug(f"Feature vector has {n_bad} non-finite values; replacing with 0.")
            feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)

        return feature_vector

    except Exception as e:
        logger.error(f"Feature extraction failed: {e}", exc_info=True)
        return None


def extract_features(audio_path: str) -> Optional[np.ndarray]:
    """
    MAIN ENTRY POINT for feature extraction from an audio file.

    Pipeline: preprocess_audio(audio_path) → extract_features_from_waveform(y, sr)

    Args:
        audio_path: Path to audio file (any format supported by librosa)

    Returns:
        1D numpy array of ~560 features, or None on failure
    """
    y, sr = preprocess_audio(str(audio_path))
    if y is None:
        logger.error(f"Preprocessing failed for {audio_path}")
        return None

    features = extract_features_from_waveform(y, sr)
    if features is None:
        logger.error(f"Feature extraction failed for {audio_path}")
        return None

    return features


def extract_features_from_bytes(
    audio_bytes: bytes,
    original_format: str = "wav",
) -> Optional[np.ndarray]:
    """
    Extract features from raw audio bytes (used in backend for live predictions).

    Args:
        audio_bytes: Raw audio bytes
        original_format: Format hint ('webm', 'wav', 'ogg', etc.)

    Returns:
        1D numpy array of features, or None on failure
    """
    from preprocessing import preprocess_audio_bytes

    y, sr = preprocess_audio_bytes(audio_bytes, original_format)
    if y is None:
        logger.error("Preprocessing from bytes failed.")
        return None

    return extract_features_from_waveform(y, sr)


def get_feature_dimension() -> int:
    """
    Return the expected feature vector dimension.
    80+80+80+2+2+2+2+2+14+24+12+4+256 = 560
    """
    dim = 0
    dim += N_MFCC * 2          # MFCC mean + std
    dim += N_MFCC * 2          # Delta mean + std
    dim += N_MFCC * 2          # Delta-delta mean + std
    dim += 2                   # ZCR
    dim += 2                   # RMS
    dim += 2                   # Spectral centroid
    dim += 2                   # Spectral rolloff
    dim += 2                   # Spectral bandwidth
    dim += (6 + 1) * 2         # Spectral contrast: 7 bands × (mean+std)
    dim += 12 * 2              # Chroma: 12 bins × (mean+std)
    dim += 6 * 2               # Tonnetz: 6 dims × (mean+std)
    dim += 4                   # F0: mean, std, min, max
    dim += N_MELS * 2          # Mel spectrogram: 128 bands × (mean+std)
    return dim
