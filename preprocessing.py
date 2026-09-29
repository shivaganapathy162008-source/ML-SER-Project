"""
preprocessing.py - Audio preprocessing pipeline for Speech Emotion Detection.

This EXACT same pipeline is used for:
  1. Training audio files
  2. Test audio files
  3. Live microphone recordings from the backend

Any change here affects ALL three — keeping training and inference consistent.
"""

import logging
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from typing import Optional, Tuple

# Import config from same src/ package
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import SAMPLE_RATE, AUDIO_DURATION, TOP_DB

logger = logging.getLogger(__name__)


def transcode_to_wav(input_path: Path, output_wav_path: Path) -> bool:
    """Convert any audio file format to 16-bit PCM WAV using imageio_ffmpeg."""
    try:
        import subprocess
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_exe, "-y",
            "-i", str(input_path),
            "-ac", "1",
            "-ar", "22050",
            "-sample_fmt", "s16",
            str(output_wav_path)
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15)
        return result.returncode == 0 and output_wav_path.exists()
    except Exception as e:
        logger.error(f"Transcoding failed for {input_path}: {e}")
        return False


def load_audio(
    audio_path: str,
    target_sr: int = SAMPLE_RATE,
) -> Tuple[Optional[np.ndarray], int]:
    """
    Load an audio file and return (waveform, sample_rate).

    Steps:
    1. Read the file with librosa.
    2. If direct read fails (e.g. webm/opus on Windows), transcode to wav via ffmpeg.
    3. Force mono and resample to target_sr.

    Returns (None, 0) if loading fails.
    """
    audio_path = Path(audio_path)
    try:
        import soundfile as sf
        import scipy.signal as sps
        y, file_sr = sf.read(str(audio_path), dtype='float32')
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        if file_sr != target_sr:
            y = sps.resample_poly(y, target_sr, file_sr).astype(np.float32)
        return y, target_sr
    except Exception:
        pass

    try:
        y, sr = librosa.load(str(audio_path), sr=target_sr, mono=True)
        return y, sr
    except Exception as e:
        logger.info(f"Direct load failed for {audio_path.name} ({e}); attempting ffmpeg transcoding...")
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            tmp_wav_path = Path(tmp_wav.name)
        try:
            if transcode_to_wav(audio_path, tmp_wav_path):
                y, sr = librosa.load(str(tmp_wav_path), sr=target_sr, mono=True)
                return y, sr
            else:
                logger.error(f"Failed to load audio {audio_path}: {e}")
                return None, 0
        except Exception as e2:
            logger.error(f"Transcoded load failed for {audio_path}: {e2}")
            return None, 0
        finally:
            if tmp_wav_path.exists():
                try:
                    tmp_wav_path.unlink()
                except Exception:
                    pass


def validate_audio(y: np.ndarray, sr: int, min_duration: float = 0.3) -> Tuple[bool, str]:
    """
    Validate a loaded audio array.

    Checks:
    - Array is not None or empty
    - Duration is above minimum threshold
    - Audio is not pure silence

    Returns (is_valid, reason_if_invalid)

    Note: The RMS silence threshold is intentionally lenient (1e-4) because
    browser-recorded WebM/Opus audio often decodes at low amplitude.
    """
    if y is None or len(y) == 0:
        return False, "Audio array is empty or None."

    duration = len(y) / sr
    if duration < min_duration:
        return False, f"Audio too short: {duration:.2f}s (minimum {min_duration}s)."

    # Check if audio is pure silence (all zeros or near-zero)
    rms = np.sqrt(np.mean(y ** 2))
    logger.debug(f"Audio RMS level: {rms:.6f}, duration: {duration:.2f}s")
    if rms < 1e-4:
        return False, f"Audio appears to be silent (RMS={rms:.2e}, too low)."

    return True, "OK"


def remove_silence(y: np.ndarray, sr: int, top_db: int = TOP_DB) -> np.ndarray:
    """
    Fast vectorized silence trimming using energy thresholding with safety margins.
    """
    MIN_KEPT_DURATION = 0.5  # seconds
    if len(y) <= int(sr * MIN_KEPT_DURATION):
        return y

    try:
        max_amp = np.max(np.abs(y))
        thresh = max(max_amp * 0.05, 0.005)
        non_silent = np.where(np.abs(y) > thresh)[0]
        if len(non_silent) > sr * MIN_KEPT_DURATION:
            start = max(0, non_silent[0] - int(sr * 0.05))
            end = min(len(y), non_silent[-1] + int(sr * 0.05))
            return y[start:end]
    except Exception:
        pass
    return y


def normalize_audio(y: np.ndarray) -> np.ndarray:
    """
    Peak normalize audio to the range [-1.0, 1.0].

    This makes feature extraction independent of recording volume,
    which is important because different microphones/recordings have
    different loudness levels.
    """
    max_val = np.max(np.abs(y))
    if max_val > 0:
        return y / max_val
    return y


def pad_or_trim(y: np.ndarray, sr: int, duration: float = AUDIO_DURATION) -> np.ndarray:
    """
    Ensure all audio clips are exactly `duration` seconds long.

    - If shorter: zero-pad at the end.
    - If longer: trim to the first `duration` seconds.

    Why fixed length? Machine learning models need consistent input size.
    Feature extraction (MFCC) can work with variable length, but using
    fixed length ensures consistent feature dimensions across all samples.
    """
    target_length = int(sr * duration)
    current_length = len(y)

    if current_length < target_length:
        # Pad with zeros at the end
        padding = target_length - current_length
        y = np.pad(y, (0, padding), mode='constant', constant_values=0)
    elif current_length > target_length:
        # Trim to target length
        y = y[:target_length]

    return y


def preprocess_audio(
    audio_path: str,
    target_sr: int = SAMPLE_RATE,
    duration: float = AUDIO_DURATION,
    apply_silence_removal: bool = True,
    apply_normalization: bool = True,
) -> Tuple[Optional[np.ndarray], int]:
    """
    MAIN PREPROCESSING FUNCTION — used for training, testing, and live audio.

    Pipeline:
    1. Load audio (mono, resampled to target_sr)
    2. Validate audio
    3. Remove silence (optional)
    4. Normalize amplitude (optional)
    5. Pad or trim to fixed duration

    Args:
        audio_path: Path to audio file (WAV, FLAC, OGG, etc.)
        target_sr: Target sample rate in Hz
        duration: Fixed duration in seconds
        apply_silence_removal: Whether to trim silence
        apply_normalization: Whether to peak-normalize

    Returns:
        (processed_waveform, sample_rate) or (None, 0) on failure
    """
    # Step 1: Load
    y, sr = load_audio(audio_path, target_sr)
    if y is None:
        return None, 0

    # Step 2: Validate
    is_valid, reason = validate_audio(y, sr)
    if not is_valid:
        logger.warning(f"Audio validation failed for {audio_path}: {reason}")
        return None, 0

    # Step 3: Remove silence
    if apply_silence_removal:
        y = remove_silence(y, sr)

    # Step 4: Normalize
    if apply_normalization:
        y = normalize_audio(y)

    # Step 5: Pad or trim to fixed length
    y = pad_or_trim(y, sr, duration)

    return y, sr


def preprocess_audio_bytes(
    audio_bytes: bytes,
    original_format: str = "wav",
    target_sr: int = SAMPLE_RATE,
    duration: float = AUDIO_DURATION,
) -> Tuple[Optional[np.ndarray], int]:
    """
    Preprocess audio from raw bytes (used in the backend for uploaded files).

    The browser may send WebM/Opus audio. This function:
    1. Writes bytes to a temporary file with the correct extension.
    2. Attempts to load with librosa directly (works for WAV/FLAC/OGG).
    3. If that fails (common for WebM/Opus on Windows), transcodes to WAV via
       imageio_ffmpeg and retries.
    4. Applies the standard preprocessing pipeline.

    Args:
        audio_bytes: Raw audio bytes from HTTP upload
        original_format: Format hint (e.g., 'webm', 'wav', 'ogg')
        target_sr: Target sample rate
        duration: Fixed duration

    Returns:
        (processed_waveform, sample_rate) or (None, 0) on failure
    """
    import tempfile
    import os

    if not audio_bytes:
        logger.error("preprocess_audio_bytes: received empty bytes.")
        return None, 0

    suffix = f".{original_format}" if not original_format.startswith('.') else original_format

    # Write the raw bytes to a temp file
    tmp_input_path = None
    tmp_wav_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_input_path = tmp.name

        logger.debug(
            f"preprocess_audio_bytes: wrote {len(audio_bytes)} bytes to {tmp_input_path} "
            f"(format={original_format})"
        )

        # --- Attempt 1: direct preprocessing (works for WAV, FLAC, OGG) ---
        result = preprocess_audio(tmp_input_path, target_sr, duration)
        if result[0] is not None:
            logger.debug("preprocess_audio_bytes: direct load succeeded.")
            return result

        # --- Attempt 2: ffmpeg transcode to WAV, then preprocess ---
        # This is the common path for browser WebM/Opus on Windows where
        # librosa/soundfile can't decode WebM without system ffmpeg.
        logger.info(
            f"Direct load failed for {suffix} audio; "
            "attempting ffmpeg transcode to WAV..."
        )
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            tmp_wav_path = tmp_wav.name

        if transcode_to_wav(Path(tmp_input_path), Path(tmp_wav_path)):
            result = preprocess_audio(tmp_wav_path, target_sr, duration)
            if result[0] is not None:
                logger.info("preprocess_audio_bytes: ffmpeg transcode + load succeeded.")
                return result
            else:
                logger.error(
                    "preprocess_audio_bytes: ffmpeg transcode succeeded but "
                    "preprocessing still failed (audio may be silent or too short)."
                )
        else:
            logger.error(
                "preprocess_audio_bytes: ffmpeg transcode failed. "
                "Ensure imageio_ffmpeg is installed: pip install imageio-ffmpeg"
            )

        return None, 0

    except Exception as e:
        logger.error(f"preprocess_audio_bytes: unexpected error: {e}", exc_info=True)
        return None, 0
    finally:
        for path in [tmp_input_path, tmp_wav_path]:
            if path:
                try:
                    os.unlink(path)
                except Exception:
                    pass
