"""
dataset.py - Dataset loading and analysis for RAVDESS Speech Emotion Dataset.

RAVDESS Dataset:
- Full name: Ryerson Audio-Visual Database of Emotional Speech and Song
- URL: https://zenodo.org/record/1188976
- License: Creative Commons Attribution (CC BY 4.0)
- Speakers: 24 professional actors (12 male, 12 female)
- Emotions: 8 (Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised)
- Files: 1440 audio files (speech only, two intensity levels)

File naming convention:
  Modality-VocalChannel-Emotion-Intensity-Statement-Repetition-Actor.wav
  Example: 03-01-05-01-01-01-01.wav
  Position 3 (0-indexed 2) = Emotion code (01-08)
  Position 7 = Actor ID (01-24)

Dataset split strategy:
- We use SPEAKER-INDEPENDENT splitting.
- Speakers are randomly divided into train/val/test groups.
- This prevents data leakage: the model never sees a training speaker
  in the test set, making evaluation more realistic.
- Split: 70% train / 15% val / 15% test (approximately)
- Train speakers: 17, Val speakers: 4, Test speakers: 3 (out of 24)
"""

import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    RAW_DIR, RAVDESS_EMOTION_MAP, EMOTION_CLASSES, RANDOM_SEED,
    TEST_SIZE, VAL_SIZE
)

logger = logging.getLogger(__name__)


def parse_ravdess_filename(filepath: Path) -> Optional[Dict]:
    """
    Parse a RAVDESS filename and return metadata.

    RAVDESS naming: Modality-VocalChannel-Emotion-Intensity-Statement-Repetition-Actor.wav
    Segments are separated by '-'.

    Returns dict with: emotion_code, emotion_label, actor_id, intensity, filepath
    Returns None if filename doesn't match expected RAVDESS pattern.
    """
    try:
        parts = filepath.stem.split('-')
        if len(parts) != 7:
            return None

        modality     = parts[0]   # 01=audio-only, 02=audio-video
        vocal_channel = parts[1]  # 01=speech, 02=song
        emotion_code = parts[2]   # 01-08
        intensity    = parts[3]   # 01=normal, 02=strong
        statement    = parts[4]   # 01="Kids are talking...", 02="Dogs are sitting..."
        repetition   = parts[5]   # 01 or 02
        actor_id     = int(parts[6])  # 01-24

        # Only use speech modality (not song)
        if vocal_channel != "01":
            return None

        emotion_label = RAVDESS_EMOTION_MAP.get(emotion_code)
        if emotion_label is None:
            logger.warning(f"Unknown emotion code {emotion_code} in {filepath.name}")
            return None

        return {
            "filepath":     str(filepath),
            "emotion_code": emotion_code,
            "emotion":      emotion_label,
            "actor_id":     actor_id,
            "intensity":    intensity,
            "statement":    statement,
            "repetition":   repetition,
            "modality":     modality,
        }
    except Exception as e:
        logger.debug(f"Could not parse {filepath.name}: {e}")
        return None


def load_dataset_info(data_dir: Path = RAW_DIR) -> pd.DataFrame:
    """
    Scan the RAVDESS directory and build a DataFrame with file metadata.

    RAVDESS structure (after extraction):
      data/raw/RAVDESS/
        Actor_01/
          03-01-01-01-01-01-01.wav
          03-01-02-01-01-01-01.wav
          ...
        Actor_02/
          ...

    Returns:
        DataFrame with columns: filepath, emotion, actor_id, intensity, etc.
        Returns empty DataFrame if dataset not found.
    """
    audio_files = list(data_dir.rglob("*.wav"))

    if not audio_files:
        logger.error(
            f"No .wav files found in {data_dir}.\n"
            f"Please download RAVDESS from: https://zenodo.org/record/1188976\n"
            f"and extract to: {data_dir}"
        )
        return pd.DataFrame()

    records = []
    skipped = 0
    for fpath in audio_files:
        info = parse_ravdess_filename(fpath)
        if info is not None:
            records.append(info)
        else:
            skipped += 1

    if skipped > 0:
        logger.info(f"Skipped {skipped} files (songs or unrecognized format).")

    df = pd.DataFrame(records)
    logger.info(f"Dataset loaded: {len(df)} audio files across {df['actor_id'].nunique()} speakers.")
    return df


def get_speaker_split(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
    random_seed: int = RANDOM_SEED,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Split speakers into train/val/test groups (speaker-independent split).

    Why speaker-independent?
    - If speaker 5's files appear in both train and test, the model
      learns that speaker's voice pattern, not the emotion.
    - Speaker-independent split forces generalization across voices.

    Returns:
        (train_actor_ids, val_actor_ids, test_actor_ids)
    """
    rng = np.random.default_rng(random_seed)
    actor_ids = sorted(df['actor_id'].unique())
    n_actors = len(actor_ids)

    shuffled = list(rng.permutation(actor_ids))

    n_test = max(1, round(n_actors * test_size))
    n_val  = max(1, round(n_actors * val_size))
    n_train = n_actors - n_test - n_val

    train_actors = shuffled[:n_train]
    val_actors   = shuffled[n_train:n_train + n_val]
    test_actors  = shuffled[n_train + n_val:]

    logger.info(
        f"Speaker split -> Train: {sorted(train_actors)} ({len(train_actors)} speakers), "
        f"Val: {sorted(val_actors)} ({len(val_actors)} speakers), "
        f"Test: {sorted(test_actors)} ({len(test_actors)} speakers)"
    )
    return train_actors, val_actors, test_actors


def split_dataset(
    df: pd.DataFrame,
    random_seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into train/val/test DataFrames using speaker-independent split.

    Returns:
        (train_df, val_df, test_df)
    """
    train_actors, val_actors, test_actors = get_speaker_split(df, random_seed=random_seed)

    train_df = df[df['actor_id'].isin(train_actors)].reset_index(drop=True)
    val_df   = df[df['actor_id'].isin(val_actors)].reset_index(drop=True)
    test_df  = df[df['actor_id'].isin(test_actors)].reset_index(drop=True)

    logger.info(
        f"Split sizes -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}"
    )
    return train_df, val_df, test_df


def verify_dataset(data_dir: Path = RAW_DIR) -> bool:
    """
    Check whether RAVDESS dataset exists and has the expected structure.
    Returns True if dataset is ready, False otherwise.
    """
    if not data_dir.exists():
        return False

    wav_files = list(data_dir.rglob("*.wav"))
    if len(wav_files) < 100:
        logger.warning(
            f"Found only {len(wav_files)} .wav files in {data_dir}. "
            f"RAVDESS should have ~1440 speech files."
        )
        return False

    return True
