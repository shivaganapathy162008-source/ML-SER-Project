"""
analyze_dataset.py - Dataset analysis and visualization for RAVDESS.

Run this AFTER placing the RAVDESS dataset in data/raw/RAVDESS/.

This script generates:
  results/emotion_distribution.png
  results/duration_distribution.png
  results/sample_waveform.png
  results/sample_spectrogram.png
  results/dataset_stats.json
"""

import sys
import json
import logging

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import librosa
import librosa.display
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RAW_DIR, RESULTS_DIR, SAMPLE_RATE
from dataset import load_dataset_info

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Use a clean matplotlib style
plt.rcParams.update({
    'figure.facecolor': '#1a1a2e',
    'axes.facecolor': '#16213e',
    'axes.labelcolor': 'white',
    'text.color': 'white',
    'xtick.color': 'white',
    'ytick.color': 'white',
    'grid.color': '#2d4059',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

EMOTION_COLORS = {
    'Neutral':   '#95a5a6',
    'Calm':      '#3498db',
    'Happy':     '#f1c40f',
    'Sad':       '#2980b9',
    'Angry':     '#e74c3c',
    'Fearful':   '#9b59b6',
    'Disgust':   '#27ae60',
    'Surprised': '#e67e22',
}


def get_audio_durations(df: pd.DataFrame) -> pd.Series:
    """Get duration (seconds) for each audio file."""
    durations = []
    for _, row in df.iterrows():
        try:
            duration = librosa.get_duration(path=row['filepath'])
            durations.append(duration)
        except Exception:
            durations.append(np.nan)
    return pd.Series(durations, index=df.index)


def plot_emotion_distribution(df: pd.DataFrame, save_path: Path):
    """Bar chart showing how many samples exist per emotion."""
    emotion_counts = df['emotion'].value_counts()
    # Reorder to standard order
    ordered_emotions = [e for e in ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']
                        if e in emotion_counts.index]
    emotion_counts = emotion_counts.reindex(ordered_emotions)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = [EMOTION_COLORS.get(e, '#ffffff') for e in emotion_counts.index]
    bars = ax.bar(emotion_counts.index, emotion_counts.values, color=colors, edgecolor='none', alpha=0.9)

    ax.set_title('RAVDESS Dataset — Emotion Distribution', fontsize=16, fontweight='bold', color='white', pad=20)
    ax.set_xlabel('Emotion', fontsize=13)
    ax.set_ylabel('Number of Audio Files', fontsize=13)
    ax.set_ylim(0, emotion_counts.max() * 1.2)
    ax.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars, emotion_counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            str(val), ha='center', va='bottom', fontsize=12, color='white', fontweight='bold'
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    logger.info(f"Saved: {save_path}")


def plot_duration_distribution(df: pd.DataFrame, save_path: Path):
    """Histogram of audio clip durations."""
    logger.info("Computing audio durations (this may take a minute)...")
    durations = get_audio_durations(df)
    durations = durations.dropna()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(durations, bins=30, color='#3498db', edgecolor='none', alpha=0.85)
    axes[0].set_title('Audio Duration Distribution', fontsize=14, color='white')
    axes[0].set_xlabel('Duration (seconds)', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].axvline(durations.mean(), color='#f1c40f', linestyle='--', linewidth=2, label=f'Mean: {durations.mean():.2f}s')
    axes[0].axvline(durations.median(), color='#e74c3c', linestyle='--', linewidth=2, label=f'Median: {durations.median():.2f}s')
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)

    # Box plot by emotion
    emotion_durations = {}
    for emotion in df['emotion'].unique():
        mask = df['emotion'] == emotion
        emotion_durations[emotion] = durations[mask].values

    box_data = [emotion_durations.get(e, []) for e in ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']
                if e in emotion_durations]
    box_labels = [e for e in ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']
                  if e in emotion_durations]

    try:
        bp = axes[1].boxplot(box_data, tick_labels=box_labels, patch_artist=True)
    except TypeError:
        bp = axes[1].boxplot(box_data, labels=box_labels, patch_artist=True)
    for patch, label in zip(bp['boxes'], box_labels):
        patch.set_facecolor(EMOTION_COLORS.get(label, '#ffffff'))
        patch.set_alpha(0.8)
    axes[1].set_title('Duration by Emotion', fontsize=14, color='white')
    axes[1].set_xlabel('Emotion', fontsize=12)
    axes[1].set_ylabel('Duration (seconds)', fontsize=12)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    logger.info(f"Saved: {save_path}")

    return durations


def plot_sample_waveform(df: pd.DataFrame, save_path: Path):
    """Plot sample waveforms for selected emotions."""
    emotions_to_show = ['Happy', 'Angry', 'Sad', 'Neutral']
    available = [e for e in emotions_to_show if e in df['emotion'].values]
    if not available:
        available = df['emotion'].unique()[:4]

    fig, axes = plt.subplots(len(available), 1, figsize=(14, 3 * len(available)))
    if len(available) == 1:
        axes = [axes]

    for ax, emotion in zip(axes, available):
        sample = df[df['emotion'] == emotion].iloc[0]
        try:
            y, sr = librosa.load(sample['filepath'], sr=SAMPLE_RATE, mono=True)
            time = np.linspace(0, len(y) / sr, len(y))
            color = EMOTION_COLORS.get(emotion, '#3498db')
            ax.plot(time, y, color=color, linewidth=0.5, alpha=0.9)
            ax.set_title(f'{emotion} — Waveform', fontsize=12, color='white')
            ax.set_xlabel('Time (s)', fontsize=10)
            ax.set_ylabel('Amplitude', fontsize=10)
            ax.set_xlim(0, max(time))
            ax.grid(alpha=0.2)
        except Exception as e:
            ax.text(0.5, 0.5, f'Error loading {emotion}', ha='center', transform=ax.transAxes)

    plt.suptitle('Sample Waveforms by Emotion', fontsize=15, fontweight='bold', color='white', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    logger.info(f"Saved: {save_path}")


def plot_sample_spectrogram(df: pd.DataFrame, save_path: Path):
    """Plot Mel spectrogram for one sample per emotion (up to 4)."""
    emotions_to_show = ['Happy', 'Angry', 'Sad', 'Neutral']
    available = [e for e in emotions_to_show if e in df['emotion'].values]
    if not available:
        available = df['emotion'].unique()[:4]

    fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 5))
    if len(available) == 1:
        axes = [axes]

    for ax, emotion in zip(axes, available):
        sample = df[df['emotion'] == emotion].iloc[0]
        try:
            y, sr = librosa.load(sample['filepath'], sr=SAMPLE_RATE, mono=True)
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            S_db = librosa.power_to_db(S, ref=np.max)
            img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', ax=ax, cmap='magma')
            ax.set_title(f'{emotion}', fontsize=13, color='white', fontweight='bold')
            ax.set_xlabel('Time (s)', fontsize=10)
            ax.set_ylabel('Mel Frequency' if emotion == available[0] else '', fontsize=10)
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {emotion}', ha='center', transform=ax.transAxes, color='white')

    plt.suptitle('Mel Spectrograms by Emotion', fontsize=15, fontweight='bold', color='white')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    logger.info(f"Saved: {save_path}")


def run_analysis():
    """Main analysis function."""
    logger.info("=" * 60)
    logger.info("RAVDESS Dataset Analysis")
    logger.info("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load dataset info
    df = load_dataset_info(RAW_DIR)

    if df.empty:
        logger.error("Dataset not found. Please download RAVDESS first.")
        logger.error("Download from: https://zenodo.org/record/1188976")
        logger.error(f"Extract to: {RAW_DIR}")
        return None

    # Basic statistics
    logger.info(f"\nTotal audio files:   {len(df)}")
    logger.info(f"Number of speakers:  {df['actor_id'].nunique()}")
    logger.info(f"Emotion classes:     {sorted(df['emotion'].unique())}")
    logger.info(f"\nSamples per emotion:")
    for emotion, count in df['emotion'].value_counts().items():
        logger.info(f"  {emotion:12s}: {count}")

    # Check for class imbalance
    counts = df['emotion'].value_counts()
    imbalance_ratio = counts.max() / counts.min() if counts.min() > 0 else float('inf')
    logger.info(f"\nClass imbalance ratio (max/min): {imbalance_ratio:.2f}")
    if imbalance_ratio > 1.5:
        logger.warning("Class imbalance detected. Will use class_weight='balanced' during training.")

    # Generate visualizations
    logger.info("\nGenerating visualizations...")
    plot_emotion_distribution(df, RESULTS_DIR / "emotion_distribution.png")

    durations = plot_duration_distribution(df, RESULTS_DIR / "duration_distribution.png")
    plot_sample_waveform(df, RESULTS_DIR / "sample_waveforms.png")
    plot_sample_spectrogram(df, RESULTS_DIR / "sample_spectrograms.png")

    # Save stats to JSON
    stats = {
        "total_files": int(len(df)),
        "n_speakers": int(df['actor_id'].nunique()),
        "emotions": sorted(df['emotion'].unique().tolist()),
        "samples_per_emotion": df['emotion'].value_counts().to_dict(),
        "duration_stats_seconds": {
            "mean":   float(durations.mean()),
            "std":    float(durations.std()),
            "min":    float(durations.min()),
            "max":    float(durations.max()),
            "median": float(durations.median()),
        },
        "imbalance_ratio": float(imbalance_ratio),
    }

    stats_path = RESULTS_DIR / "dataset_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"\nSaved dataset statistics to: {stats_path}")

    logger.info("\n✅ Dataset analysis complete!")
    logger.info(f"Check results in: {RESULTS_DIR}")

    return df


if __name__ == "__main__":
    run_analysis()
