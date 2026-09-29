"""
download_dataset.py - Helper script to guide RAVDESS dataset download.

RAVDESS Dataset Information:
  Name:    Ryerson Audio-Visual Database of Emotional Speech and Song
  URL:     https://zenodo.org/record/1188976
  License: Creative Commons Attribution (CC BY 4.0)
  Size:    ~24.8 GB (full), ~1.5 GB (audio-speech only files)

This script:
1. Checks if the dataset already exists
2. Provides download instructions
3. Verifies the dataset after download
4. Lists what was found
"""

import sys
import zipfile
import json
import logging
import urllib.request
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from config import RAW_DIR
from dataset import load_dataset_info, verify_dataset


def print_banner():
    print("\n" + "="*70)
    print(" 🎙️  RAVDESS Dataset Setup for Speech Emotion Detection")
    print("="*70 + "\n")


def check_existing():
    """Check what's already in the data directory."""
    wav_files = list(RAW_DIR.rglob("*.wav"))
    zip_files = list(RAW_DIR.glob("*.zip")) + list(RAW_DIR.parent.glob("*.zip"))
    return wav_files, zip_files


def auto_extract_zip(zip_path: Path):
    """Extract a RAVDESS zip file to the correct location."""
    logger.info(f"Extracting {zip_path.name}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            members = zf.namelist()
            logger.info(f"  Found {len(members)} files in zip")
            zf.extractall(RAW_DIR)
        logger.info(f"  ✅ Extracted to {RAW_DIR}")
        return True
    except Exception as e:
        logger.error(f"  ❌ Extraction failed: {e}")
        return False


def download_ravdess_speech(dest_zip: Path):
    """Download Audio_Speech_Actors_01-24.zip directly from Zenodo API."""
    url = "https://zenodo.org/api/records/1188976/files/Audio_Speech_Actors_01-24.zip/content"
    logger.info("Attempting automatic download of Audio_Speech_Actors_01-24.zip (~208 MB)...")
    logger.info(f"URL: {url}")
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            total_size = int(resp.headers.get('Content-Length', 0))
            logger.info(f"Connected. Total size: {total_size / (1024 * 1024):.1f} MB")
            downloaded = 0
            chunk_size = 1024 * 1024  # 1MB chunks
            last_percent = -1
            with open(dest_zip, 'wb') as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = int(downloaded * 100 / total_size)
                        if pct % 10 == 0 and pct != last_percent:
                            logger.info(f"  Downloading: {pct}% ({downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB)")
                            last_percent = pct
            logger.info("✅ Download complete!")
            return True
    except Exception as e:
        logger.error(f"Download failed: {e}")
        if dest_zip.exists():
            dest_zip.unlink()
        return False


def main():
    print_banner()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    wav_files, zip_files = check_existing()

    # Case 1: Dataset already present
    if len(wav_files) >= 100:
        logger.info(f"✅ Dataset appears to be present: {len(wav_files)} .wav files found")
        df = load_dataset_info(RAW_DIR)
        if not df.empty:
            logger.info(f"   Speakers: {df['actor_id'].nunique()}")
            logger.info(f"   Emotions: {sorted(df['emotion'].unique().tolist())}")
            logger.info(f"   Total speech files: {len(df)}")
            logger.info("\n✅ Dataset is ready! You can now run training:")
            logger.info("   python src/train.py")
        return

    # Case 2: ZIP files found — try to extract
    if zip_files:
        logger.info(f"Found {len(zip_files)} zip file(s). Attempting extraction...")
        for zf in zip_files:
            auto_extract_zip(zf)

        wav_files_after = list(RAW_DIR.rglob("*.wav"))
        if len(wav_files_after) >= 100:
            logger.info("✅ Extraction successful!")
            main()  # Re-check
            return

    # Case 3: Try automatic download
    target_zip = RAW_DIR / "Audio_Speech_Actors_01-24.zip"
    if download_ravdess_speech(target_zip):
        if auto_extract_zip(target_zip):
            wav_files_after = list(RAW_DIR.rglob("*.wav"))
            if len(wav_files_after) >= 100:
                logger.info("✅ Download and extraction successful!")
                main()
                return

    # Case 4: Need manual download fallback
    print("\n" + "─"*70)
    print("  📥  MANUAL DOWNLOAD REQUIRED")
    print("─"*70)
    print("""
RAVDESS dataset requires a free download from Zenodo.

STEP 1: Go to this URL:
  https://zenodo.org/record/1188976

STEP 2: Download ONLY the speech files (NOT the video/song files):
  File: Audio_Speech_Actors_01-24.zip
  Size: ~208 MB

  Direct download link:
  https://zenodo.org/api/records/1188976/files/Audio_Speech_Actors_01-24.zip/content

STEP 3: Place the downloaded ZIP file in:
  """ + str(RAW_DIR) + """

STEP 4: Run this script again:
  python src/download_dataset.py

  OR extract manually and ensure structure is:
  data/raw/RAVDESS/
    Actor_01/
      03-01-01-01-01-01-01.wav
      03-01-02-01-01-01-01.wav
      ...
    Actor_02/
      ...
    ...
    Actor_24/
      ...

ABOUT THE DATASET:
  Name:     RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
  Speakers: 24 professional actors (12 male, 12 female)
  Emotions: 8 (Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised)
  Files:    1440 speech audio files
  License:  CC BY 4.0 (free for academic/research use with attribution)
  Citation: Livingstone SR, Russo FA (2018) The Ryerson Audio-Visual Database of
            Emotional Speech and Song (RAVDESS). PLoS ONE 13(5): e0196391.
""")
    print("─"*70)


def verify_and_report():
    """Verify dataset and print statistics."""
    df = load_dataset_info(RAW_DIR)
    if df.empty:
        logger.error("Dataset not found or not recognized.")
        return False

    print("\n📊 Dataset Statistics:")
    print(f"  Total files:  {len(df)}")
    print(f"  Speakers:     {df['actor_id'].nunique()}")
    print(f"  Emotions:     {df['emotion'].nunique()}")
    print("\n  Samples per emotion:")
    for emotion, count in sorted(df['emotion'].value_counts().items()):
        bar = "█" * (count // 10)
        print(f"    {emotion:12s}: {count:3d}  {bar}")
    return True


if __name__ == "__main__":
    main()
    if verify_dataset(RAW_DIR):
        verify_and_report()
