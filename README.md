# 🎙️ Speech Emotion Detection System

> **Real-time Speech Emotion Recognition using Machine Learning**  
> A complete end-to-end ML project: Dataset → Training → Backend API → Web Frontend

---

## 🎯 Project Overview

This is a **complete, real Machine Learning project** that detects human emotions from spoken audio. Unlike demos or mock systems, every prediction comes from a **genuinely trained ML model** (SVM/Random Forest) trained on the **RAVDESS academic dataset**.

### What it does:
1. You open a web page and click a microphone button
2. You speak naturally for a few seconds
3. Your audio is sent to a Python/FastAPI backend
4. The backend applies the same preprocessing and feature extraction used during training
5. The trained ML model predicts the emotion
6. Results are displayed: emotion, confidence %, and probability bars for all 8 classes

---

## 🎭 Supported Emotions (RAVDESS Dataset)

| # | Emotion   | Emoji | RAVDESS Code |
|---|-----------|-------|--------------|
| 1 | Neutral   | 😐    | 01           |
| 2 | Calm      | 😌    | 02           |
| 3 | Happy     | 😊    | 03           |
| 4 | Sad       | 😢    | 04           |
| 5 | Angry     | 😠    | 05           |
| 6 | Fearful   | 😨    | 06           |
| 7 | Disgust   | 🤢    | 07           |
| 8 | Surprised | 😲    | 08           |

---

## 📦 Dataset

**RAVDESS** — Ryerson Audio-Visual Database of Emotional Speech and Song

| Property    | Value |
|-------------|-------|
| Speakers    | 24 professional actors (12M, 12F) |
| Total files | 1440 speech audio files |
| Emotions    | 8 classes |
| License     | CC BY 4.0 (free for academic use) |
| Source      | https://zenodo.org/record/1188976 |

### Download Instructions:
1. Go to: https://zenodo.org/record/1188976
2. Download: `Audio_Speech_Actors_01-24.zip` (~215 MB)
3. Place the ZIP in: `data/raw/RAVDESS/`
4. Run: `python src/download_dataset.py`

---

## 🧠 ML Pipeline

```
Raw Audio Files (WAV)
       ↓
Load (mono, 22050 Hz)
       ↓
Remove Silence
       ↓
Peak Normalize
       ↓
Pad/Trim to 3 seconds
       ↓
Feature Extraction (192 features):
  • MFCC (40 coefficients) → mean + std = 80 values
  • MFCC Delta (rate of change) → 80 values
  • Zero Crossing Rate → 2 values
  • RMS Energy → 2 values
  • Spectral Centroid → 2 values
  • Spectral Rolloff → 2 values
  • Chroma (12 bins) → mean + std = 24 values
       ↓
StandardScaler normalization
       ↓
ML Classification Model (SVM / Random Forest / MLP)
       ↓
Emotion Label + Probability Distribution
```

---

## 🏗️ Project Structure

```
speech-emotion-detection/
├── data/
│   ├── raw/RAVDESS/          ← Place RAVDESS dataset here
│   ├── processed/            ← Intermediate processing
│   └── features/             ← Cached feature arrays (.npz)
├── models/
│   ├── speech_emotion_model.pkl   ← Trained model
│   ├── scaler.pkl                 ← Feature scaler
│   ├── label_encoder.pkl          ← Label encoder
│   └── model_metadata.json        ← Model info + metrics
├── results/
│   ├── confusion_matrix.png       ← Evaluation visualization
│   ├── classification_report.txt  ← Per-class metrics
│   ├── metrics.json               ← All metrics
│   ├── model_comparison.csv       ← Model comparison table
│   ├── emotion_distribution.png   ← Dataset analysis
│   └── training.log               ← Training log
├── src/
│   ├── config.py                  ← All configuration
│   ├── preprocessing.py           ← Audio preprocessing pipeline
│   ├── feature_extraction.py      ← Feature extraction (192 features)
│   ├── dataset.py                 ← RAVDESS loading + speaker split
│   ├── analyze_dataset.py         ← Dataset analysis + visualizations
│   ├── train.py                   ← Training pipeline
│   └── download_dataset.py        ← Dataset setup helper
├── backend/
│   └── main.py                    ← FastAPI backend
├── frontend/
│   ├── index.html                 ← Web UI
│   ├── style.css                  ← Styling
│   └── app.js                     ← Frontend logic
├── tests/
│   └── test_pipeline.py           ← Test suite
├── docs/                          ← Documentation and reports
├── requirements.txt
├── start.py                       ← Python launcher
├── run.bat                        ← Windows batch launcher
└── README.md
```

---

## 🚀 Installation & Running

### Prerequisites
- Python 3.10+ (tested on 3.14.3)
- ~500 MB free disk space for dataset + model
- A microphone (for live prediction)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Download Dataset
```bash
python src/download_dataset.py
```
Follow the printed instructions to download RAVDESS from Zenodo.

### Step 3: (Optional) Analyze Dataset
```bash
python src/analyze_dataset.py
```
Generates charts in `results/`.

### Step 4: Train the Model
```bash
python src/train.py
```
This takes 10–30 minutes. Trains 4 models, selects the best, saves everything to `models/`.

### Step 5: Start the Backend
```bash
python backend/main.py
```
Backend runs at: http://127.0.0.1:8000  
API docs at: http://127.0.0.1:8000/docs

### Step 6: Open the Frontend
Open `frontend/index.html` in your browser (Chrome/Edge recommended).

---

## 🪟 Windows Quick Start

```batch
# One-time setup
run.bat setup

# Train model (once)
run.bat train

# Start backend (keep this terminal open)
run.bat backend

# Open frontend (in another terminal)
run.bat frontend

# Or run everything at once
run.bat all
```

---

## 🧪 Running Tests

```bash
python -m pytest tests/ -v
```

Or using the launcher:
```bash
python start.py test
```

Tests cover:
- Audio loading and preprocessing
- Feature extraction correctness
- Feature dimension verification
- Model loading (if trained)
- Backend API endpoints
- Dataset parsing
- Error handling

---

## 🔌 Backend API

### `GET /health`
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_type": "SVM_RBF",
  "emotion_classes": ["Angry", "Calm", "Disgust", "Fearful", "Happy", "Neutral", "Sad", "Surprised"],
  "test_accuracy": 0.78
}
```

### `POST /predict`
Send a multipart form with an `audio` file (WAV, WebM, OGG, FLAC).
```json
{
  "emotion": "Happy",
  "confidence": 0.87,
  "probabilities": {
    "Neutral": 0.05,
    "Calm": 0.04,
    "Happy": 0.87,
    "Sad": 0.01,
    "Angry": 0.01,
    "Fearful": 0.01,
    "Disgust": 0.00,
    "Surprised": 0.01
  },
  "timestamp": "2026-09-27T10:30:00"
}
```

### `GET /emotions` — List emotion classes
### `GET /model-info` — Model metadata

---

## 📊 Data Splitting Strategy

To prevent **data leakage**, we use **speaker-independent splitting**:
- Speakers (24 total) are randomly divided into train/val/test groups
- NO speaker appears in more than one split
- This ensures the model is evaluated on truly unseen voices

| Split      | Speakers | ~Files |
|------------|----------|--------|
| Training   | 17       | ~1010  |
| Validation | 4        | ~240   |
| Test       | 3        | ~180   |
| **Random seed** | 42 | — |

---

## 🎨 UI Features

- 🎙️ One-click microphone recording
- 🔴 Live recording timer and waveform animation
- ⏳ "Analyzing..." loading state
- 😊 Dynamic emotion icon (from model output)
- 📊 Animated confidence bar
- 📈 Probability visualization for all 8 emotions
- 🗂️ Prediction history (localStorage, no data stored on server)
- 🔒 Privacy-first: audio not permanently stored
- 📱 Responsive design (desktop, tablet, mobile)

---

## ⚠️ Known Limitations

1. **Probabilistic, not certain** — emotion detection is an estimate, not a ground truth
2. **English + professional actors** — model may not generalize well to all accents/languages
3. **Background noise** — affects prediction accuracy significantly
4. **Short recordings** — less than 2 seconds may yield unreliable results
5. **Microphone quality** — low-quality mics reduce accuracy
6. **Not a clinical tool** — do not use for psychological diagnosis

---

## 🔮 Future Enhancements

- Deep learning (CNN/LSTM on mel spectrograms)
- Transformer-based models (wav2vec 2.0, HuBERT)
- Real-time streaming prediction
- Multilingual dataset support
- Noise-robust preprocessing (spectral gating)
- Mobile application (React Native / Flutter)
- Cloud deployment (Docker + cloud hosting)
- Speaker adaptation fine-tuning

---

## 🔒 Privacy

> The microphone is used **only** when you click "Start Recording."  
> Audio is sent to the local application for emotion prediction and is **not permanently stored**.  
> No data is uploaded to external servers.

---

## 📚 References

1. Livingstone SR, Russo FA (2018). *The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)*. PLoS ONE 13(5): e0196391. https://doi.org/10.1371/journal.pone.0196391

2. Davis SB, Mermelstein P (1980). *Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences*. IEEE TASSP, 28(4):357-366.

3. Schuller B, et al. (2013). *The INTERSPEECH 2013 computational paralinguistics challenge: social signals, conflict, emotion, autism*. INTERSPEECH.

4. librosa: McFee et al. (2015). *librosa: Audio and Music Signal Analysis in Python*. SciPy.

5. scikit-learn: Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR 12:2825-2830.

---

## 👨‍🎓 Academic Details

- **Project type**: B.E. Computer Science — Machine Learning Project
- **Dataset**: RAVDESS (CC BY 4.0)
- **ML approach**: Classical supervised classification (SVM, Random Forest, MLP)
- **Features**: MFCC + delta + spectral features (192-dimensional vector)
- **Backend**: Python/FastAPI
- **Frontend**: HTML/CSS/JavaScript (Vanilla)
- **Random seed**: 42 (for reproducibility)
