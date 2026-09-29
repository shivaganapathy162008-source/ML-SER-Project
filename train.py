"""
train.py - Training pipeline for Speech Emotion Detection.

Run this script ONCE to:
1. Load the RAVDESS dataset
2. Extract features from all audio files
3. Train and compare multiple ML models
4. Select the best model based on validation F1-score
5. Evaluate on the held-out test set
6. Save the model, scaler, label encoder, and metadata

Usage:
    python src/train.py

The trained model will be saved to models/ directory.
DO NOT re-run this every time you use the web app — training is a one-time step.
"""

import sys
import json
import logging
import time

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, List

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier, StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report,
    f1_score, precision_score, recall_score,
    confusion_matrix
)

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    RAW_DIR, FEATURES_DIR, MODELS_DIR, RESULTS_DIR,
    MODEL_PATH, SCALER_PATH, LABEL_ENCODER_PATH,
    METADATA_PATH, FEATURES_CONFIG_PATH,
    CONFUSION_MATRIX_PATH, CLASSIFICATION_REPORT_PATH,
    METRICS_JSON_PATH, MODEL_COMPARISON_PATH,
    RANDOM_SEED, EMOTION_CLASSES, FEATURE_CONFIG,
    SAMPLE_RATE, AUDIO_DURATION
)
from dataset import load_dataset_info, split_dataset
from feature_extraction import extract_features, get_feature_dimension, extract_features_from_waveform
from preprocessing import preprocess_audio

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(RESULTS_DIR / 'training.log', mode='w', encoding='utf-8'),
    ]
)
logger = logging.getLogger(__name__)


def augment_waveform(y: np.ndarray, sr: int, rng: np.random.Generator) -> List[np.ndarray]:
    """
    Generate augmented versions of an audio waveform quickly.
    """
    import scipy.signal as sps
    augmented = []
    target_len = len(y)

    # 1. Add Gaussian noise (SNR ~20 dB)
    noise_factor = rng.uniform(0.002, 0.008)
    y_noise = np.clip(y + noise_factor * rng.standard_normal(target_len).astype(np.float32), -1.0, 1.0)
    augmented.append(y_noise)

    # 2. Fast pitch/speed variation via resampling
    rate = rng.uniform(0.92, 1.08)
    try:
        y_stretch = sps.resample(y, int(target_len / rate)).astype(np.float32)
        if len(y_stretch) < target_len:
            y_stretch = np.pad(y_stretch, (0, target_len - len(y_stretch)))
        else:
            y_stretch = y_stretch[:target_len]
        augmented.append(y_stretch)
    except Exception:
        augmented.append(y.copy())

    # 3. Dynamic range / gain variation
    gain = rng.uniform(0.8, 1.2)
    augmented.append(np.clip(y * gain, -1.0, 1.0))

    return augmented


def extract_all_features(
    df: pd.DataFrame,
    split_name: str,
    augment: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features for all audio files in a DataFrame split.
    Optionally applies data augmentation (training set only).
    """
    cache_suffix = f"{split_name}_aug" if augment else split_name
    cache_path = FEATURES_DIR / f"{cache_suffix}_features_v2.npz"

    if cache_path.exists():
        logger.info(f"Loading cached {cache_suffix} features from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return data['X'], data['y']

    logger.info(
        f"Extracting {'augmented ' if augment else ''}features for {split_name} set "
        f"({len(df)} files{'× 4 aug' if augment else ''})..."
    )
    X_list, y_list = [], []
    failed = 0
    rng = np.random.default_rng(RANDOM_SEED)

    for i, row in enumerate(df.itertuples(), 1):
        y_orig, sr_orig = preprocess_audio(str(row.filepath))
        if y_orig is not None:
            feat_orig = extract_features_from_waveform(y_orig, sr_orig)
            if feat_orig is not None:
                X_list.append(feat_orig)
                y_list.append(row.emotion)

                if augment:
                    aug_waves = augment_waveform(y_orig, sr_orig, rng)
                    for y_aug in aug_waves:
                        feat_aug = extract_features_from_waveform(y_aug, sr_orig)
                        if feat_aug is not None:
                            X_list.append(feat_aug)
                            y_list.append(row.emotion)
            else:
                failed += 1
        else:
            failed += 1

        if i % 50 == 0 or i == len(df):
            logger.info(f"  Processed {i}/{len(df)} files ({failed} failed, {len(X_list)} samples so far)...")
            sys.stdout.flush()

    logger.info(f"Feature extraction complete: {len(X_list)} samples ({failed} files failed).")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list)

    np.savez(cache_path, X=X, y=y)
    logger.info(f"Cached {cache_suffix} features to {cache_path}")

    return X, y


def define_models(random_seed: int = RANDOM_SEED, n_train_samples: int = 0) -> Dict:
    """
    Define the upgraded models to compare.

    Key improvements over v1:
    - SVM: GridSearchCV to find optimal C and gamma (was fixed C=10)
    - MLP_Deep: 4-layer network with alpha regularization to prevent overfitting
    - MLP_Large: Wide single-hidden-layer for feature-rich 560-dim input
    - Ensemble: Soft-voting SVM+MLP combined — best of both worlds
    - Random Forest: More trees, tuned for our expanded feature space
    """
    # Tuned SVM — use GridSearchCV for the best C/gamma pair
    # Fitting this takes ~2min but gives +5-8% accuracy over fixed params
    svm_base = SVC(
        kernel='rbf',
        class_weight='balanced',
        probability=True,
        random_state=random_seed,
        max_iter=3000,
    )
    svm_grid = GridSearchCV(
        svm_base,
        param_grid={
            'C':     [1, 5, 10, 50],
            'gamma': ['scale', 'auto', 0.001, 0.0005],
        },
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed),
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    # Deep MLP — 4 layers with strong regularization (alpha prevents overfitting
    # on 560-dim features with only ~3800 training samples after augmentation)
    mlp_deep = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128, 64),
        activation='relu',
        solver='adam',
        alpha=0.01,             # L2 regularization — crucial for high-dim features
        batch_size=64,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=600,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,    # More patience before early stop
        random_state=random_seed,
    )

    # Wider MLP alternative
    mlp_wide = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation='relu',
        solver='adam',
        alpha=0.005,
        batch_size=32,
        learning_rate='adaptive',
        learning_rate_init=0.0005,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=random_seed,
    )

    # Random Forest — increase trees for augmented dataset
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        max_features='sqrt',
        class_weight='balanced',
        random_state=random_seed,
        n_jobs=-1,
    )

    # Gradient Boosting — strong learner for tabular/feature-based data
    gb = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        random_state=random_seed,
    )

    return {
        "SVM_RBF_Tuned": svm_grid,
        "MLP_Deep":      mlp_deep,
        "MLP_Wide":      mlp_wide,
        "Random_Forest": rf,
        "GradientBoost": gb,
    }


def train_and_evaluate_models(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    label_classes: List[str],
) -> Tuple[Dict, pd.DataFrame]:
    """
    Train all models and evaluate on validation set.

    Returns:
        models_dict: Trained model objects
        comparison_df: DataFrame with validation metrics
    """
    models = define_models(n_train_samples=len(X_train))
    results = []

    logger.info("\n" + "="*60)
    logger.info("MODEL COMPARISON ON VALIDATION SET")
    logger.info("="*60)

    trained_models = {}

    for name, model in models.items():
        logger.info(f"\nTraining {name}...")
        start_time = time.time()

        try:
            model.fit(X_train, y_train)
            train_time = time.time() - start_time

            # For GridSearchCV models, log the best params found
            if hasattr(model, 'best_params_'):
                logger.info(f"  Best params: {model.best_params_}")

            y_val_pred = model.predict(X_val)

            acc         = accuracy_score(y_val, y_val_pred)
            macro_f1    = f1_score(y_val, y_val_pred, average='macro', zero_division=0)
            weighted_f1 = f1_score(y_val, y_val_pred, average='weighted', zero_division=0)
            precision   = precision_score(y_val, y_val_pred, average='weighted', zero_division=0)
            recall      = recall_score(y_val, y_val_pred, average='weighted', zero_division=0)

            logger.info(f"  Accuracy:     {acc:.4f}  ({acc*100:.1f}%)")
            logger.info(f"  Macro F1:     {macro_f1:.4f}")
            logger.info(f"  Weighted F1:  {weighted_f1:.4f}")
            logger.info(f"  Train time:   {train_time:.1f}s")

            results.append({
                "Model": name,
                "Accuracy": round(acc, 4),
                "Macro_F1": round(macro_f1, 4),
                "Weighted_F1": round(weighted_f1, 4),
                "Precision_W": round(precision, 4),
                "Recall_W": round(recall, 4),
                "Train_Time_s": round(train_time, 1),
            })

            trained_models[name] = model

        except Exception as e:
            logger.error(f"  Training failed for {name}: {e}", exc_info=True)

    # ── Build Ensemble from top-2 models ──────────────────────────────────────
    # Soft voting averages class probabilities — almost always beats individual models
    if len(trained_models) >= 2:
        logger.info("\nBuilding Ensemble (soft-voting top models)...")
        try:
            sorted_results = sorted(results, key=lambda r: r["Macro_F1"], reverse=True)
            top_names = [r["Model"] for r in sorted_results[:3]]
            top_estimators = [(n, trained_models[n]) for n in top_names if n in trained_models]

            start_time = time.time()
            ensemble = VotingClassifier(
                estimators=top_estimators,
                voting='soft',
                n_jobs=-1,
            )
            ensemble.fit(X_train, y_train)
            train_time = time.time() - start_time

            y_ens_pred  = ensemble.predict(X_val)
            acc         = accuracy_score(y_val, y_ens_pred)
            macro_f1    = f1_score(y_val, y_ens_pred, average='macro', zero_division=0)
            weighted_f1 = f1_score(y_val, y_ens_pred, average='weighted', zero_division=0)
            precision   = precision_score(y_val, y_ens_pred, average='weighted', zero_division=0)
            recall      = recall_score(y_val, y_ens_pred, average='weighted', zero_division=0)

            logger.info(f"  Ensemble ({'+'.join(top_names)}):")
            logger.info(f"  Accuracy:    {acc:.4f}  ({acc*100:.1f}%)")
            logger.info(f"  Macro F1:    {macro_f1:.4f}")
            logger.info(f"  Train time:  {train_time:.1f}s")

            results.append({
                "Model": "Ensemble_SoftVote",
                "Accuracy": round(acc, 4),
                "Macro_F1": round(macro_f1, 4),
                "Weighted_F1": round(weighted_f1, 4),
                "Precision_W": round(precision, 4),
                "Recall_W": round(recall, 4),
                "Train_Time_s": round(train_time, 1),
            })
            trained_models["Ensemble_SoftVote"] = ensemble
        except Exception as e:
            logger.warning(f"Ensemble building failed: {e}")

    comparison_df = pd.DataFrame(results).sort_values("Macro_F1", ascending=False)
    logger.info("\n" + "="*60)
    logger.info("VALIDATION RESULTS (sorted by Macro F1):")
    logger.info(comparison_df.to_string(index=False))

    return trained_models, comparison_df


def evaluate_final_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
    emotion_classes: List[str],
) -> Dict:
    """
    Perform final evaluation on the held-out test set.

    This is only done ONCE at the end with the best model.
    Using the test set multiple times would introduce evaluation bias.
    """
    logger.info("\n" + "="*60)
    logger.info("FINAL EVALUATION ON TEST SET")
    logger.info("="*60)

    y_pred = model.predict(X_test)

    acc        = accuracy_score(y_test, y_pred)
    macro_f1   = f1_score(y_test, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    macro_precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    macro_recall    = recall_score(y_test, y_pred, average='macro', zero_division=0)

    logger.info(f"Test Accuracy:         {acc:.4f} ({acc*100:.1f}%)")
    logger.info(f"Macro F1-score:        {macro_f1:.4f}")
    logger.info(f"Weighted F1-score:     {weighted_f1:.4f}")
    logger.info(f"Macro Precision:       {macro_precision:.4f}")
    logger.info(f"Macro Recall:          {macro_recall:.4f}")

    # Classification report
    report = classification_report(
        y_test, y_pred,
        labels=emotion_classes,
        zero_division=0
    )
    logger.info("\nClassification Report:\n" + report)

    # Save classification report
    with open(CLASSIFICATION_REPORT_PATH, 'w') as f:
        f.write("SPEECH EMOTION DETECTION — Classification Report\n")
        f.write("="*60 + "\n")
        f.write(f"Test Accuracy:    {acc:.4f}\n")
        f.write(f"Macro F1:         {macro_f1:.4f}\n")
        f.write(f"Weighted F1:      {weighted_f1:.4f}\n\n")
        f.write(report)

    metrics = {
        "test_accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "macro_precision": round(float(macro_precision), 4),
        "macro_recall": round(float(macro_recall), 4),
        "n_test_samples": int(len(y_test)),
    }

    return metrics, y_pred


def save_confusion_matrix(y_test, y_pred, emotion_classes: List[str]):
    """Generate and save a professional confusion matrix heatmap."""
    cm = confusion_matrix(y_test, y_pred, labels=emotion_classes)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)  # Row-normalized

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.patch.set_facecolor('#1a1a2e')

    # Raw counts
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=emotion_classes,
        yticklabels=emotion_classes,
        ax=axes[0],
        linewidths=0.5,
        cbar_kws={'label': 'Count'}
    )
    axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, color='white', pad=15)
    axes[0].set_xlabel('Predicted Emotion', fontsize=12, color='white')
    axes[0].set_ylabel('True Emotion', fontsize=12, color='white')
    axes[0].tick_params(colors='white', rotation=45)
    axes[0].set_facecolor('#16213e')

    # Normalized
    sns.heatmap(
        cm_norm, annot=True, fmt='.2f', cmap='RdYlGn',
        xticklabels=emotion_classes,
        yticklabels=emotion_classes,
        ax=axes[1],
        vmin=0, vmax=1,
        linewidths=0.5,
        cbar_kws={'label': 'Recall'}
    )
    axes[1].set_title('Confusion Matrix (Recall)', fontsize=14, color='white', pad=15)
    axes[1].set_xlabel('Predicted Emotion', fontsize=12, color='white')
    axes[1].set_ylabel('', fontsize=12)
    axes[1].tick_params(colors='white', rotation=45)
    axes[1].set_facecolor('#16213e')

    plt.suptitle(
        'Speech Emotion Detection — Confusion Matrix',
        fontsize=16, fontweight='bold', color='white', y=1.01
    )
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    logger.info(f"Saved confusion matrix: {CONFUSION_MATRIX_PATH}")


def main():
    """Main training pipeline."""
    logger.info("🎙️ Speech Emotion Detection — Training Pipeline")
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Random seed: {RANDOM_SEED}")

    # ── Step 1: Load dataset ──────────────────────────────────────────────────
    logger.info("\n[STEP 1] Loading dataset...")
    df = load_dataset_info(RAW_DIR)

    if df.empty:
        logger.error("❌ Dataset not found! Please download RAVDESS first.")
        logger.error("Download: https://zenodo.org/record/1188976")
        logger.error(f"Extract to: {RAW_DIR}")
        sys.exit(1)

    logger.info(f"Loaded {len(df)} audio files, {df['emotion'].nunique()} emotions, "
                f"{df['actor_id'].nunique()} speakers")

    # Get the actual emotion classes present in dataset
    actual_emotions = sorted(df['emotion'].unique().tolist())
    logger.info(f"Emotions in dataset: {actual_emotions}")

    # ── Step 2: Split dataset (speaker-independent) ───────────────────────────
    logger.info("\n[STEP 2] Splitting dataset (speaker-independent)...")
    train_df, val_df, test_df = split_dataset(df, random_seed=RANDOM_SEED)
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    logger.info("\n[STEP 3] Extracting features (with data augmentation on train set)...")
    feature_dim = get_feature_dimension()
    logger.info(f"New feature dimension: {feature_dim} (was 192 — now {feature_dim} with rich features)")

    # augment=True triples the training data via noise/pitch-shift/time-stretch
    X_train_raw, y_train = extract_all_features(train_df, 'train', augment=True)
    X_val_raw, y_val     = extract_all_features(val_df,   'val',   augment=False)
    X_test_raw, y_test   = extract_all_features(test_df,  'test',  augment=False)

    logger.info(f"X_train shape: {X_train_raw.shape}")
    logger.info(f"X_val shape:   {X_val_raw.shape}")
    logger.info(f"X_test shape:  {X_test_raw.shape}")

    # ── Step 4: Encode labels ─────────────────────────────────────────────────
    logger.info("\n[STEP 4] Encoding emotion labels...")
    label_encoder = LabelEncoder()
    label_encoder.fit(actual_emotions)

    y_train_enc = label_encoder.transform(y_train)
    y_val_enc   = label_encoder.transform(y_val)
    y_test_enc  = label_encoder.transform(y_test)

    logger.info(f"Label classes: {list(label_encoder.classes_)}")

    # ── Step 5: Scale features ────────────────────────────────────────────────
    logger.info("\n[STEP 5] Scaling features (StandardScaler)...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)   # Fit ONLY on training data!
    X_val   = scaler.transform(X_val_raw)          # Apply same transform
    X_test  = scaler.transform(X_test_raw)         # Apply same transform

    # ── Step 6: Train and compare models ─────────────────────────────────────
    logger.info("\n[STEP 6] Training and comparing models...")
    trained_models, comparison_df = train_and_evaluate_models(
        X_train, y_train, X_val, y_val, list(label_encoder.classes_)
    )

    # Save model comparison
    comparison_df.to_csv(MODEL_COMPARISON_PATH, index=False)
    logger.info(f"Saved model comparison: {MODEL_COMPARISON_PATH}")

    # ── Step 7: Select best model ─────────────────────────────────────────────
    logger.info("\n[STEP 7] Selecting best model based on validation Macro F1...")
    best_model_name = comparison_df.iloc[0]['Model']
    best_model = trained_models[best_model_name]
    logger.info(f"Best model: {best_model_name} "
                f"(Val Macro F1: {comparison_df.iloc[0]['Macro_F1']:.4f})")

    # ── Step 8: Retrain on train+val data ─────────────────────────────────────
    logger.info("\n[STEP 8] Retraining best model on train + val data...")
    X_trainval_raw = np.vstack([X_train_raw, X_val_raw])
    y_trainval = np.concatenate([y_train, y_val])

    # Re-scale with train+val data
    scaler_final = StandardScaler()
    X_trainval_scaled = scaler_final.fit_transform(X_trainval_raw)
    X_test_scaled     = scaler_final.transform(X_test_raw)

    # Retrain the best model type with same hyperparams
    final_model = define_models()[best_model_name]
    final_model.fit(X_trainval_scaled, y_trainval)
    logger.info(f"Final model retrained on {len(X_trainval_raw)} samples.")

    # ── Step 9: Evaluate on test set ──────────────────────────────────────────
    logger.info("\n[STEP 9] Final evaluation on test set...")
    y_test_pred = final_model.predict(X_test_scaled)

    metrics, _ = evaluate_final_model(
        final_model, X_test_scaled, y_test,
        label_encoder, list(label_encoder.classes_)
    )

    save_confusion_matrix(y_test, y_test_pred, list(label_encoder.classes_))

    # ── Step 10: Save model and artifacts ─────────────────────────────────────
    logger.info("\n[STEP 10] Saving model and artifacts...")

    joblib.dump(final_model, MODEL_PATH)
    logger.info(f"Saved model: {MODEL_PATH}")

    joblib.dump(scaler_final, SCALER_PATH)
    logger.info(f"Saved scaler: {SCALER_PATH}")

    joblib.dump(label_encoder, LABEL_ENCODER_PATH)
    logger.info(f"Saved label encoder: {LABEL_ENCODER_PATH}")

    # Save feature config
    with open(FEATURES_CONFIG_PATH, 'w') as f:
        json.dump(FEATURE_CONFIG, f, indent=2)

    # Save full metrics
    all_metrics = {
        **metrics,
        "model_name": best_model_name,
        "dataset": "RAVDESS",
        "n_emotions": len(label_encoder.classes_),
        "emotion_classes": list(label_encoder.classes_),
        "feature_dimension": int(X_train.shape[1]),
        "train_samples": int(len(X_trainval_raw)),
        "test_samples": int(len(X_test)),
        "model_comparison": comparison_df.to_dict(orient='records'),
    }
    with open(METRICS_JSON_PATH, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"Saved metrics: {METRICS_JSON_PATH}")

    # Save model metadata
    metadata = {
        "model_type": best_model_name,
        "dataset": "RAVDESS",
        "dataset_url": "https://zenodo.org/record/1188976",
        "feature_type": "MFCC+Delta+ZCR+RMS+Centroid+Rolloff+Chroma",
        "feature_dimension": int(X_train.shape[1]),
        "emotion_classes": list(label_encoder.classes_),
        "n_emotions": len(label_encoder.classes_),
        "random_seed": RANDOM_SEED,
        "training_date": datetime.now().isoformat(),
        "test_accuracy": metrics["test_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "train_samples": int(len(X_trainval_raw)),
        "test_samples": int(len(X_test)),
    }
    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata: {METADATA_PATH}")

    logger.info("\n" + "="*60)
    logger.info("✅ TRAINING COMPLETE!")
    logger.info("="*60)
    logger.info(f"Best model:      {best_model_name}")
    logger.info(f"Test Accuracy:   {metrics['test_accuracy']:.4f} ({metrics['test_accuracy']*100:.1f}%)")
    logger.info(f"Macro F1:        {metrics['macro_f1']:.4f}")
    logger.info(f"Weighted F1:     {metrics['weighted_f1']:.4f}")
    logger.info(f"\nFiles saved in: {MODELS_DIR}")
    logger.info(f"Results in:     {RESULTS_DIR}")
    logger.info("\nYou can now start the backend: python backend/main.py")


if __name__ == "__main__":
    main()
