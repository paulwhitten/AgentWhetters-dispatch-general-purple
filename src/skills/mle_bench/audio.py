"""Audio modality pipeline -- generates scripts for audio classification.

Uses librosa for feature extraction (MFCCs, mel-spectrograms) and sklearn
or PyTorch for classification. Falls back to basic signal statistics if
librosa is not available.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _find_audio_dir(data_dir: Path, prefix: str) -> Path | None:
    """Find an audio directory matching prefix (e.g. 'train', 'test')."""
    audio_exts = (".wav", ".mp3", ".flac", ".ogg", ".aif")
    # Try exact match first, then prefix match
    candidates = [data_dir / prefix]
    candidates += sorted(
        d for d in data_dir.iterdir()
        if d.is_dir() and d.name.lower().startswith(prefix)
    )
    for d in candidates:
        if d.exists() and d.is_dir():
            if any(f.suffix.lower() in audio_exts for f in list(d.iterdir())[:20] if f.is_file()):
                return d
    return None


def generate_audio_script(data_dir: Path, description: str) -> str | None:
    """Generate an audio classification script.

    Handles varied directory structures (train/, train2/, etc.) and
    label-in-filename patterns.
    """
    train_dir = _find_audio_dir(data_dir, "train")
    test_dir = _find_audio_dir(data_dir, "test")

    if not train_dir:
        return None

    script = f"""import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from pathlib import Path

START = time.time()
DATA_DIR = '{data_dir}'
SAMPLE_RATE = 22050
MAX_DURATION = 5  # seconds
N_MFCC = 40

print('Audio classification pipeline', flush=True)

# Check for librosa
try:
    import librosa
    HAS_LIBROSA = True
    print('Using librosa for feature extraction', flush=True)
except ImportError:
    HAS_LIBROSA = False
    print('librosa not available, using scipy fallback', flush=True)

# Load labels
train_csv_path = os.path.join(DATA_DIR, 'train.csv')
if os.path.exists(train_csv_path):
    train_df = pd.read_csv(train_csv_path)
    print(f'Train CSV: {{train_df.shape}}, columns: {{list(train_df.columns)}}', flush=True)
else:
    # Infer from folder structure
    train_dir = os.path.join(DATA_DIR, 'train')
    entries = []
    for class_name in sorted(os.listdir(train_dir)):
        class_path = os.path.join(train_dir, class_name)
        if os.path.isdir(class_path):
            for fname in os.listdir(class_path):
                entries.append({{'fname': os.path.join(class_name, fname), 'label': class_name}})
    train_df = pd.DataFrame(entries)
    print(f'Inferred {{len(train_df)}} train samples from folders', flush=True)

# Detect columns
fname_col = None
label_col = None
for col in train_df.columns:
    cl = col.lower()
    if any(kw in cl for kw in ('file', 'fname', 'path', 'audio', 'clip')):
        fname_col = col
    elif any(kw in cl for kw in ('label', 'target', 'class', 'category', 'word')):
        label_col = col

if fname_col is None:
    fname_col = train_df.columns[0]
if label_col is None:
    label_col = train_df.columns[-1]

print(f'File col: {{fname_col}}, Label col: {{label_col}}', flush=True)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_labels = le.fit_transform(train_df[label_col].astype(str))
n_classes = len(le.classes_)
print(f'Classes: {{n_classes}}', flush=True)


def extract_features(filepath, sr=SAMPLE_RATE, max_dur=MAX_DURATION):
    \"\"\"Extract audio features from a single file.\"\"\"
    try:
        if HAS_LIBROSA:
            y, sr = librosa.load(filepath, sr=sr, duration=max_dur)
            # MFCCs
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
            mfcc_mean = mfcc.mean(axis=1)
            mfcc_std = mfcc.std(axis=1)
            # Spectral features
            spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
            spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
            spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
            zcr = librosa.feature.zero_crossing_rate(y).mean()
            rms = librosa.feature.rms(y=y).mean()
            features = np.concatenate([
                mfcc_mean, mfcc_std,
                [spec_cent, spec_bw, spec_rolloff, zcr, rms]
            ])
        else:
            from scipy.io import wavfile
            sr_file, data = wavfile.read(filepath)
            if len(data.shape) > 1:
                data = data.mean(axis=1)
            data = data.astype(float)
            # Basic statistics as features
            features = np.array([
                data.mean(), data.std(), data.max(), data.min(),
                np.median(data), np.percentile(data, 25), np.percentile(data, 75),
                len(data) / sr_file,  # duration
                np.sum(np.abs(np.diff(np.sign(data)))) / len(data),  # ZCR approx
            ])
            # Pad to fixed size
            features = np.pad(features, (0, max(0, N_MFCC * 2 + 5 - len(features))))[:N_MFCC * 2 + 5]
        return features
    except Exception as e:
        return np.zeros(N_MFCC * 2 + 5)


# Extract train features
print('Extracting train features...', flush=True)
X_train = []
valid_indices = []

train_audio_dir = os.path.join(DATA_DIR, 'train')
for i, row in train_df.iterrows():
    fname = str(row[fname_col])
    # Try multiple paths
    candidates = [
        os.path.join(DATA_DIR, fname),
        os.path.join(train_audio_dir, fname),
        os.path.join(DATA_DIR, 'audio_train', fname),
    ]
    filepath = None
    for c in candidates:
        if os.path.exists(c):
            filepath = c
            break

    if filepath is None:
        X_train.append(np.zeros(N_MFCC * 2 + 5))
        valid_indices.append(i)
        continue

    feat = extract_features(filepath)
    X_train.append(feat)
    valid_indices.append(i)

    if (i + 1) % 500 == 0:
        print(f'  Processed {{i+1}}/{{len(train_df)}} ({{time.time()-START:.0f}}s)', flush=True)
    if time.time() - START > 400:
        print(f'  Time limit: stopping at {{i+1}} samples', flush=True)
        break

X_train = np.array(X_train)
y_train = train_labels[:len(X_train)]
print(f'Train features: {{X_train.shape}}', flush=True)

# Train classifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score

print('Training ensemble...', flush=True)
# Use GBM for better accuracy on small feature sets
clf = GradientBoostingClassifier(
    n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
)
clf.fit(X_train, y_train)

# Quick CV score
if len(X_train) > 50:
    cv_scores = cross_val_score(clf, X_train, y_train, cv=3, scoring='accuracy')
    print(f'CV accuracy: {{cv_scores.mean():.4f}} +/- {{cv_scores.std():.4f}}', flush=True)

# Extract test features
print('Extracting test features...', flush=True)
test_csv_path = os.path.join(DATA_DIR, 'test.csv')
if os.path.exists(test_csv_path):
    test_df = pd.read_csv(test_csv_path)
else:
    # Check sample submission
    for name in ['sample_submission.csv', 'sampleSubmission.csv']:
        p = os.path.join(DATA_DIR, name)
        if os.path.exists(p):
            test_df = pd.read_csv(p)
            break
    else:
        test_dir = os.path.join(DATA_DIR, 'test')
        test_files = sorted([f for f in os.listdir(test_dir) if not os.path.isdir(os.path.join(test_dir, f))])
        test_df = pd.DataFrame({{fname_col: test_files}})

X_test = []
test_audio_dir = os.path.join(DATA_DIR, 'test')
for i, row in test_df.iterrows():
    fname = str(row[test_df.columns[0]])
    candidates = [
        os.path.join(DATA_DIR, fname),
        os.path.join(test_audio_dir, fname),
        os.path.join(DATA_DIR, 'audio_test', fname),
    ]
    filepath = None
    for c in candidates:
        if os.path.exists(c):
            filepath = c
            break

    feat = extract_features(filepath) if filepath else np.zeros(N_MFCC * 2 + 5)
    X_test.append(feat)

X_test = np.array(X_test)
print(f'Test features: {{X_test.shape}}', flush=True)

# Predict
preds = clf.predict(X_test)
pred_labels = le.inverse_transform(preds)

# Write submission
submission = pd.DataFrame({{test_df.columns[0]: test_df.iloc[:, 0], label_col: pred_labels}})
submission.to_csv(os.path.join(DATA_DIR, 'submission.csv'), index=False)
print(f'Submission: {{submission.shape}}', flush=True)
print(f'Total time: {{time.time()-START:.0f}}s', flush=True)
"""
    return script
