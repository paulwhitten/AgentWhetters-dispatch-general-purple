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

    # Detect if sample submission expects probabilities
    sample_sub = data_dir / "sampleSubmission.csv"
    if not sample_sub.exists():
        sample_sub = data_dir / "sample_submission.csv"
    wants_probability = False
    sub_columns = None
    if sample_sub.exists():
        import pandas as pd
        samp = pd.read_csv(sample_sub, nrows=3)
        sub_columns = list(samp.columns)
        # If there's a "probability" column or the values are floats, it wants probabilities
        if any("prob" in c.lower() for c in samp.columns):
            wants_probability = True

    test_dir_str = str(test_dir) if test_dir else ""

    script = f"""import os
import sys
import time
import re
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from pathlib import Path

START = time.time()
DATA_DIR = '{data_dir}'
TRAIN_DIR = '{train_dir}'
TEST_DIR = '{test_dir_str}'
SAMPLE_RATE = 2000  # whale .aif files are typically 2kHz
MAX_DURATION = 5  # seconds
N_MFCC = 20
WANTS_PROBABILITY = {wants_probability}
SUB_COLUMNS = {sub_columns}

print('Audio classification pipeline', flush=True)
print(f'Train dir: {{TRAIN_DIR}}', flush=True)
print(f'Test dir: {{TEST_DIR}}', flush=True)

# Check for librosa or soundfile
HAS_LIBROSA = False
HAS_SOUNDFILE = False
try:
    import librosa
    HAS_LIBROSA = True
    print('Using librosa for feature extraction', flush=True)
except ImportError:
    pass
if not HAS_LIBROSA:
    try:
        import soundfile as sf
        HAS_SOUNDFILE = True
        print('Using soundfile for audio loading', flush=True)
    except ImportError:
        pass

if not HAS_LIBROSA and not HAS_SOUNDFILE:
    import struct
    print('Using struct-based AIFF parser for .aif loading', flush=True)


def load_audio(filepath):
    \"\"\"Load audio file, return (samples_array, sample_rate).\"\"\"
    if HAS_LIBROSA:
        y, sr = librosa.load(filepath, sr=SAMPLE_RATE, duration=MAX_DURATION)
        return y, sr
    elif HAS_SOUNDFILE:
        import soundfile as sf
        data, sr = sf.read(filepath)
        if len(data.shape) > 1:
            data = data.mean(axis=1)
        # Resample if needed (simple decimation)
        if sr != SAMPLE_RATE:
            ratio = SAMPLE_RATE / sr
            n = int(len(data) * ratio)
            data = np.interp(np.linspace(0, len(data)-1, n), np.arange(len(data)), data)
        return data.astype(np.float32), SAMPLE_RATE
    else:
        # Pure struct-based AIFF parser (no aifc needed)
        import struct
        with open(filepath, 'rb') as f:
            # Read FORM header
            form_id = f.read(4)
            form_size = struct.unpack('>I', f.read(4))[0]
            form_type = f.read(4)
            # Find COMM and SSND chunks
            n_channels = 1
            sampwidth = 2
            n_frames = 0
            sr = 2000
            audio_data = b''
            while f.tell() < 12 + form_size:
                chunk_id = f.read(4)
                if len(chunk_id) < 4:
                    break
                chunk_size = struct.unpack('>I', f.read(4))[0]
                if chunk_id == b'COMM':
                    n_channels = struct.unpack('>h', f.read(2))[0]
                    n_frames = struct.unpack('>I', f.read(4))[0]
                    sampwidth = struct.unpack('>h', f.read(2))[0] // 8
                    # 80-bit extended float for sample rate
                    exp_bytes = f.read(10)
                    exp = struct.unpack('>H', exp_bytes[:2])[0]
                    mantissa = struct.unpack('>Q', exp_bytes[2:10])[0]
                    sign = (-1) ** (exp >> 15)
                    exp = (exp & 0x7FFF) - 16383
                    sr = int(sign * (mantissa / (1 << 63)) * (2 ** exp))
                    remaining = chunk_size - 18
                    if remaining > 0:
                        f.read(remaining)
                elif chunk_id == b'SSND':
                    offset = struct.unpack('>I', f.read(4))[0]
                    block_size = struct.unpack('>I', f.read(4))[0]
                    if offset > 0:
                        f.read(offset)
                    audio_data = f.read(chunk_size - 8 - offset)
                else:
                    f.read(chunk_size)
                # Chunks are padded to even size
                if chunk_size % 2 != 0:
                    f.read(1)
        if not audio_data:
            return np.zeros(100, dtype=np.float32), sr
        if sampwidth == 2:
            samples = np.frombuffer(audio_data, dtype='>i2').astype(np.float32) / 32768.0
        elif sampwidth == 1:
            samples = np.frombuffer(audio_data, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
        elif sampwidth == 3:
            # 24-bit big-endian
            raw = np.frombuffer(audio_data, dtype=np.uint8)
            n_samp = len(raw) // 3
            raw = raw[:n_samp * 3].reshape(-1, 3)
            samples = ((raw[:, 0].astype(np.int32) << 24) | (raw[:, 1].astype(np.int32) << 16) | (raw[:, 2].astype(np.int32) << 8)).astype(np.float32) / 2147483648.0
        else:
            samples = np.frombuffer(audio_data, dtype='>i4').astype(np.float32) / 2147483648.0
        if n_channels > 1:
            samples = samples.reshape(-1, n_channels).mean(axis=1)
        max_samples = int(MAX_DURATION * sr)
        samples = samples[:max_samples]
        return samples, sr


def extract_features(filepath):
    \"\"\"Extract audio features from a single file.\"\"\"
    try:
        y, sr = load_audio(filepath)
        if len(y) == 0:
            return np.zeros(N_MFCC * 2 + 5)

        if HAS_LIBROSA:
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
            mfcc_mean = mfcc.mean(axis=1)
            mfcc_std = mfcc.std(axis=1)
            spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
            spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
            spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
            zcr = librosa.feature.zero_crossing_rate(y).mean()
            rms = librosa.feature.rms(y=y).mean()
            features = np.concatenate([mfcc_mean, mfcc_std, [spec_cent, spec_bw, spec_rolloff, zcr, rms]])
        else:
            # Basic spectral features without librosa
            fft = np.fft.rfft(y)
            magnitude = np.abs(fft)
            freqs = np.fft.rfftfreq(len(y), 1.0/sr)

            # Spectral centroid
            spec_cent = np.sum(freqs * magnitude) / (np.sum(magnitude) + 1e-10)
            # Spectral bandwidth
            spec_bw = np.sqrt(np.sum(((freqs - spec_cent)**2) * magnitude) / (np.sum(magnitude) + 1e-10))
            # ZCR
            zcr = np.sum(np.abs(np.diff(np.sign(y)))) / (2.0 * len(y))
            # RMS
            rms = np.sqrt(np.mean(y**2))
            # Energy in frequency bands
            band_edges = np.linspace(0, sr/2, N_MFCC + 1)
            band_energies = []
            for i in range(N_MFCC):
                mask = (freqs >= band_edges[i]) & (freqs < band_edges[i+1])
                band_energies.append(np.sum(magnitude[mask]**2) + 1e-10)
            band_energies = np.log(np.array(band_energies))

            # Statistics
            stats = np.array([y.mean(), y.std(), y.max(), y.min(), np.median(y),
                            spec_cent, spec_bw, zcr, rms, len(y)/sr])
            features = np.concatenate([band_energies, stats])
            # Pad/trim to expected size
            target_size = N_MFCC * 2 + 5
            if len(features) < target_size:
                features = np.pad(features, (0, target_size - len(features)))
            else:
                features = features[:target_size]
        return features
    except Exception as e:
        print(f'  Feature extraction failed for {{filepath}}: {{e}}', flush=True)
        return np.zeros(N_MFCC * 2 + 5)


# --- Discover training labels ---
train_csv_path = os.path.join(DATA_DIR, 'train.csv')
audio_exts = ('.aif', '.wav', '.mp3', '.flac', '.ogg')

if os.path.exists(train_csv_path):
    train_df = pd.read_csv(train_csv_path)
    print(f'Train CSV: {{train_df.shape}}, columns: {{list(train_df.columns)}}', flush=True)
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
else:
    # Try to infer labels from filenames (e.g. TRAIN19505_0.aif where _0/_1 is label)
    train_files = sorted([f for f in os.listdir(TRAIN_DIR)
                         if os.path.isfile(os.path.join(TRAIN_DIR, f)) and
                         f.lower().endswith(audio_exts)])
    print(f'Found {{len(train_files)}} audio files in train dir', flush=True)

    # Check if labels are in filename (pattern: ..._LABEL.ext)
    label_pattern = re.compile(r'_(\\d+)\\.\\w+$')
    labels_from_fname = []
    for f in train_files[:20]:
        m = label_pattern.search(f)
        if m:
            labels_from_fname.append(m.group(1))

    if len(labels_from_fname) >= 10:
        # Labels are encoded in filenames
        print(f'Labels detected in filenames (e.g. _0, _1 suffix)', flush=True)
        entries = []
        for f in train_files:
            m = label_pattern.search(f)
            label = m.group(1) if m else '0'
            entries.append({{'fname': f, 'label': int(label)}})
        train_df = pd.DataFrame(entries)
        fname_col = 'fname'
        label_col = 'label'
    else:
        # Check for subfolders as class labels
        subdirs = [d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))]
        if subdirs:
            entries = []
            for class_name in sorted(subdirs):
                class_path = os.path.join(TRAIN_DIR, class_name)
                for fname in os.listdir(class_path):
                    if fname.lower().endswith(audio_exts):
                        entries.append({{'fname': os.path.join(class_name, fname), 'label': class_name}})
            train_df = pd.DataFrame(entries)
            fname_col = 'fname'
            label_col = 'label'
        else:
            print('ERROR: Cannot determine training labels', flush=True)
            sys.exit(1)

print(f'Training samples: {{len(train_df)}}, Label col: {{label_col}}', flush=True)
print(f'Label distribution: {{train_df[label_col].value_counts().to_dict()}}', flush=True)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_labels = le.fit_transform(train_df[label_col].astype(str))
n_classes = len(le.classes_)
print(f'Classes: {{n_classes}}', flush=True)

# --- Extract train features ---
print('Extracting train features...', flush=True)
X_train = []
MAX_TRAIN = 8000  # Limit for time

for i, row in train_df.iterrows():
    if i >= MAX_TRAIN:
        print(f'  Reached max train samples ({{MAX_TRAIN}})', flush=True)
        break

    fname = str(row[fname_col])
    filepath = os.path.join(TRAIN_DIR, fname)
    if not os.path.exists(filepath):
        filepath = os.path.join(DATA_DIR, fname)

    if os.path.exists(filepath):
        feat = extract_features(filepath)
    else:
        feat = np.zeros(N_MFCC * 2 + 5)
    X_train.append(feat)

    if (i + 1) % 1000 == 0:
        print(f'  Processed {{i+1}}/{{min(len(train_df), MAX_TRAIN)}} ({{time.time()-START:.0f}}s)', flush=True)
    if time.time() - START > 400:
        print(f'  Time limit: stopping at {{i+1}} samples', flush=True)
        break

X_train = np.array(X_train)
y_train = train_labels[:len(X_train)]
print(f'Train features: {{X_train.shape}}', flush=True)

# --- Train classifier ---
from sklearn.ensemble import GradientBoostingClassifier

print('Training classifier...', flush=True)
if WANTS_PROBABILITY or n_classes == 2:
    # Binary classification with probability output
    clf = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42
    )
else:
    clf = GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
    )
clf.fit(X_train, y_train)
print(f'Training complete ({{time.time()-START:.0f}}s)', flush=True)

# --- Identify test files ---
print('Identifying test files...', flush=True)

# Use sample submission to get test file list and output format
sample_sub_path = None
for name in ['sampleSubmission.csv', 'sample_submission.csv']:
    p = os.path.join(DATA_DIR, name)
    if os.path.exists(p):
        sample_sub_path = p
        break

if sample_sub_path:
    sub_df = pd.read_csv(sample_sub_path)
    test_ids = sub_df.iloc[:, 0].tolist()
    id_col_name = sub_df.columns[0]
    target_col_name = sub_df.columns[1] if len(sub_df.columns) > 1 else label_col
    print(f'Sample submission: {{len(test_ids)}} entries, cols: {{list(sub_df.columns)}}', flush=True)
else:
    # List test directory
    if TEST_DIR and os.path.exists(TEST_DIR):
        test_ids = sorted([f for f in os.listdir(TEST_DIR)
                          if os.path.isfile(os.path.join(TEST_DIR, f)) and
                          f.lower().endswith(audio_exts)])
    else:
        test_ids = []
    id_col_name = 'clip'
    target_col_name = label_col

print(f'Test samples: {{len(test_ids)}}', flush=True)

# --- Extract test features ---
print('Extracting test features...', flush=True)
X_test = []

for i, test_id in enumerate(test_ids):
    # Try to find the file
    filepath = None
    candidates = []
    if TEST_DIR:
        candidates.append(os.path.join(TEST_DIR, test_id))
    candidates.append(os.path.join(DATA_DIR, test_id))
    # Strip extension and try
    base = os.path.splitext(test_id)[0]
    if TEST_DIR:
        for ext in audio_exts:
            candidates.append(os.path.join(TEST_DIR, base + ext))

    for c in candidates:
        if os.path.exists(c):
            filepath = c
            break

    if filepath:
        feat = extract_features(filepath)
    else:
        feat = np.zeros(N_MFCC * 2 + 5)
    X_test.append(feat)

    if (i + 1) % 5000 == 0:
        print(f'  Processed {{i+1}}/{{len(test_ids)}} ({{time.time()-START:.0f}}s)', flush=True)

X_test = np.array(X_test)
print(f'Test features: {{X_test.shape}}', flush=True)

# --- Predict and write submission ---
print('Generating predictions...', flush=True)

if WANTS_PROBABILITY and n_classes == 2:
    # Output probability of positive class
    probs = clf.predict_proba(X_test)[:, 1]
    submission = pd.DataFrame({{id_col_name: test_ids, target_col_name: probs}})
elif WANTS_PROBABILITY:
    probs = clf.predict_proba(X_test)
    # Multi-class probability - use max prob
    submission = pd.DataFrame({{id_col_name: test_ids, target_col_name: probs.max(axis=1)}})
else:
    preds = clf.predict(X_test)
    pred_labels = le.inverse_transform(preds)
    submission = pd.DataFrame({{id_col_name: test_ids, target_col_name: pred_labels}})

# Match sample submission column names exactly
if SUB_COLUMNS and len(SUB_COLUMNS) >= 2:
    submission.columns = SUB_COLUMNS[:2]

submission.to_csv(os.path.join(DATA_DIR, 'submission.csv'), index=False)
print(f'Submission: {{submission.shape}}', flush=True)
print(f'First 3 rows:\\n{{submission.head(3)}}', flush=True)
print(f'Total time: {{time.time()-START:.0f}}s', flush=True)
"""
    return script
