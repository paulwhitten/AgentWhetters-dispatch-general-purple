"""Image modality pipeline -- lightweight CNN/transfer script generators.

For image classification competitions, generates scripts that use:
- A simple CNN baseline with data augmentation (Keras/TF or torchvision)
- ResNet/EfficientNet transfer learning when torchvision is available
- Falls back to flattened pixel features + sklearn if no DL framework available
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def generate_image_script(data_dir: Path, description: str) -> str | None:
    """Generate an image classification script.

    Expects: train/ and test/ directories with class subfolders (train) or flat images (test),
    or a CSV mapping filenames to labels.
    """
    desc_lower = description.lower()

    # Detect structure
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"
    has_subdirs = train_dir.exists() and any(d.is_dir() for d in train_dir.iterdir())

    # Check for CSV-based labels
    train_csv = data_dir / "train.csv"
    has_csv_labels = train_csv.exists()

    if not train_dir.exists() and not has_csv_labels:
        logger.warning("No train/ directory or train.csv found for image pipeline")
        return None

    # Detect number of classes from folder structure or CSV
    n_classes = 2  # default
    if has_subdirs:
        n_classes = sum(1 for d in train_dir.iterdir() if d.is_dir())

    # Detect image size from description
    img_size = 64
    if "224" in description or "imagenet" in desc_lower:
        img_size = 224
    elif "128" in description:
        img_size = 128
    elif "96" in description:
        img_size = 96
    elif n_classes > 10:
        img_size = 128

    # Detect if binary (probability output) or multiclass
    is_binary = n_classes == 2

    script = f"""import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

START = time.time()
DATA_DIR = '{data_dir}'
IMG_SIZE = {img_size}
N_CLASSES = {n_classes}
BATCH_SIZE = 32

print(f'Image classification: {{N_CLASSES}} classes, img_size={{IMG_SIZE}}', flush=True)

# Try torchvision first, fallback to sklearn
try:
    import torch
    import torchvision
    from torchvision import transforms, datasets, models
    from torch.utils.data import DataLoader, Dataset
    from torch import nn, optim
    HAS_TORCH = True
    print('Using PyTorch/torchvision', flush=True)
except ImportError:
    HAS_TORCH = False
    print('PyTorch not available, using sklearn fallback', flush=True)

"""

    if has_subdirs:
        script += _torch_folder_pipeline(data_dir, img_size, n_classes, is_binary)
    elif has_csv_labels:
        script += _torch_csv_pipeline(data_dir, img_size, n_classes, is_binary)
    else:
        return None

    return script


def _torch_folder_pipeline(data_dir: Path, img_size: int, n_classes: int, is_binary: bool) -> str:
    """Pipeline for train/<class>/<images> structure."""
    return f"""
if HAS_TORCH:
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize(({img_size}, {img_size})),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(({img_size}, {img_size})),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_dir = os.path.join(DATA_DIR, 'train')
    test_dir = os.path.join(DATA_DIR, 'test')

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    class_names = train_dataset.classes
    print(f'Classes: {{class_names[:10]}}...', flush=True)
    print(f'Train samples: {{len(train_dataset)}}', flush=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # Simple CNN model
    class SimpleCNN(nn.Module):
        def __init__(self, n_classes):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d(4),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 4 * 4, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SimpleCNN(N_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train for limited epochs (time budget)
    N_EPOCHS = min(10, max(3, int(300 / max(1, len(train_dataset) / BATCH_SIZE))))
    print(f'Training {{N_EPOCHS}} epochs on {{device}}...', flush=True)

    for epoch in range(N_EPOCHS):
        model.train()
        running_loss = 0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        if (epoch + 1) % 2 == 0 or epoch == N_EPOCHS - 1:
            print(f'  Epoch {{epoch+1}}/{{N_EPOCHS}}: loss={{running_loss/len(train_loader):.4f}}, acc={{100*correct/total:.1f}}%', flush=True)
        if time.time() - START > 500:
            print('Time limit approaching, stopping training', flush=True)
            break

    # Predict on test set
    model.eval()
    test_preds = []
    test_files = []

    # Handle both subfolder and flat test directory structures
    if any(os.path.isdir(os.path.join(test_dir, d)) for d in os.listdir(test_dir)):
        test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=2)
        for images, _ in test_loader:
            images = images.to(device)
            with torch.no_grad():
                outputs = model(images)
                _, predicted = outputs.max(1)
                test_preds.extend(predicted.cpu().numpy())
        test_files = [os.path.basename(p[0]) for p in test_dataset.imgs]
    else:
        from PIL import Image
        test_images = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        for img_name in test_images:
            img_path = os.path.join(test_dir, img_name)
            img = Image.open(img_path).convert('RGB')
            img_t = test_transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(img_t)
                _, pred = out.max(1)
                test_preds.append(pred.item())
            test_files.append(img_name)

    # Map predictions back to class names
    pred_labels = [class_names[p] for p in test_preds]

    submission = pd.DataFrame({{'id': test_files, 'label': pred_labels}})
    # Try to match sample submission format
    sample_sub = None
    for name in ['sample_submission.csv', 'sampleSubmission.csv', 'SampleSubmission.csv']:
        path = os.path.join(DATA_DIR, name)
        if os.path.exists(path):
            sample_sub = pd.read_csv(path)
            break
    if sample_sub is not None:
        submission.columns = sample_sub.columns[:2]

    submission.to_csv(os.path.join(DATA_DIR, 'submission.csv'), index=False)
    print(f'Submission: {{submission.shape}}', flush=True)

else:
    # sklearn fallback: flatten images to feature vectors
    from PIL import Image
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder

    train_dir = os.path.join(DATA_DIR, 'train')
    test_dir = os.path.join(DATA_DIR, 'test')

    # Load training images
    X_train, y_train = [], []
    for class_name in sorted(os.listdir(train_dir)):
        class_path = os.path.join(train_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        for img_name in list(os.listdir(class_path))[:200]:  # limit for speed
            try:
                img = Image.open(os.path.join(class_path, img_name)).convert('RGB').resize((32, 32))
                X_train.append(np.array(img).flatten())
                y_train.append(class_name)
            except Exception:
                continue

    X_train = np.array(X_train)
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)

    print(f'Training RF on {{len(X_train)}} images...', flush=True)
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    clf.fit(X_train, y_train)

    # Predict
    test_files = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
    X_test = []
    for img_name in test_files:
        try:
            img = Image.open(os.path.join(test_dir, img_name)).convert('RGB').resize((32, 32))
            X_test.append(np.array(img).flatten())
        except Exception:
            X_test.append(np.zeros(32*32*3))

    X_test = np.array(X_test)
    preds = clf.predict(X_test)
    pred_labels = le.inverse_transform(preds)

    submission = pd.DataFrame({{'id': test_files, 'label': pred_labels}})
    submission.to_csv(os.path.join(DATA_DIR, 'submission.csv'), index=False)
    print(f'Submission (sklearn fallback): {{submission.shape}}', flush=True)

print(f'Total time: {{time.time()-START:.0f}}s', flush=True)
"""


def _torch_csv_pipeline(data_dir: Path, img_size: int, n_classes: int, is_binary: bool) -> str:
    """Pipeline for CSV-based labels with image paths."""
    return f"""
# CSV-based image classification
train_csv = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
print(f'Train CSV: {{train_csv.shape}}', flush=True)
print(f'Columns: {{list(train_csv.columns)}}', flush=True)

# Detect image column and label column
img_col = None
label_col = None
for col in train_csv.columns:
    if any(kw in col.lower() for kw in ('image', 'file', 'path', 'img', 'id')):
        img_col = col
    elif any(kw in col.lower() for kw in ('label', 'target', 'class', 'category')):
        label_col = col

if img_col is None:
    img_col = train_csv.columns[0]
if label_col is None:
    label_col = train_csv.columns[-1]

print(f'Image col: {{img_col}}, Label col: {{label_col}}', flush=True)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
labels = le.fit_transform(train_csv[label_col].astype(str))
n_cls = len(le.classes_)
print(f'Classes: {{n_cls}}', flush=True)

if HAS_TORCH:
    from PIL import Image
    from torch.utils.data import Dataset, DataLoader

    class CSVImageDataset(Dataset):
        def __init__(self, df, img_col, labels, data_dir, transform):
            self.df = df
            self.img_col = img_col
            self.labels = labels
            self.data_dir = data_dir
            self.transform = transform

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            img_name = str(self.df.iloc[idx][self.img_col])
            # Try multiple paths
            for prefix in ['train/', 'train/train/', 'images/', '']:
                path = os.path.join(self.data_dir, prefix, img_name)
                if os.path.exists(path):
                    break
                # Try adding extension
                for ext in ['.jpg', '.png', '.jpeg']:
                    if os.path.exists(path + ext):
                        path = path + ext
                        break
            try:
                img = Image.open(path).convert('RGB')
            except Exception:
                img = Image.new('RGB', ({img_size}, {img_size}))
            img = self.transform(img)
            label = self.labels[idx] if self.labels is not None else 0
            return img, label

    train_transform = transforms.Compose([
        transforms.Resize(({img_size}, {img_size})),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(({img_size}, {img_size})),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_dataset = CSVImageDataset(train_csv, img_col, labels, DATA_DIR, train_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # Simple CNN
    class SimpleCNN(nn.Module):
        def __init__(self, n_classes):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d(4),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(256, n_classes),
            )
        def forward(self, x):
            return self.classifier(self.features(x))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SimpleCNN(n_cls).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    N_EPOCHS = min(8, max(3, int(300 / max(1, len(train_dataset) / BATCH_SIZE))))
    print(f'Training {{N_EPOCHS}} epochs...', flush=True)

    for epoch in range(N_EPOCHS):
        model.train()
        for images, lbl in train_loader:
            images, lbl = images.to(device), lbl.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), lbl)
            loss.backward()
            optimizer.step()
        if time.time() - START > 500:
            break

    # Test predictions
    model.eval()
    test_csv_path = os.path.join(DATA_DIR, 'test.csv')
    if os.path.exists(test_csv_path):
        test_df = pd.read_csv(test_csv_path)
    else:
        # Use sample submission for IDs
        for name in ['sample_submission.csv', 'sampleSubmission.csv']:
            p = os.path.join(DATA_DIR, name)
            if os.path.exists(p):
                test_df = pd.read_csv(p)
                break
        else:
            test_dir = os.path.join(DATA_DIR, 'test')
            test_files = sorted(os.listdir(test_dir))
            test_df = pd.DataFrame({{img_col: test_files}})

    test_dataset = CSVImageDataset(test_df, img_col, None, DATA_DIR, test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=2)

    preds = []
    for images, _ in test_loader:
        images = images.to(device)
        with torch.no_grad():
            out = model(images)
            _, pred = out.max(1)
            preds.extend(pred.cpu().numpy())

    pred_labels = le.inverse_transform(preds)
    submission = pd.DataFrame({{img_col: test_df[img_col], label_col: pred_labels}})
    submission.to_csv(os.path.join(DATA_DIR, 'submission.csv'), index=False)
    print(f'Submission: {{submission.shape}}', flush=True)

else:
    # Minimal sklearn fallback
    from PIL import Image
    from sklearn.ensemble import RandomForestClassifier

    X_train = []
    for _, row in train_csv.head(500).iterrows():
        img_name = str(row[img_col])
        for prefix in ['train/', 'images/', '']:
            path = os.path.join(DATA_DIR, prefix, img_name)
            if os.path.exists(path):
                break
        try:
            img = Image.open(path).convert('RGB').resize((32, 32))
            X_train.append(np.array(img).flatten())
        except Exception:
            X_train.append(np.zeros(32*32*3))

    X_train = np.array(X_train)
    clf = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42)
    clf.fit(X_train, labels[:len(X_train)])

    # Predict
    test_csv_path = os.path.join(DATA_DIR, 'test.csv')
    if os.path.exists(test_csv_path):
        test_df = pd.read_csv(test_csv_path)
    else:
        for name in ['sample_submission.csv', 'sampleSubmission.csv']:
            p = os.path.join(DATA_DIR, name)
            if os.path.exists(p):
                test_df = pd.read_csv(p)
                break

    X_test = []
    for _, row in test_df.iterrows():
        img_name = str(row[img_col])
        for prefix in ['test/', 'images/', '']:
            path = os.path.join(DATA_DIR, prefix, img_name)
            if os.path.exists(path):
                break
        try:
            img = Image.open(path).convert('RGB').resize((32, 32))
            X_test.append(np.array(img).flatten())
        except Exception:
            X_test.append(np.zeros(32*32*3))

    preds = clf.predict(np.array(X_test))
    pred_labels = le.inverse_transform(preds)
    submission = pd.DataFrame({{img_col: test_df[img_col], label_col: pred_labels}})
    submission.to_csv(os.path.join(DATA_DIR, 'submission.csv'), index=False)
    print(f'Submission (sklearn): {{submission.shape}}', flush=True)

print(f'Total time: {{time.time()-START:.0f}}s', flush=True)
"""
