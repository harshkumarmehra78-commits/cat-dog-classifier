# ============================================================
# Cats vs Dogs Image Classification - Complete ML Project
# ============================================================
# Author: ML Project
# Description: Multi-model comparison for binary image classification
# Models: Logistic Regression, SVM, Random Forest, CNN (TensorFlow/Keras)
# ============================================================

# ─────────────────────────────────────────────────────────────
# SECTION 0: Install Dependencies (run once if needed)
# ─────────────────────────────────────────────────────────────
# pip install kagglehub tensorflow scikit-learn numpy matplotlib seaborn pillow

# ─────────────────────────────────────────────────────────────
# SECTION 1: Imports
# ─────────────────────────────────────────────────────────────
import os
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from pathlib import Path
from PIL import Image
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# SECTION 2: Configuration
# ─────────────────────────────────────────────────────────────
IMG_SIZE    = (224, 224)   # Resize all images to this size (increased from 128)
CHANNELS    = 3            # RGB
BATCH_SIZE  = 16           # Smaller batch size for better gradient estimates
EPOCHS      = 50           # More epochs (EarlyStopping will prevent overfitting)
TEST_SIZE   = 0.2
RANDOM_SEED = 42
CLASS_NAMES = {0: "Cat", 1: "Dog"}

# ─────────────────────────────────────────────────────────────
# SECTION 3: Download Dataset via KaggleHub
# ─────────────────────────────────────────────────────────────
def download_dataset():
    """Download the Cats and Dogs dataset from Kaggle."""
    print("=" * 60)
    print("STEP 1: Downloading Dataset")
    print("=" * 60)
    try:
        import kagglehub
        path = kagglehub.dataset_download(
            "samuelcortinhas/cats-and-dogs-image-classification"
        )
        print(f"✔ Dataset downloaded to: {path}")
        return path
    except Exception as e:
        print(f"✘ KaggleHub download failed: {e}")
        print("  → Trying to locate a local copy...")
        # Fallback: look for a local 'data' folder
        local = Path("data")
        if local.exists():
            print(f"  ✔ Found local data folder at: {local.resolve()}")
            return str(local)
        sys.exit("No dataset found. Please download manually or configure Kaggle API.")


# ─────────────────────────────────────────────────────────────
# SECTION 4: Discover Image Paths
# ─────────────────────────────────────────────────────────────
def find_image_files(dataset_path):
    """
    Walk the dataset directory and collect (image_path, label) pairs.
    Label: 0 = cat, 1 = dog  (detected from folder name or filename).
    """
    print("\n" + "=" * 60)
    print("STEP 2: Scanning Dataset Directory")
    print("=" * 60)

    image_paths, labels = [], []
    root = Path(dataset_path)
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    # Strategy A: labelled sub-folders  (train/cat/, train/dog/, etc.)
    for folder in sorted(root.rglob("*")):
        if folder.is_dir():
            folder_lower = folder.name.lower()
            if "cat" in folder_lower:
                label = 0
            elif "dog" in folder_lower:
                label = 1
            else:
                continue  # skip non-cat/dog directories
            for img_path in folder.iterdir():
                if img_path.suffix.lower() in extensions:
                    image_paths.append(img_path)
                    labels.append(label)

    # Strategy B: flat directory – infer label from filename
    if not image_paths:
        print("  No labelled sub-folders found; inferring labels from filenames...")
        for img_path in root.rglob("*"):
            if img_path.suffix.lower() in extensions:
                name = img_path.stem.lower()
                if "cat" in name:
                    image_paths.append(img_path)
                    labels.append(0)
                elif "dog" in name:
                    image_paths.append(img_path)
                    labels.append(1)

    counts = Counter(labels)
    print(f"  Found {len(image_paths)} images  |  Cats: {counts[0]}  |  Dogs: {counts[1]}")
    return image_paths, np.array(labels)


# ─────────────────────────────────────────────────────────────
# SECTION 5: Preprocess Images
# ─────────────────────────────────────────────────────────────
def load_and_preprocess_images(image_paths, labels, img_size=None):
    if img_size is None:
        img_size = IMG_SIZE
    """
    Load each image, resize it, normalise pixel values to [0, 1].
    Returns:
        X  – float32 array  (N, H, W, C)
        y  – int array      (N,)
    """
    print("\n" + "=" * 60)
    print("STEP 3: Loading & Preprocessing Images")
    print("=" * 60)

    X, y_valid = [], []
    skipped = 0

    for idx, (img_path, label) in enumerate(zip(image_paths, labels)):
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize(img_size, Image.LANCZOS)
            arr = np.array(img, dtype=np.float32) / 255.0  # normalise
            X.append(arr)
            y_valid.append(label)
        except Exception:
            skipped += 1
            continue

        if (idx + 1) % 200 == 0:
            print(f"  Processed {idx + 1}/{len(image_paths)} images …")

    print(f"\n  ✔ Loaded  : {len(X)} images")
    print(f"  ✘ Skipped : {skipped} (corrupt/unreadable)")

    X = np.array(X, dtype=np.float32)
    y = np.array(y_valid, dtype=np.int32)
    print(f"  Dataset shape : {X.shape}  |  Labels shape : {y.shape}")
    return X, y


# ─────────────────────────────────────────────────────────────
# SECTION 6: Balance Classes
# ─────────────────────────────────────────────────────────────
def balance_classes(X, y, random_seed=RANDOM_SEED):
    """Under-sample the majority class so both classes are equal."""
    print("\n" + "=" * 60)
    print("STEP 4: Balancing Classes")
    print("=" * 60)

    counts = Counter(y)
    min_count = min(counts.values())

    X_bal, y_bal = [], []
    for cls in [0, 1]:
        idx = np.where(y == cls)[0]
        np.random.seed(random_seed)
        chosen = np.random.choice(idx, min_count, replace=False)
        X_bal.append(X[chosen])
        y_bal.append(y[chosen])

    X_bal = np.concatenate(X_bal, axis=0)
    y_bal = np.concatenate(y_bal, axis=0)

    # Shuffle
    perm = np.random.permutation(len(y_bal))
    X_bal, y_bal = X_bal[perm], y_bal[perm]

    print(f"  Per-class count : {min_count}  |  Total : {len(y_bal)}")
    return X_bal, y_bal


# ─────────────────────────────────────────────────────────────
# SECTION 7: Train / Test Split
# ─────────────────────────────────────────────────────────────
def split_data(X, y):
    """Split data into training and test sets (80/20)."""
    print("\n" + "=" * 60)
    print("STEP 5: Splitting into Train & Test Sets")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    print(f"  Train : {X_train.shape[0]} samples")
    print(f"  Test  : {X_test.shape[0]} samples")
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────────────────────
# SECTION 8: Flatten + Scale (for classical ML models)
# ─────────────────────────────────────────────────────────────
def prepare_flat_features(X_train, X_test):
    """
    Flatten 3-D image arrays to 1-D feature vectors and apply
    StandardScaler (zero-mean, unit-variance) for classical models.
    """
    X_tr_flat = X_train.reshape(len(X_train), -1)
    X_te_flat = X_test.reshape(len(X_test), -1)

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_flat)
    X_te_scaled  = scaler.transform(X_te_flat)
    return X_tr_scaled, X_te_scaled


# ─────────────────────────────────────────────────────────────
# SECTION 9: Metrics Helper
# ─────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, model_name):
    """Compute and print standard classification metrics."""
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    print(f"\n  {'Metric':<12} {'Value':>8}")
    print(f"  {'-'*22}")
    print(f"  {'Accuracy':<12} {acc:>8.4f}")
    print(f"  {'Precision':<12} {prec:>8.4f}")
    print(f"  {'Recall':<12} {rec:>8.4f}")
    print(f"  {'F1-Score':<12} {f1:>8.4f}")
    return {"Model": model_name, "Accuracy": acc, "Precision": prec,
            "Recall": rec, "F1-Score": f1, "y_pred": y_pred}


# ─────────────────────────────────────────────────────────────
# SECTION 10: Model 1 – Logistic Regression
# ─────────────────────────────────────────────────────────────
def train_logistic_regression(X_train, X_test, y_train, y_test):
    print("\n" + "=" * 60)
    print("MODEL 1: Logistic Regression (Baseline)")
    print("=" * 60)

    model = LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_SEED)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    result = compute_metrics(y_test, y_pred, "Logistic Regression")
    print("\n" + classification_report(y_test, y_pred,
                                       target_names=["Cat", "Dog"]))
    return result, model


# ─────────────────────────────────────────────────────────────
# SECTION 11: Model 2 – Support Vector Machine
# ─────────────────────────────────────────────────────────────
def train_svm(X_train, X_test, y_train, y_test):
    print("\n" + "=" * 60)
    print("MODEL 2: Support Vector Machine (SVM)")
    print("=" * 60)

    # Use a linear kernel for speed; change to 'rbf' for potentially higher accuracy
    model = SVC(kernel="linear", C=1.0, probability=True, random_state=RANDOM_SEED)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    result = compute_metrics(y_test, y_pred, "SVM")
    print("\n" + classification_report(y_test, y_pred,
                                       target_names=["Cat", "Dog"]))
    return result, model


# ─────────────────────────────────────────────────────────────
# SECTION 12: Model 3 – Random Forest
# ─────────────────────────────────────────────────────────────
def train_random_forest(X_train, X_test, y_train, y_test):
    print("\n" + "=" * 60)
    print("MODEL 3: Random Forest")
    print("=" * 60)

    model = RandomForestClassifier(
        n_estimators=200, max_depth=20, n_jobs=-1, random_state=RANDOM_SEED
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    result = compute_metrics(y_test, y_pred, "Random Forest")
    print("\n" + classification_report(y_test, y_pred,
                                       target_names=["Cat", "Dog"]))
    return result, model


# ─────────────────────────────────────────────────────────────
# SECTION 13: Model 4 – Convolutional Neural Network (CNN)
# ─────────────────────────────────────────────────────────────
def build_cnn(input_shape):
    """Define a transfer learning model using MobileNetV2 for better accuracy."""
    # Load pre-trained MobileNetV2 backbone (trained on ImageNet)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze the pre-trained weights initially
    base_model.trainable = False
    
    # Build custom classifier head
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        
        # Dense layers for classification
        layers.Dense(512, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(1, activation="sigmoid")   # Binary output
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model, base_model


def train_cnn(X_train, X_test, y_train, y_test):
    """Train CNN with transfer learning, advanced augmentation, and fine-tuning."""
    print("\n" + "=" * 60)
    print("MODEL 4: Transfer Learning CNN (MobileNetV2)")
    print("=" * 60)

    # Enhanced data augmentation for better generalization
    aug_gen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,        # Added
        rotation_range=25,         # Increased from 20
        zoom_range=0.3,            # Increased from 0.2
        width_shift_range=0.15,    # Increased from 0.1
        height_shift_range=0.15,   # Increased from 0.1
        shear_range=0.2,           # Added
        fill_mode='nearest',
        validation_split=0.1,      # 10% validation split
    )

    train_gen = aug_gen.flow(X_train, y_train, batch_size=BATCH_SIZE,
                             subset="training",   seed=RANDOM_SEED)
    val_gen   = aug_gen.flow(X_train, y_train, batch_size=BATCH_SIZE,
                             subset="validation", seed=RANDOM_SEED)

    input_shape = (IMG_SIZE[0], IMG_SIZE[1], CHANNELS)
    print(f"\n📏 Building model with input shape: {input_shape}")
    cnn, base_model = build_cnn(input_shape)
    cnn.summary()

    # Phase 1: Train only the classifier head (frozen backbone)
    print("\n→ Phase 1: Training classifier head (frozen backbone)...")
    callbacks_phase1 = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=0),
    ]

    history_phase1 = cnn.fit(
        train_gen,
        validation_data=val_gen,
        epochs=min(15, EPOCHS),
        callbacks=callbacks_phase1,
        verbose=1,
    )

    # Phase 2: Fine-tune the last layers of the backbone
    print("\n→ Phase 2: Fine-tuning last layers of backbone...")
    base_model.trainable = True
    # Freeze all but the last 30 layers
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    # Recompile with lower learning rate for fine-tuning
    cnn.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    callbacks_phase2 = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=0),
    ]

    history_phase2 = cnn.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks_phase2,
        verbose=1,
    )

    # Combine histories
    history = type('History', (), {})()
    history.history = {
        'accuracy': history_phase1.history['accuracy'] + history_phase2.history['accuracy'],
        'val_accuracy': history_phase1.history['val_accuracy'] + history_phase2.history['val_accuracy'],
        'loss': history_phase1.history['loss'] + history_phase2.history['loss'],
        'val_loss': history_phase1.history['val_loss'] + history_phase2.history['val_loss'],
    }

    # Evaluate on the held-out test set
    print(f"\n🔍 Predicting on test set with shape: {X_test.shape}")
    y_prob = cnn.predict(X_test, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    result = compute_metrics(y_test, y_pred, "CNN (Transfer Learning)")
    print("\n" + classification_report(y_test, y_pred,
                                       target_names=["Cat", "Dog"]))
    return result, cnn, history


# ─────────────────────────────────────────────────────────────
# SECTION 14: Visualisations
# ─────────────────────────────────────────────────────────────
def plot_confusion_matrices(results, y_test):
    """Plot a confusion matrix for each model side-by-side."""
    print("\nGenerating confusion matrices …")
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 5))
    if len(results) == 1:
        axes = [axes]

    for ax, res in zip(axes, results):
        cm = confusion_matrix(y_test, res["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Cat", "Dog"],
                    yticklabels=["Cat", "Dog"], ax=ax)
        ax.set_title(f"{res['Model']}\nAccuracy={res['Accuracy']:.3f}",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("Actual", fontsize=11)

    plt.suptitle("Confusion Matrices — All Models", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  ✔ Saved: confusion_matrices.png")


def plot_cnn_history(history):
    """Plot training vs validation accuracy and loss for the CNN."""
    print("\nGenerating CNN training curves …")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs_ran = range(1, len(history.history["accuracy"]) + 1)

    # Accuracy
    ax1.plot(epochs_ran, history.history["accuracy"],     label="Train Acc",   color="#2196F3")
    ax1.plot(epochs_ran, history.history["val_accuracy"], label="Val Acc",     color="#F44336", linestyle="--")
    ax1.set_title("CNN – Accuracy", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    # Loss
    ax2.plot(epochs_ran, history.history["loss"],     label="Train Loss", color="#4CAF50")
    ax2.plot(epochs_ran, history.history["val_loss"], label="Val Loss",   color="#FF9800", linestyle="--")
    ax2.set_title("CNN – Loss", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.suptitle("CNN Training History", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("cnn_training_history.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  ✔ Saved: cnn_training_history.png")


def plot_model_comparison(results):
    """Grouped bar chart comparing all models on 4 metrics."""
    print("\nGenerating model comparison chart …")
    metrics  = ["Accuracy", "Precision", "Recall", "F1-Score"]
    models   = [r["Model"] for r in results]
    colours  = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]

    x      = np.arange(len(models))
    width  = 0.18
    offsets = [-1.5, -0.5, 0.5, 1.5]

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, (metric, colour, offset) in enumerate(zip(metrics, colours, offsets)):
        vals = [r[metric] for r in results]
        bars = ax.bar(x + offset * width, vals, width, label=metric,
                      color=colour, alpha=0.85, edgecolor="white")
        for bar in bars:
            ax.annotate(f"{bar.get_height():.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7.5)

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison – All Metrics", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=0.5, color="red", linestyle=":", alpha=0.5, label="Random baseline")

    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  ✔ Saved: model_comparison.png")


def show_sample_predictions(X_test, y_test, cnn_model, n=12):
    """Display sample test images with true vs predicted labels from CNN."""
    print("\nGenerating sample predictions …")
    y_prob = cnn_model.predict(X_test[:n], verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.2))
    axes = axes.flatten()

    for i in range(n):
        img = (X_test[i] * 255).astype(np.uint8)
        true_lbl  = CLASS_NAMES[y_test[i]]
        pred_lbl  = CLASS_NAMES[y_pred[i]]
        confidence = y_prob[i] if y_pred[i] == 1 else 1 - y_prob[i]

        axes[i].imshow(img)
        axes[i].axis("off")
        colour = "green" if true_lbl == pred_lbl else "red"
        axes[i].set_title(
            f"True : {true_lbl}\nPred : {pred_lbl} ({confidence:.0%})",
            fontsize=8, color=colour, fontweight="bold"
        )

    # Hide any extra subplots
    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.suptitle("CNN Sample Predictions  (Green = Correct, Red = Wrong)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("sample_predictions.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  ✔ Saved: sample_predictions.png")


# ─────────────────────────────────────────────────────────────
# SECTION 15: Final Comparison Table
# ─────────────────────────────────────────────────────────────
def save_models(cnn_model):
    """Save the trained CNN model to disk."""
    print("\n" + "=" * 60)
    print("SAVING TRAINED MODEL")
    print("=" * 60)
    
    try:
        model_path = "cats_dogs_cnn.keras"
        cnn_model.save(model_path)
        print(f"✔ CNN model saved successfully to: {model_path}")
    except Exception as e:
        print(f"✘ Error saving model: {e}")


def print_comparison_table(results):
    """Print a neatly formatted comparison table and identify the best model."""
    print("\n" + "=" * 68)
    print("FINAL MODEL COMPARISON TABLE")
    print("=" * 68)
    header = f"{'Model':<22} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}"
    print(header)
    print("-" * 68)
    for r in results:
        print(f"  {r['Model']:<20} {r['Accuracy']:>10.4f} {r['Precision']:>10.4f}"
              f" {r['Recall']:>10.4f} {r['F1-Score']:>10.4f}")
    print("=" * 68)

    best = max(results, key=lambda r: r["F1-Score"])
    print(f"\n🏆 Best Model: {best['Model']}  (F1-Score = {best['F1-Score']:.4f})")
    print("""
WHY CNN TYPICALLY WINS:
  • Classical models (LR, SVM, RF) rely on hand-crafted pixel features after
    flattening images – spatial relationships are lost.
  • CNNs learn hierarchical, spatially-aware features via convolution:
      Layer 1 → edges & colours
      Layer 2 → textures & shapes
      Layer 3 → object parts (ears, snout, fur patterns)
  • Data augmentation further improves CNN generalisation.
  • For raw pixel inputs, deep learning dominates classical ML.

WHEN CLASSICAL MODELS ARE PREFERRED:
  • Small dataset (< 500 samples) – CNNs overfit easily.
  • Limited compute – LR/SVM are much faster to train.
  • Interpretability required – Random Forest feature importances are useful.
""")


# ─────────────────────────────────────────────────────────────
# SECTION 16: Main Pipeline
# ─────────────────────────────────────────────────────────────
def main():
    print("\n" + "🐱" * 15 + "  CATS vs DOGS  " + "🐶" * 15)
    print("   Multi-Model Image Classification Project\n")

    # ── 1. Download ──────────────────────────────────────────
    dataset_path = download_dataset()

    # ── 2. Discover images ───────────────────────────────────
    image_paths, raw_labels = find_image_files(dataset_path)
    if len(image_paths) == 0:
        sys.exit("No images found. Check the dataset path.")

    # ── 3. Load & preprocess ─────────────────────────────────
    print(f"\n📏 Using image size: {IMG_SIZE}")
    X, y = load_and_preprocess_images(image_paths, raw_labels, img_size=IMG_SIZE)
    print(f"✓ Loaded images shape: {X.shape}")

    # ── 4. Balance classes ───────────────────────────────────
    X, y = balance_classes(X, y)

    # ── 5. Split ─────────────────────────────────────────────
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"✓ Train shape: {X_train.shape} | Test shape: {X_test.shape}")

    # ── 6. Flat + scaled features (classical models) ─────────
    X_tr_scaled, X_te_scaled = prepare_flat_features(X_train, X_test)

    # ── 7. Train all models ──────────────────────────────────
    results = []

    lr_result, _    = train_logistic_regression(X_tr_scaled, X_te_scaled, y_train, y_test)
    results.append(lr_result)

    svm_result, _   = train_svm(X_tr_scaled, X_te_scaled, y_train, y_test)
    results.append(svm_result)

    rf_result, _    = train_random_forest(X_tr_scaled, X_te_scaled, y_train, y_test)
    results.append(rf_result)

    cnn_result, cnn_model, history = train_cnn(X_train, X_test, y_train, y_test)
    results.append(cnn_result)

    # ── 8. Visualisations ────────────────────────────────────
    plot_confusion_matrices(results, y_test)
    plot_cnn_history(history)
    plot_model_comparison(results)
    show_sample_predictions(X_test, y_test, cnn_model)

    # ── 9. Summary table ─────────────────────────────────────
    print_comparison_table(results)

    # ── 10. Save trained model ───────────────────────────────
    save_models(cnn_model)

    print("\n✅  Pipeline complete!  All plots saved as PNG files.\n")
	

# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()