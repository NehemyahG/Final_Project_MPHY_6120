"""
FINAL PROJECT - 3-CLASS BRAIN TUMOR TYPE CLASSIFICATION

This script is written as a staged project workflow:
1. Data audit and simple visualization on a small subset first.
2. Baseline CNN training and validation.
3. Transfer learning with ResNet18.
4. Test-set evaluation and error analysis.
5. Interpretability with filter visualization and Grad-CAM.

Run with:
    uv run python brain_tumor_prediction.py

Project note:
The dataset structure is a good fit for 3-class classification because the folders
represent tumor types. A binary setup would only make sense if a healthy class were
available, which is not the case here.
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve
from torch.utils.data import DataLoader, Subset
from torchvision import models
from torchvision.datasets import ImageFolder


# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "Brain_Cancer raw MRI data" / "Brain_Cancer"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Start small, validate the pipeline, then increase to 1.0 for the full dataset.
USE_SMALL_SUBSET = True
SMALL_SUBSET_FRACTION = 0.25

IMAGE_SIZE = 224
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15

BATCH_SIZE = 32
NUM_WORKERS = 0  # Keep this 0 on Windows for stability.
SEED = 42

CLASS_NAMES_HUMAN = {
    "brain_glioma": "Glioma",
    "brain_menin": "Meningioma",
    "brain_tumor": "Pituitary Tumor",
}


# =============================================================================
# Reproducibility and plotting defaults
# =============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 11


# =============================================================================
# Utility helpers
# =============================================================================

def ensure_output_dir() -> None:
    """Create the output folder once so all plots can be saved without errors."""
    OUTPUT_DIR.mkdir(exist_ok=True)


def unwrap_dataset(dataset: torch.utils.data.Dataset) -> ImageFolder:
    """Return the underlying ImageFolder even if the dataset is wrapped in Subset."""
    if isinstance(dataset, Subset):
        return dataset.dataset  # type: ignore[return-value]
    return dataset  # type: ignore[return-value]


def get_subset_indices(dataset: torch.utils.data.Dataset) -> np.ndarray:
    """Return the sample indices covered by a dataset or subset."""
    if isinstance(dataset, Subset):
        return np.asarray(dataset.indices)
    return np.arange(len(dataset))


def get_labels_from_dataset(dataset: torch.utils.data.Dataset) -> np.ndarray:
    """Extract labels from an ImageFolder or a Subset of an ImageFolder."""
    if isinstance(dataset, Subset):
        base_targets = np.asarray(dataset.dataset.targets)
        return base_targets[np.asarray(dataset.indices)]
    return np.asarray(dataset.targets)  # type: ignore[attr-defined]


def to_display_name(class_folder_name: str) -> str:
    """Map folder names to a clean label for plots and tables."""
    return CLASS_NAMES_HUMAN.get(class_folder_name, class_folder_name.replace("_", " ").title())


def denormalize_tensor(img: torch.Tensor) -> torch.Tensor:
    """Undo the [-1, 1] normalization so the image can be displayed."""
    return torch.clamp(img * 0.5 + 0.5, 0.0, 1.0)


def build_transforms(train: bool) -> transforms.Compose:
    """Build a transform pipeline for either training or evaluation.

    The medical-imaging reasoning is simple:
    - resize to a consistent input size,
    - convert to one grayscale channel,
    - normalize to a stable range,
    - and optionally augment only the training set.
    """
    ops = [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Grayscale(num_output_channels=1),
    ]

    if train:
        # These augmentations are intentionally modest because MRI structure
        # should be preserved. We only want small perturbations that mimic real
        # acquisition variation rather than unrealistic distortions.
        ops.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.05, 0.05),
                    scale=(0.95, 1.05),
                ),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
            ]
        )

    ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    return transforms.Compose(ops)

def estimate_patient_count(image_folder: ImageFolder) -> int:
    """Estimate unique patient/image IDs from the filename suffix.

    This dataset does not ship with explicit patient metadata in the folder
    structure, so we treat the numeric suffix as an estimate only.
    """
    ids = set()
    for file_path, _ in image_folder.samples:
        stem = Path(file_path).stem
        match = re.search(r"_(\d+)$", stem)
        if match:
            ids.add(int(match.group(1)))
    return len(ids)


def stratified_subset_indices(labels: np.ndarray, fraction: float) -> np.ndarray:
    """Take a class-balanced subset while keeping the original class ratios."""
    if fraction >= 1.0:
        return np.arange(len(labels))

    rng = np.random.default_rng(SEED)
    selected: List[int] = []
    for class_id in np.unique(labels):
        class_indices = np.flatnonzero(labels == class_id)
        take = max(1, int(round(len(class_indices) * fraction)))
        chosen = rng.choice(class_indices, size=take, replace=False)
        selected.extend(chosen.tolist())

    selected = sorted(selected)
    return np.asarray(selected)


def split_indices(labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split indices into train/validation/test with stratification."""
    all_indices = np.arange(len(labels))
    train_idx, temp_idx = train_test_split(
        all_indices,
        test_size=(1.0 - TRAIN_FRACTION),
        random_state=SEED,
        stratify=labels,
    )

    temp_labels = labels[temp_idx]
    val_share_of_temp = VAL_FRACTION / (VAL_FRACTION + TEST_FRACTION)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(1.0 - val_share_of_temp),
        random_state=SEED,
        stratify=temp_labels,
    )

    return np.asarray(train_idx), np.asarray(val_idx), np.asarray(test_idx)


def save_text_summary(filename: str, lines: Sequence[str]) -> None:
    """Write a lightweight audit note that can be included in the project folder."""
    ensure_output_dir()
    path = OUTPUT_DIR / filename
    path.write_text("\n".join(lines), encoding="utf-8")


def build_dataloaders(
    subset_fraction: float,
    batch_size: int = BATCH_SIZE,
) -> Tuple[DataLoader, DataLoader, DataLoader, Subset, Subset, List[str]]:
    """Create train/validation/test loaders from the ImageFolder structure.

    We first create a class-balanced subset if requested, then split that subset
    into train/validation/test partitions. This lets the project start on a small
    sample, catch bugs early, and then scale to the full dataset later.
    """
    base_for_labels = ImageFolder(DATA_ROOT, transform=None)
    full_labels = np.asarray(base_for_labels.targets)

    if subset_fraction < 1.0:
        working_indices = stratified_subset_indices(full_labels, subset_fraction)
        working_labels = full_labels[working_indices]
    else:
        working_indices = np.arange(len(full_labels))
        working_labels = full_labels

    # Split the working pool into train / validation / test.
    train_idx, val_idx, test_idx = split_indices(working_labels)

    # Convert the subset-relative indices back to dataset indices.
    train_indices = working_indices[train_idx]
    val_indices = working_indices[val_idx]
    test_indices = working_indices[test_idx]

    # Separate datasets for each split so we can assign different transforms.
    train_base = ImageFolder(DATA_ROOT, transform=build_transforms(train=True))
    eval_base = ImageFolder(DATA_ROOT, transform=build_transforms(train=False))

    train_dataset = Subset(train_base, train_indices.tolist())
    val_dataset = Subset(eval_base, val_indices.tolist())
    test_dataset = Subset(eval_base, test_indices.tolist())

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    class_names = list(train_base.classes)
    return train_loader, val_loader, test_loader, train_dataset, test_dataset, class_names


def show_tensor_grid(dataset: torch.utils.data.Dataset, class_names: Sequence[str], num_images: int = 16) -> None:
    """Display a small grid of samples so the user can sanity-check labels."""
    count = min(num_images, len(dataset))
    rows = int(np.ceil(np.sqrt(count)))
    cols = int(np.ceil(count / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_2d(axes)

    for plot_idx in range(rows * cols):
        ax = axes[plot_idx // cols, plot_idx % cols]
        if plot_idx >= count:
            ax.axis("off")
            continue

        sample_img, sample_label = dataset[plot_idx]
        img = denormalize_tensor(sample_img).squeeze().cpu().numpy()
        label_name = to_display_name(class_names[int(sample_label)])

        ax.imshow(img, cmap="gray")
        ax.set_title(label_name)
        ax.axis("off")

    plt.suptitle("Sample MRI Images From the Training Split")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "1_2_sample_grid.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_class_distribution(dataset: torch.utils.data.Dataset, class_names: Sequence[str]) -> None:
    """Plot the class balance so the dataset skew is visible at a glance."""
    labels = get_labels_from_dataset(dataset)
    counts = np.bincount(labels, minlength=len(class_names))
    total = counts.sum()
    percentages = counts / total * 100.0

    plt.figure(figsize=(8, 4))
    bars = plt.bar([to_display_name(c) for c in class_names], percentages, color="#4C78A8")
    plt.ylabel("Percentage of images")
    plt.title("Training Split Class Distribution")
    plt.ylim(0, max(percentages) * 1.15)

    for bar, value in zip(bars, percentages):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.1f}%", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "1_2_class_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()


# =============================================================================
# Part 1: Data Loading and Exploration
# =============================================================================

def task_1_1_load_data() -> Tuple[DataLoader, DataLoader, DataLoader, Subset, Subset, List[str]]:
    """Load the dataset and print a compact audit summary.

    The goal here is not just to load data, but to verify that the folder layout
    is usable for a 3-class experiment and that the class counts are sensible.
    """
    print("=" * 60)
    print("Task 1.1: Load the Dataset")
    print("=" * 60)

    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"Could not find dataset root: {DATA_ROOT}")

    subset_fraction = SMALL_SUBSET_FRACTION if USE_SMALL_SUBSET else 1.0
    train_loader, val_loader, test_loader, train_dataset, test_dataset, class_names = build_dataloaders(subset_fraction)

    base = unwrap_dataset(train_dataset)
    estimated_patients = estimate_patient_count(base)

    print(f"Dataset root: {DATA_ROOT}")
    print(f"Subset fraction used: {subset_fraction:.2f}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Image tensor shape: {train_dataset[0][0].shape}")
    print(f"Classes: {[to_display_name(c) for c in class_names]}")
    print(f"Estimated unique numeric IDs from filenames: {estimated_patients}")

    audit_lines = [
        f"Dataset root: {DATA_ROOT}",
        f"Subset fraction: {subset_fraction:.2f}",
        f"Train samples: {len(train_dataset)}",
        f"Validation samples: {len(val_loader.dataset)}",
        f"Test samples: {len(test_loader.dataset)}",
        f"Classes: {[to_display_name(c) for c in class_names]}",
        f"Estimated unique numeric IDs from filenames: {estimated_patients}",
    ]
    save_text_summary("1_1_data_audit.txt", audit_lines)

    return train_loader, val_loader, test_loader, train_dataset, test_dataset, class_names


def task_1_2_visualize_data(train_dataset: Subset, class_names: Sequence[str]) -> None:
    """Create a sample grid and class-balance plot for the training split."""
    print("\n" + "=" * 60)
    print("Exercise 1.2: Visualize the Data")
    print("=" * 60)

    show_tensor_grid(train_dataset, class_names, num_images=16)
    plot_class_distribution(train_dataset, class_names)

    print("Saved: outputs/1_2_sample_grid.png")
    print("Saved: outputs/1_2_class_distribution.png")


def task_1_3_data_augmentation(train_dataset: Subset, class_names: Sequence[str]) -> DataLoader:
    """Build an augmented training loader and show what the augmentations do.

    The point of augmentation here is to make the model less sensitive to small
    shifts, flips, and exposure changes while preserving the anatomical signal.
    """
    print("\n" + "=" * 60)
    print("Exercise 1.3: Data Augmentation")
    print("=" * 60)

    augmented_base = ImageFolder(DATA_ROOT, transform=build_transforms(train=True))
    train_indices = get_subset_indices(train_dataset)
    augmented_train_dataset = Subset(augmented_base, train_indices.tolist())
    augmented_loader = DataLoader(
        augmented_train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    # Demonstrate the effect of the training transforms on the same raw image.
    raw_base = ImageFolder(DATA_ROOT, transform=None)
    sample_index = int(train_indices[0])
    raw_img, raw_label = raw_base[sample_index]

    resize = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))
    grayscale = transforms.Grayscale(num_output_channels=1)
    flip = transforms.RandomHorizontalFlip(p=1.0)
    rotate = transforms.RandomRotation(10)
    jitter = transforms.ColorJitter(brightness=0.15, contrast=0.15)

    pil_gray = grayscale(resize(raw_img))
    demo_images = [
        transforms.ToTensor()(pil_gray),
        transforms.ToTensor()(flip(pil_gray)),
        transforms.ToTensor()(rotate(pil_gray)),
        transforms.ToTensor()(jitter(pil_gray)),
        transforms.ToTensor()(pil_gray),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(16, 4))
    for ax, img_tensor, title in zip(axes, demo_images, ["Original", "Flip", "Rotate", "Jitter", "Base"]):
        ax.imshow(img_tensor.squeeze().numpy(), cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    plt.suptitle(f"Augmentation Examples for {to_display_name(class_names[int(raw_label)])}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "1_3_augmentation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: outputs/1_3_augmentation.png")

    save_text_summary(
        "1_3_augmentation_note.txt",
        [
            "Why augmentation matters in medical imaging:",
            "- It reduces overfitting when the dataset is small.",
            "- It teaches the model to ignore tiny acquisition differences that are not clinically meaningful.",
            "- It usually improves generalization more than training longer on a tiny fixed sample.",
        ],
    )

    return augmented_loader


# =============================================================================
# Part 2: Baseline CNN
# =============================================================================

class SimpleCNN(nn.Module):
    """A compact CNN baseline for 3-class tumor typing.

    This baseline is intentionally small so that we can compare it against the
    pretrained ResNet18 later. If the pretrained model wins, the comparison is
    meaningful because the baseline is not over-engineered.
    """

    def __init__(self, num_classes: int = 3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Adaptive pooling removes the need to hard-code the flattened size.
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.30),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
) -> Tuple[float, float]:
    """Run one training epoch and return loss + accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.long().to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return running_loss / max(total, 1), correct / max(total, 1)


def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate a model and return loss, accuracy, labels, predictions, and probabilities."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels: List[int] = []
    all_predictions: List[int] = []
    all_probabilities: List[np.ndarray] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.long().to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            probabilities = torch.softmax(logits, dim=1)
            predictions = probabilities.argmax(dim=1)

            running_loss += loss.item() * images.size(0)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())
            all_probabilities.extend(probabilities.cpu().numpy())

    return (
        running_loss / max(total, 1),
        correct / max(total, 1),
        np.asarray(all_labels),
        np.asarray(all_predictions),
        np.asarray(all_probabilities),
    )


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    patience: int = 4,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """Train a model with early stopping and a ReduceLROnPlateau scheduler."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, _, _, _ = evaluate_epoch(model, val_loader, criterion)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch}/{epochs} - "
            f"Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - "
            f"Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def plot_training_curves(history: Dict[str, List[float]], title: str, filename: str) -> None:
    """Plot loss and accuracy curves for a trained model."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{title} Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["train_acc"], label="Train Acc")
    axes[1].plot(history["val_acc"], label="Val Acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"{title} Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()


def task_2_2_training_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 8, #epochs: int = 8
    learning_rate: float = 1e-3,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """Train the baseline CNN on the augmented training split."""
    print("\n" + "=" * 60)
    print("Exercise 2.2: Training Loop")
    print("=" * 60)
    return train_model(model, train_loader, val_loader, epochs=epochs, learning_rate=learning_rate)


def task_2_3_training_curves(history: Dict[str, List[float]], model_name: str = "Baseline CNN") -> None:
    """Save training curves and add a concise interpretation note."""
    print("\n" + "=" * 60)
    print("Exercise 2.3: Training Curves")
    print("=" * 60)

    plot_training_curves(history, model_name, "2_3_training_curves.png")
    print("Saved: outputs/2_3_training_curves.png")

    save_text_summary(
        "2_3_training_curve_note.txt",
        [
            "How to read the curves:",
            "- If training loss keeps dropping while validation loss flattens or rises, the model is starting to overfit.",
            "- A small gap between training and validation is normal; a large gap means the model is memorizing instead of learning general patterns.",
            "- This is one reason to begin with a small subset first and confirm the pipeline before scaling up.",
        ],
    )


# =============================================================================
# Part 3: Transfer Learning
# =============================================================================

def build_resnet18_for_grayscale(num_classes: int = 3, pretrained: bool = True) -> nn.Module:
    """Build a ResNet18 and adapt it to single-channel MRI inputs.

    ResNet18 is a strong choice for this project because:
    - it is larger than the baseline CNN but still manageable,
    - pretrained weights usually transfer well when the dataset is small,
    - the residual blocks help optimization when we fine-tune the network.
    """
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    try:
        model = models.resnet18(weights=weights)
    except Exception as exc:
        # Some environments do not have network access for the pretrained weights.
        # Falling back to a scratch model keeps the project runnable instead of failing outright.
        print(f"Warning: could not load pretrained ResNet18 weights ({exc}). Falling back to random init.")
        model = models.resnet18(weights=None)

    # Adapt the first convolution to 1-channel images.
    old_weights = model.conv1.weight.data
    new_weights = old_weights.mean(dim=1, keepdim=True)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        model.conv1.weight.copy_(new_weights)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def task_3_1_pretrained_model(num_classes: int = 3) -> nn.Module:
    """Load the pretrained ResNet18 transfer-learning backbone."""
    print("\n" + "=" * 60)
    print("Exercise 3.1: Load Pretrained Model")
    print("=" * 60)

    model = build_resnet18_for_grayscale(num_classes=num_classes, pretrained=True)
    print("Modified ResNet18 architecture:")
    print(f"First conv: {model.conv1}")
    print(f"Final fc: {model.fc}")

    save_text_summary(
        "3_1_transfer_learning_note.txt",
        [
            "Why ImageNet pretraining helps:",
            "- Early layers already know how to detect edges, corners, blobs, and texture transitions.",
            "- Medical image datasets are often small, so starting from a pretrained model reduces the amount of data needed.",
            "- Fine-tuning lets the model adapt those generic features to tumor-specific patterns.",
        ],
    )

    return model


def task_3_2_fine_tuning(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """Two-stage fine-tuning: classifier-only first, then full-network adaptation."""
    print("\n" + "=" * 60)
    print("Exercise 3.2: Fine-Tuning Strategy")
    print("=" * 60)

    model = model.to(device)

    # Stage 1: freeze the backbone and train only the classifier head.
    print("\nStage 1: Training classifier only")
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    stage1_history = train_model(model, train_loader, val_loader, epochs=4, learning_rate=1e-3, patience=3)[1] #epochs = 4

    # Stage 2: unfreeze the full model and fine-tune with a smaller learning rate.
    print("\nStage 2: Fine-tuning the full model")
    for param in model.parameters():
        param.requires_grad = True

    stage2_model, stage2_history = train_model(model, train_loader, val_loader, epochs=6, learning_rate=1e-4, patience=3) #epochs = 6

    history = {
        "train_loss": stage1_history["train_loss"] + stage2_history["train_loss"],
        "val_loss": stage1_history["val_loss"] + stage2_history["val_loss"],
        "train_acc": stage1_history["train_acc"] + stage2_history["train_acc"],
        "val_acc": stage1_history["val_acc"] + stage2_history["val_acc"],
    }

    return stage2_model, history


def task_3_3_compare_models(
    simple_cnn: nn.Module,
    resnet_scratch: nn.Module,
    resnet_pretrained: nn.Module,
    test_loader: DataLoader,
    class_names: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    """Compare the baseline and transfer-learning models on the test set."""
    print("\n" + "=" * 60)
    print("Exercise 3.3: Compare Models")
    print("=" * 60)

    def evaluate_model(model: nn.Module, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        model.eval()
        y_true: List[int] = []
        y_pred: List[int] = []
        y_prob: List[np.ndarray] = []

        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(device)
                labels = labels.long().to(device)
                logits = model(images)
                probabilities = torch.softmax(logits, dim=1)
                predictions = probabilities.argmax(dim=1)

                y_true.extend(labels.cpu().tolist())
                y_pred.extend(predictions.cpu().tolist())
                y_prob.extend(probabilities.cpu().numpy())

        return np.asarray(y_true), np.asarray(y_pred), np.asarray(y_prob)

    metrics: Dict[str, Dict[str, float]] = {}
    models_to_compare = {
        "Simple CNN": simple_cnn,
        "ResNet18 Scratch": resnet_scratch,
        "ResNet18 Pretrained": resnet_pretrained,
    }

    for model_name, model in models_to_compare.items():
        y_true, y_pred, y_prob = evaluate_model(model.to(device), test_loader)
        metrics[model_name] = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "auc_ovr_macro": roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"),
        }

    print("\nModel Comparison")
    print("-" * 88)
    print(f"{'Model':<24} {'Acc':<8} {'Macro P':<10} {'Macro R':<10} {'Macro F1':<10} {'Macro AUC':<10}")
    print("-" * 88)
    for model_name, values in metrics.items():
        print(
            f"{model_name:<24} {values['accuracy']:<8.4f} {values['precision_macro']:<10.4f} "
            f"{values['recall_macro']:<10.4f} {values['f1_macro']:<10.4f} {values['auc_ovr_macro']:<10.4f}"
        )

    save_text_summary(
        "3_3_model_comparison_note.txt",
        [
            "How to interpret the comparison:",
            "- The baseline CNN is the sanity-check model.",
            "- ResNet18 from scratch shows whether deeper architecture alone helps.",
            "- ResNet18 pretrained is the main transfer-learning candidate and should usually win when the dataset is small.",
        ],
    )

    return metrics


# =============================================================================
# Part 4: Model Evaluation
# =============================================================================

def compute_confusion_matrix(
    model: nn.Module,
    model_name: str,
    test_loader: DataLoader,
    class_names: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Plot the confusion matrix and print a detailed classification report."""
    print("\n" + "=" * 60)
    print("Exercise 4.1: Confusion Matrix & Metrics")
    print("=" * 60)

    criterion = nn.CrossEntropyLoss()
    loss, acc, y_true, y_pred, y_prob = evaluate_epoch(model.to(device), test_loader, criterion)
    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {acc:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    im = plt.imshow(cm, cmap=plt.cm.Blues)
    plt.colorbar(im)
    plt.xticks(ticks=np.arange(len(class_names)), labels=[to_display_name(c) for c in class_names], rotation=20)
    plt.yticks(ticks=np.arange(len(class_names)), labels=[to_display_name(c) for c in class_names])
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(f"{model_name} : Confusion Matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"4_1_confusion_matrix_{model_name}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: outputs/4_1_confusion_matrix_{model_name}.png")

    print("\nClassification Report")
    print(classification_report(y_true, y_pred, target_names=[to_display_name(c) for c in class_names], zero_division=0))

    save_text_summary(
        "4_1_evaluation_note.txt",
        [
            "In a medical classification setting, recall is often the most important metric when missed cases are costly.",
            "For a 3-class tumor project, macro-F1 and the confusion matrix are also important because they show whether one tumor type is being ignored.",
        ],
    )

    return y_true, y_prob, y_pred

def model_calibration_plot(
    model_name: str,
    labels: np.ndarray,
    probs: np.ndarray,
    class_names: Sequence[str],
) -> None:
    """Plot reliability diagrams to check if predicted probabilities are well-calibrated."""
    print("\n" + "=" * 60)
    print("Exercise 4.2: Model Calibration")
    print("=" * 60)

    n_classes = len(class_names)
    y_true_bin = label_binarize(labels, classes=list(range(n_classes)))

    brier_log = brier_score_loss(y_true_bin.ravel(), probs.ravel())
    print(f"Brier score loss: {brier_log:.4f}")
    
    plt.figure(figsize=(6, 6))
    for class_id in range(n_classes):
        prob_true, prob_pred = calibration_curve(y_true_bin[:, class_id], probs[:, class_id], n_bins=5)
        plt.plot(prob_pred, prob_true, marker="o", label=to_display_name(class_names[class_id]))
        
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title(f"{model_name}: Calibration Curves ({brier_log:.4f} Brier Loss)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"4_2_calibration_curves_{model_name}.png", dpi=150, bbox_inches="tight") # Include model used on the file name
    plt.close()
    print(f"Saved: outputs/4_2_calibration_curves_{model_name}.png")
    
    
def test_roc_pr_curves(
    model_name: str,
    labels: np.ndarray,
    probs: np.ndarray,
    class_names: Sequence[str]
    ) -> None:
    """Plot ROC and precision-recall curves using one-vs-rest targets."""
    print("\n" + "=" * 60)
    print("Exercise 4.2: ROC and PR Curves")
    print("=" * 60)

    n_classes = len(class_names)
    y_true_bin = label_binarize(labels, classes=list(range(n_classes)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ROC curves
    for class_id in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, class_id], probs[:, class_id])
        auc_score = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, label=f"{to_display_name(class_names[class_id])} (AUC={auc_score:.3f})")

    fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), probs.ravel())
    auc_micro = auc(fpr_micro, tpr_micro)
    axes[0].plot(fpr_micro, tpr_micro, linestyle="--", color="black", label=f"Micro-average (AUC={auc_micro:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title(f"{model_name}: ROC Curves")
    axes[0].legend(fontsize=8)

    # Precision-recall curves
    for class_id in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, class_id], probs[:, class_id])
        pr_auc = average_precision_score(y_true_bin[:, class_id], probs[:, class_id])
        axes[1].plot(recall, precision, label=f"{to_display_name(class_names[class_id])} (AP={pr_auc:.3f})")

    precision_micro, recall_micro, _ = precision_recall_curve(y_true_bin.ravel(), probs.ravel())
    ap_micro = average_precision_score(y_true_bin, probs, average="micro")
    axes[1].plot(recall_micro, precision_micro, linestyle="--", color="black", label=f"Micro-average (AP={ap_micro:.3f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"{model_name}: Precision-Recall Curves")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"4_2_roc_pr_curves_{model_name}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: outputs/4_2_roc_pr_curves_{model_name}.png")


def error_analysis(
    model: nn.Module,
    model_name: str,
    test_dataset: Subset,
    class_names: Sequence[str],
) -> None:
    """Show examples of correct and incorrect predictions.

    The old binary TP/TN/FP/FN framing does not fit a 3-class problem cleanly,
    so this version shows correct examples and misclassifications with labels.
    """
    print("\n" + "=" * 60)
    print("Exercise 4.3: Error Analysis")
    print("=" * 60)

    model.eval()
    correct_examples: List[Tuple[torch.Tensor, int, int, float]] = []
    wrong_examples: List[Tuple[torch.Tensor, int, int, float]] = []

    loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.long().to(device)
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)
            pred = int(probabilities.argmax(dim=1).item())
            true = int(labels.item())
            confidence = float(probabilities[0, pred].item())
            image_cpu = images.cpu().squeeze(0)

            if pred == true and len(correct_examples) < 4:
                correct_examples.append((image_cpu, true, pred, confidence))
            elif pred != true and len(wrong_examples) < 4:
                wrong_examples.append((image_cpu, true, pred, confidence))

            if len(correct_examples) == 4 and len(wrong_examples) == 4:
                break

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f"{model_name} Error Analysis: Correct vs Misclassified Examples")

    def plot_example(ax, item: Tuple[torch.Tensor, int, int, float]) -> None:
        image, true_label, pred_label, confidence = item
        ax.imshow(denormalize_tensor(image).squeeze().numpy(), cmap="gray")
        ax.set_title(
            f"T: {to_display_name(class_names[true_label])}\n"
            f"P: {to_display_name(class_names[pred_label])} ({confidence:.2f})",
            fontsize=9,
        )
        ax.axis("off")

    for col in range(4):
        if col < len(correct_examples):
            plot_example(axes[0, col], correct_examples[col])
        else:
            axes[0, col].axis("off")

        if col < len(wrong_examples):
            plot_example(axes[1, col], wrong_examples[col])
        else:
            axes[1, col].axis("off")

    axes[0, 0].set_ylabel("Correct", fontsize=12)
    axes[1, 0].set_ylabel("Misclassified", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"4_3_error_analysis_{model_name}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: outputs/4_3_error_analysis_{model_name}.png")


# =============================================================================
# Part 5: Interpretability
# =============================================================================

def visualize_filters(model_name: str, model: nn.Module) -> None:
    """Visualize the first convolutional filters learned by the model."""
    print("\n" + "=" * 60)
    print("Exercise 5.1: Visualize Filters")
    print("=" * 60)

    if isinstance(model, SimpleCNN):
        filters = model.features[0].weight.detach().cpu().numpy()
    elif isinstance(model, models.ResNet):
        filters = model.conv1.weight.detach().cpu().numpy()
    else:
        print("Unsupported model type for filter visualization.")
        return

    num_filters = min(filters.shape[0], 32)
    num_cols = 8
    num_rows = int(np.ceil(num_filters / num_cols))
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2))
    axes = np.atleast_2d(axes)

    for idx in range(num_rows * num_cols):
        ax = axes[idx // num_cols, idx % num_cols]
        if idx >= num_filters:
            ax.axis("off")
            continue
        filter_img = filters[idx, 0]
        ax.imshow(filter_img, cmap="gray")
        ax.axis("off")

    plt.suptitle(f"{model_name} Learned Filters from the First Convolutional Layer")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"5_1_{model_name}_filters.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: outputs/5_1_{model_name}_filters.png")


def get_gradcam_target_layer(model: nn.Module) -> nn.Module:
    """Return the layer that makes the most sense for Grad-CAM."""
    if isinstance(model, SimpleCNN):
        # The last convolution in the baseline feature stack.
        return model.features[8]
    if isinstance(model, models.ResNet):
        return model.layer4[-1]
    raise TypeError("Unsupported model type for Grad-CAM")


def task_5_2_gradcam(model: nn.Module, test_dataset: Subset, class_names: Sequence[str]) -> None:
    """Generate Grad-CAM heatmaps for a few test images."""
    print("\n" + "=" * 60)
    print("Exercise 5.2: Grad-CAM Visualization")
    print("=" * 60)

    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = True

    target_layer = get_gradcam_target_layer(model)
    num_examples = 3
    fig, axes = plt.subplots(3, num_examples, figsize=(14, 10))
    fig.suptitle("Grad-CAM Visualizations", fontsize=16)

    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        shown = 0
        for idx in range(len(test_dataset)):
            img, label = test_dataset[idx]
            input_tensor = img.unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(input_tensor)
                probabilities = torch.softmax(logits, dim=1)
                pred_class = int(probabilities.argmax(dim=1).item())
                pred_prob = float(probabilities[0, pred_class].item())

            grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(pred_class)])[0]
            grayscale_cam = grayscale_cam / max(grayscale_cam.max(), 1e-8)

            img_for_display = denormalize_tensor(img).cpu().numpy()
            rgb_img = np.repeat(np.transpose(img_for_display, (1, 2, 0)), 3, axis=2)
            cam_overlay = show_cam_on_image(rgb_img.astype(np.float32), grayscale_cam, use_rgb=True, image_weight=0.65)

            axes[0, shown].imshow(img_for_display.squeeze(), cmap="gray")
            axes[0, shown].set_title(f"True: {to_display_name(class_names[int(label)])}")
            axes[0, shown].axis("off")

            axes[1, shown].imshow(grayscale_cam, cmap="jet")
            axes[1, shown].set_title("Grad-CAM Heatmap")
            axes[1, shown].axis("off")

            axes[2, shown].imshow(cam_overlay)
            axes[2, shown].set_title(
                f"Pred: {to_display_name(class_names[pred_class])} ({pred_prob:.2f})"
            )
            axes[2, shown].axis("off")

            shown += 1
            if shown >= num_examples:
                break

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(OUTPUT_DIR / "5_2_gradcam.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: outputs/5_2_gradcam.png")
    
def task_5_3_occlusion_sensitivity(
    model: nn.Module,
    model_name: str,
    test_dataset: Subset,
    class_names: Sequence[str],
    patch_size: int = 16,
    stride: int = 8,
    num_examples: int = 3) -> None:
    """Occlusion sensitivity: mask image patches and measure drop in confidence."""
    print("\n" + "=" * 60)
    print("Exercise 5.3: Occlusion Sensitivity")
    print("=" * 60)

    model = model.to(device)
    model.eval()

    fig, axes = plt.subplots(3, num_examples, figsize=(14, 10))
    fig.suptitle(f"{model_name}: Occlusion Sensitivity", fontsize=16)

    shown = 0

    for idx in range(len(test_dataset)):
        img, label = test_dataset[idx]
        input_tensor = img.unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1)
            pred_class = int(probs.argmax(dim=1).item())
            original_prob = float(probs[0, pred_class].item())

        _, h, w = img.shape
        sensitivity_map = np.zeros((h, w), dtype=np.float32)
        count_map = np.zeros((h, w), dtype=np.float32)

        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                occluded = input_tensor.clone()

                # Since images are normalized to [-1, 1], value 0 means neutral gray.
                occluded[:, :, y:y + patch_size, x:x + patch_size] = 0.0

                with torch.no_grad():
                    occluded_logits = model(occluded)
                    occluded_probs = torch.softmax(occluded_logits, dim=1)
                    occluded_prob = float(occluded_probs[0, pred_class].item())

                confidence_drop = original_prob - occluded_prob

                sensitivity_map[y:y + patch_size, x:x + patch_size] += confidence_drop
                count_map[y:y + patch_size, x:x + patch_size] += 1

        sensitivity_map = sensitivity_map / np.maximum(count_map, 1e-8)

        # Normalize for display
        sensitivity_map = sensitivity_map - sensitivity_map.min()
        sensitivity_map = sensitivity_map / max(sensitivity_map.max(), 1e-8)

        img_display = denormalize_tensor(img).squeeze().cpu().numpy()

        axes[0, shown].imshow(img_display, cmap="gray")
        axes[0, shown].set_title(f"True: {to_display_name(class_names[int(label)])}")
        axes[0, shown].axis("off")

        axes[1, shown].imshow(sensitivity_map, cmap="hot")
        axes[1, shown].set_title("Occlusion Sensitivity")
        axes[1, shown].axis("off")

        axes[2, shown].imshow(img_display, cmap="gray")
        axes[2, shown].imshow(sensitivity_map, cmap="hot", alpha=0.45)
        axes[2, shown].set_title(
            f"Pred: {to_display_name(class_names[pred_class])}\n"
            f"Confidence: {original_prob:.2f}"
        )
        axes[2, shown].axis("off")

        shown += 1
        if shown >= num_examples:
            break

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(OUTPUT_DIR / f"5_3_occlusion_sensitivity_{model_name}.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved: outputs/5_3_occlusion_sensitivity_{model_name}.png")


# =============================================================================
# Main pipeline
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MPHY 6120 - Final Project: Brain Tumor Type Classification")
    print("=" * 60 + "\n")

    ensure_output_dir()

    # Part 1: data audit, visualization, and augmentation.
    train_loader, val_loader, test_loader, train_dataset, test_dataset, class_names = task_1_1_load_data()
    task_1_2_visualize_data(train_dataset, class_names)
    train_loader_aug = task_1_3_data_augmentation(train_dataset, class_names)

    # Part 2: baseline CNN.
    simple_cnn = SimpleCNN(num_classes=len(class_names)).to(device)
    print(f"\nSimpleCNN architecture:\n{simple_cnn}")
    simple_cnn, simple_history = task_2_2_training_loop(simple_cnn, train_loader_aug, val_loader, epochs=8, learning_rate=1e-3) #epochs = 8 
    task_2_3_training_curves(simple_history, model_name="Simple CNN")

    # Part 3: transfer learning with ResNet18.
    resnet_pretrained = task_3_1_pretrained_model(num_classes=len(class_names))
    resnet_pretrained, pretrained_history = task_3_2_fine_tuning(resnet_pretrained, train_loader_aug, val_loader)
    plot_training_curves(pretrained_history, "ResNet18 Pretrained", "3_2_pretrained_training_curves.png")
    print("Saved: outputs/3_2_pretrained_training_curves.png")

    # A scratch ResNet gives a fair transfer-learning comparison.
    resnet_scratch = build_resnet18_for_grayscale(num_classes=len(class_names), pretrained=False)
    resnet_scratch, scratch_history = train_model(resnet_scratch, train_loader_aug, val_loader, epochs=6, learning_rate=1e-3) #epochs = 6
    plot_training_curves(scratch_history, "ResNet18 Scratch", "3_2_scratch_training_curves.png")
    print("Saved: outputs/3_2_scratch_training_curves.png")

    task_3_3_compare_models(simple_cnn, resnet_scratch, resnet_pretrained, test_loader, class_names)

    # Part 4: test-set evaluation.
    evaluated_model = "resnet_pretrained"  # Choose which model to evaluate on the test set.
    labels, probs, _ = compute_confusion_matrix(resnet_pretrained, evaluated_model, test_loader, class_names)
    test_roc_pr_curves(evaluated_model, labels, probs, class_names)
    model_calibration_plot(evaluated_model, labels, probs, class_names)
    error_analysis(resnet_pretrained, evaluated_model, test_dataset, class_names)

    # Part 5: interpretability.
    visualize_filters(evaluated_model, resnet_pretrained)
    task_5_2_gradcam(resnet_pretrained, test_dataset, class_names)


    task_5_3_occlusion_sensitivity(resnet_pretrained, evaluated_model, test_dataset, class_names)

    print("\n" + "=" * 60)
    print("FINAL PROJECT COMPLETE")
    print("=" * 60)