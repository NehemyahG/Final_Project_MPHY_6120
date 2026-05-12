"""
Full-dataset runner for the MPHY 6120 brain tumor project.

This script reuses the core pipeline in brain_tumor_prediction.py but forces:
- full dataset usage (no small subset), and
- a separate output directory: outputs_full/
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import models
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

import brain_tumor_prediction as project


FULL_IMAGE_SIZE = 256


def build_transforms_full(train: bool) -> transforms.Compose:
    """Higher-resolution transforms with explicit anti-center-bias augmentation."""
    if train:
        ops = [
            transforms.Resize((288, 288)),
            transforms.RandomResizedCrop(FULL_IMAGE_SIZE, scale=(0.80, 1.0), ratio=(0.9, 1.1)),
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(12),
            transforms.RandomAffine(degrees=0, translate=(0.10, 0.10), scale=(0.95, 1.05)),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    else:
        ops = [
            transforms.Resize((FULL_IMAGE_SIZE, FULL_IMAGE_SIZE)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    return transforms.Compose(ops)


def build_group_keys(samples: Sequence[Tuple[str, int]]) -> np.ndarray:
    """Build class+ID group keys from sample file paths.

    This keeps train/validation/test disjoint by inferred patient/image ID.
    """
    keys: List[str] = []
    for file_path, _ in samples:
        path_obj = Path(file_path)
        class_folder = path_obj.parent.name
        stem = path_obj.stem
        match = re.search(r"_(\d+)$", stem)

        if match:
            keys.append(f"{class_folder}:{match.group(1)}")
        else:
            # Fall back to per-file key if no numeric suffix is present.
            keys.append(f"{class_folder}:{stem}")

    return np.asarray(keys)


def split_indices_grouped(labels: np.ndarray, group_keys: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split sample indices with strict group isolation across splits."""
    unique_groups, inverse = np.unique(group_keys, return_inverse=True)
    group_ids = np.arange(len(unique_groups))

    group_labels = np.zeros(len(unique_groups), dtype=int)
    for gid in group_ids:
        member_idx = np.flatnonzero(inverse == gid)
        group_labels[gid] = int(labels[member_idx[0]])

    train_gid, temp_gid = train_test_split(
        group_ids,
        test_size=(1.0 - project.TRAIN_FRACTION),
        random_state=project.SEED,
        stratify=group_labels,
    )

    temp_group_labels = group_labels[temp_gid]
    val_share_of_temp = project.VAL_FRACTION / (project.VAL_FRACTION + project.TEST_FRACTION)
    val_gid, test_gid = train_test_split(
        temp_gid,
        test_size=(1.0 - val_share_of_temp),
        random_state=project.SEED,
        stratify=temp_group_labels,
    )

    train_idx = np.flatnonzero(np.isin(inverse, train_gid))
    val_idx = np.flatnonzero(np.isin(inverse, val_gid))
    test_idx = np.flatnonzero(np.isin(inverse, test_gid))

    return np.asarray(train_idx), np.asarray(val_idx), np.asarray(test_idx)


def build_dataloaders_grouped(
    subset_fraction: float,
    batch_size: int = project.BATCH_SIZE,
) -> Tuple[DataLoader, DataLoader, DataLoader, Subset, Subset, List[str]]:
    """Build dataloaders with grouped (leakage-safe) split behavior."""
    base_for_labels = ImageFolder(project.DATA_ROOT, transform=None)
    full_labels = np.asarray(base_for_labels.targets)
    full_group_keys = build_group_keys(base_for_labels.samples)

    if subset_fraction < 1.0:
        working_indices = project.stratified_subset_indices(full_labels, subset_fraction)
        working_labels = full_labels[working_indices]
        working_group_keys = full_group_keys[working_indices]
    else:
        working_indices = np.arange(len(full_labels))
        working_labels = full_labels
        working_group_keys = full_group_keys

    train_idx, val_idx, test_idx = split_indices_grouped(working_labels, working_group_keys)

    train_groups = set(working_group_keys[train_idx].tolist())
    val_groups = set(working_group_keys[val_idx].tolist())
    test_groups = set(working_group_keys[test_idx].tolist())
    if (train_groups & val_groups) or (train_groups & test_groups) or (val_groups & test_groups):
        raise RuntimeError("Group leakage detected across train/val/test splits.")

    train_indices = working_indices[train_idx]
    val_indices = working_indices[val_idx]
    test_indices = working_indices[test_idx]

    train_base = ImageFolder(project.DATA_ROOT, transform=project.build_transforms(train=True))
    eval_base = ImageFolder(project.DATA_ROOT, transform=project.build_transforms(train=False))

    train_dataset = Subset(train_base, train_indices.tolist())
    val_dataset = Subset(eval_base, val_indices.tolist())
    test_dataset = Subset(eval_base, test_indices.tolist())

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=project.NUM_WORKERS,
        pin_memory=project.torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=project.NUM_WORKERS,
        pin_memory=project.torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=project.NUM_WORKERS,
        pin_memory=project.torch.cuda.is_available(),
    )

    class_names = list(train_base.classes)
    return train_loader, val_loader, test_loader, train_dataset, test_dataset, class_names


def get_earlier_gradcam_layer(model: torch.nn.Module) -> torch.nn.Module:
    """Pick an earlier conv block than the default final block for CAM."""
    if isinstance(model, models.ResNet):
        return model.layer3[-1]
    return project.get_gradcam_target_layer(model)


def normalize_stem_for_mask(stem: str) -> str:
    """Normalize filename stems so image/mask pairs can be matched more robustly."""
    cleaned = stem.lower()
    for token in ["_mask", "-mask", " mask", "_seg", "-seg", "_label", "-label", "_annotation", "-annotation"]:
        cleaned = cleaned.replace(token, "")
    return cleaned


def discover_mask_files(root: Path) -> Dict[str, Path]:
    """Scan for possible mask files and map normalized stem to path."""
    known_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    candidate_tokens = ("mask", "seg", "label", "annotation")
    stem_to_mask: Dict[str, Path] = {}

    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in known_exts:
            continue

        lower_name = path.name.lower()
        lower_parts = [part.lower() for part in path.parts]
        if any(token in lower_name for token in candidate_tokens) or any(token in part for token in candidate_tokens for part in lower_parts):
            key = normalize_stem_for_mask(path.stem)
            stem_to_mask.setdefault(key, path)

    return stem_to_mask


def load_mask_binary(mask_path: Path, shape_hw: Tuple[int, int]) -> np.ndarray:
    """Load mask image and convert it to a binary array aligned with CAM size."""
    mask_img = Image.open(mask_path).convert("L")
    mask_img = mask_img.resize((shape_hw[1], shape_hw[0]), resample=Image.BILINEAR)
    arr = np.asarray(mask_img, dtype=np.float32)
    arr = arr / max(arr.max(), 1e-8)
    return (arr > 0.5).astype(np.uint8)


def compute_overlap_metrics(mask_bin: np.ndarray, cam_bin: np.ndarray) -> Tuple[float, float]:
    """Return IoU and Dice overlap between binary mask and binary CAM region."""
    intersection = float(np.logical_and(mask_bin == 1, cam_bin == 1).sum())
    union = float(np.logical_or(mask_bin == 1, cam_bin == 1).sum())
    mask_sum = float((mask_bin == 1).sum())
    cam_sum = float((cam_bin == 1).sum())

    iou = intersection / max(union, 1.0)
    dice = (2.0 * intersection) / max(mask_sum + cam_sum, 1.0)
    return iou, dice


def task_5_2_gradcam_variants_with_mask_check(
    model_name: str,
    model: torch.nn.Module,
    test_dataset: Subset,
    class_names: Sequence[str],
) -> None:
    """Generate Grad-CAM + Grad-CAM++ from an earlier layer and optionally compare with masks."""
    print("\n" + "=" * 60)
    print("Exercise 5.2 (Full): Grad-CAM Early Layer + Grad-CAM++")
    print("=" * 60)

    model = model.to(project.device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = True

    target_layer = get_earlier_gradcam_layer(model)
    print(f"Using earlier Grad-CAM target layer: {target_layer.__class__.__name__}")

    mask_index = discover_mask_files(project.DATA_ROOT)
    mask_available = len(mask_index) > 0
    if mask_available:
        print(f"Discovered {len(mask_index)} potential mask files for overlap checks.")
    else:
        print("No mask files discovered. Overlap metrics will be skipped.")

    num_examples = 3
    fig, axes = plt.subplots(num_examples, 5, figsize=(20, 5 * num_examples))
    axes = np.atleast_2d(axes)
    fig.suptitle("Early Layer Grad-CAM vs Grad-CAM++ (Full Run)", fontsize=15)

    overlap_lines: List[str] = [
        "Grad-CAM mask overlap audit (full run)",
        f"Mask files discovered: {len(mask_index)}",
    ]
    ious: List[float] = []
    dices: List[float] = []

    with GradCAM(model=model, target_layers=[target_layer]) as cam, GradCAMPlusPlus(model=model, target_layers=[target_layer]) as campp:
        shown = 0
        for idx in range(len(test_dataset)):
            img, label = test_dataset[idx]
            input_tensor = img.unsqueeze(0).to(project.device)

            with torch.no_grad():
                logits = model(input_tensor)
                probabilities = torch.softmax(logits, dim=1)
                pred_class = int(probabilities.argmax(dim=1).item())
                pred_prob = float(probabilities[0, pred_class].item())

            targets = [ClassifierOutputTarget(pred_class)]
            gradcam_map = cam(input_tensor=input_tensor, targets=targets)[0]
            gradcampp_map = campp(input_tensor=input_tensor, targets=targets)[0]

            gradcam_map = gradcam_map / max(float(gradcam_map.max()), 1e-8)
            gradcampp_map = gradcampp_map / max(float(gradcampp_map.max()), 1e-8)
            gradcampp_bin = (gradcampp_map >= np.percentile(gradcampp_map, 80)).astype(np.uint8)

            img_for_display = project.denormalize_tensor(img).cpu().numpy()
            rgb_img = np.repeat(np.transpose(img_for_display, (1, 2, 0)), 3, axis=2)
            overlay_cam = show_cam_on_image(rgb_img.astype(np.float32), gradcam_map, use_rgb=True, image_weight=0.65)
            overlay_campp = show_cam_on_image(rgb_img.astype(np.float32), gradcampp_map, use_rgb=True, image_weight=0.65)

            sample_global_idx = int(test_dataset.indices[idx])
            sample_path = Path(test_dataset.dataset.samples[sample_global_idx][0])
            sample_key = normalize_stem_for_mask(sample_path.stem)
            mask_path = mask_index.get(sample_key)

            axes[shown, 0].imshow(img_for_display.squeeze(), cmap="gray")
            axes[shown, 0].set_title(
                f"True: {project.to_display_name(class_names[int(label)])}\n"
                f"Pred: {project.to_display_name(class_names[pred_class])} ({pred_prob:.2f})"
            )
            axes[shown, 0].axis("off")

            axes[shown, 1].imshow(gradcam_map, cmap="jet")
            axes[shown, 1].set_title("Grad-CAM (early layer)")
            axes[shown, 1].axis("off")

            axes[shown, 2].imshow(gradcampp_map, cmap="jet")
            axes[shown, 2].set_title("Grad-CAM++ (early layer)")
            axes[shown, 2].axis("off")

            axes[shown, 3].imshow(overlay_cam)
            axes[shown, 3].set_title("Overlay: Grad-CAM")
            axes[shown, 3].axis("off")

            if mask_path is not None:
                mask_bin = load_mask_binary(mask_path, shape_hw=gradcampp_map.shape)
                iou, dice = compute_overlap_metrics(mask_bin, gradcampp_bin)
                ious.append(iou)
                dices.append(dice)
                overlap_lines.append(f"{sample_path.name} | mask={mask_path.name} | IoU={iou:.4f} | Dice={dice:.4f}")

                axes[shown, 4].imshow(mask_bin, cmap="gray")
                axes[shown, 4].set_title(f"Mask (IoU={iou:.2f}, Dice={dice:.2f})")
                axes[shown, 4].axis("off")
            else:
                overlap_lines.append(f"{sample_path.name} | mask=NOT_FOUND")
                axes[shown, 4].imshow(overlay_campp)
                axes[shown, 4].set_title("Overlay: Grad-CAM++")
                axes[shown, 4].axis("off")

            shown += 1
            if shown >= num_examples:
                break

    if ious:
        overlap_lines.append(f"Mean IoU: {float(np.mean(ious)):.4f}")
        overlap_lines.append(f"Mean Dice: {float(np.mean(dices)):.4f}")
    else:
        overlap_lines.append("Mean IoU: n/a (no matched masks)")
        overlap_lines.append("Mean Dice: n/a (no matched masks)")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(project.OUTPUT_DIR / f"5_2_{model_name}_gradcam_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: outputs_full/5_2_{model_name}_gradcam_comparison.png")

    project.save_text_summary(f"5_2_{model_name}_mask_overlap_report.txt", overlap_lines)
    print(f"Saved: outputs_full/5_2_{model_name}_mask_overlap_report.txt")


def main() -> None:
    # Force full-data mode and write artifacts to a dedicated folder.
    project.USE_SMALL_SUBSET = False
    project.SMALL_SUBSET_FRACTION = 1.0
    project.IMAGE_SIZE = FULL_IMAGE_SIZE
    project.OUTPUT_DIR = project.PROJECT_ROOT / "outputs_full"
    project.build_transforms = build_transforms_full
    project.build_dataloaders = build_dataloaders_grouped

    print("\n" + "=" * 60)
    print("MPHY 6120 - Full Dataset Run")
    print("=" * 60)
    print(f"Output directory: {project.OUTPUT_DIR}")
    print("Subset mode disabled: using full dataset\n")
    print("Leakage guard enabled: grouped split by class + numeric filename ID\n")
    print(f"Increased input resolution enabled: {project.IMAGE_SIZE}x{project.IMAGE_SIZE}")
    print("Center-bias check enabled: random resized crops + random shifts during training\n")

    project.ensure_output_dir()

    # Part 1: data audit, visualization, and augmentation.
    train_loader, val_loader, test_loader, train_dataset, test_dataset, class_names = project.task_1_1_load_data()
    
    
    project.task_1_2_visualize_data(train_dataset, class_names)
    train_loader_aug = project.task_1_3_data_augmentation(train_dataset, class_names)

    # Part 2: baseline CNN.
    simple_cnn = project.SimpleCNN(num_classes=len(class_names)).to(project.device)
    print(f"\nSimpleCNN architecture:\n{simple_cnn}")
    simple_cnn, simple_history = project.task_2_2_training_loop(
        simple_cnn,
        train_loader_aug,
        val_loader,
        epochs=8, #epochs = 8 --- IGNORE ---
        learning_rate=1e-3,
    )
    project.task_2_3_training_curves(simple_history, model_name="Simple CNN")

    # Part 3: transfer learning with ResNet18.
    resnet_pretrained = project.task_3_1_pretrained_model(num_classes=len(class_names))
    resnet_pretrained, pretrained_history = project.task_3_2_fine_tuning(resnet_pretrained, train_loader_aug, val_loader)
    project.plot_training_curves(pretrained_history, "ResNet18 Pretrained", "3_2_pretrained_training_curves.png")
    print("Saved: outputs_full/3_2_pretrained_training_curves.png")

    # A scratch ResNet gives a fair transfer-learning comparison.
    resnet_scratch = project.build_resnet18_for_grayscale(num_classes=len(class_names), pretrained=False)
    resnet_scratch, scratch_history = project.train_model(
        resnet_scratch,
        train_loader_aug,
        val_loader,
        epochs=6, #epochs = 6
        learning_rate=1e-3,
    )
    project.plot_training_curves(scratch_history, "ResNet18 Scratch", "3_2_scratch_training_curves.png")
    print("Saved: outputs_full/3_2_scratch_training_curves.png")

    project.task_3_3_compare_models(simple_cnn, resnet_scratch, resnet_pretrained, test_loader, class_names)

    # Part 4: test-set evaluation.
    evaluated_model = "resnet_pretrained"  # Choose which model to evaluate on the test set.
    labels, probs, _ = project.compute_confusion_matrix(resnet_pretrained, evaluated_model, test_loader, class_names)
    project.test_roc_pr_curves(evaluated_model, labels, probs, class_names)
    project.model_calibration_plot(evaluated_model, labels, probs, class_names)
    project.error_analysis(resnet_pretrained, evaluated_model, test_dataset, class_names)

    # Part 5: interpretability.
    project.visualize_filters(evaluated_model, resnet_pretrained)
    task_5_2_gradcam_variants_with_mask_check(evaluated_model, resnet_pretrained, test_dataset, class_names)
    project.task_5_3_occlusion_sensitivity(resnet_pretrained, evaluated_model, test_dataset, class_names)

    
    
    # Test the resnet_pretrained model on the brain_tumor_2 dataset
    # Load the brain_tumor_2 dataset
    brain_tumor_2_test_dataset = Subset(
    ImageFolder(
        root=project.PROJECT_ROOT / "data/brain_tumor_2",
        transform=build_transforms_full(train=False)
    ),
    indices=list(range(len(ImageFolder(
        root=project.PROJECT_ROOT / "data/brain_tumor_2",
        transform=build_transforms_full(train=False)
    ))))
)

    # Create the DataLoader for the test dataset
    brain_tumor_2_test_loader = DataLoader(brain_tumor_2_test_dataset, batch_size=32, shuffle=False)

    # Evaluate the model on the brain_tumor_2 dataset
    labels2, probs2, _ = project.compute_confusion_matrix(resnet_pretrained, "resnet_pretrained_dataset2", brain_tumor_2_test_loader, class_names)
    project.test_roc_pr_curves("resnet_pretrained_dataset2", labels2, probs2, class_names)
    project.model_calibration_plot("resnet_pretrained_dataset2", labels2, probs2, class_names)
    project.error_analysis(resnet_pretrained, "resnet_pretrained_dataset2", brain_tumor_2_test_dataset, class_names)

    # Visualize filters and Grad-CAM for the brain_tumor_2 dataset
    project.visualize_filters("resnet_pretrained_dataset2", resnet_pretrained)
    task_5_2_gradcam_variants_with_mask_check("resnet_pretrained_dataset2", resnet_pretrained, brain_tumor_2_test_dataset, class_names)
    project.task_5_3_occlusion_sensitivity(resnet_pretrained, "resnet_pretrained_dataset2", brain_tumor_2_test_dataset, class_names)
    
    
    print("\n" + "=" * 60)
    print("FULL DATASET RUN COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
