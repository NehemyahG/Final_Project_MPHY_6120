# Final_Project_MPHY_6120

Brain tumor MRI 3-class classification project for MPHY 6120.

## Dataset

- Source: https://www.kaggle.com/datasets/orvile/brain-cancer-mri-dataset
- Expected path in this repo:
	- `Brain_Cancer raw MRI data/Brain_Cancer/brain_glioma`
	- `Brain_Cancer raw MRI data/Brain_Cancer/brain_menin`
	- `Brain_Cancer raw MRI data/Brain_Cancer/brain_tumor`

## Environment

Create/activate a Python virtual environment, then install packages:

```powershell
pip install numpy matplotlib scikit-learn torch torchvision grad-cam
```

## Run

From the repo root:

```powershell
python brain_tumor_prediction.py
```

For a full-dataset run with separate artifacts:

```powershell
python brain_tumor_prediction_full.py
```

The script performs:
1. Data audit + visualization
2. Baseline CNN training
3. ResNet18 transfer learning
4. Test evaluation and curves
5. Interpretability (filters + Grad-CAM)

Generated artifacts are written to:
- `outputs/` for subset/debug run (`brain_tumor_prediction.py`)
- `outputs_full/` for full-data run (`brain_tumor_prediction_full.py`)

## Latest Full Run (Apr 24, 2026)

The pipeline was executed successfully end-to-end on CPU.

### Data Split

- Train: 1059
- Validation: 227
- Test: 228
- Classes: Glioma, Meningioma, Pituitary Tumor

### Model Comparison (Test Set)

- Simple CNN
	- Accuracy: 0.7719
	- Macro Precision: 0.8381
	- Macro Recall: 0.7734
	- Macro F1: 0.7759
	- Macro AUC (OvR): 0.9520
- ResNet18 Scratch
	- Accuracy: 0.5439
	- Macro Precision: 0.7680
	- Macro Recall: 0.5426
	- Macro F1: 0.5023
	- Macro AUC (OvR): 0.9237
- ResNet18 Pretrained (Best)
	- Accuracy: 0.9693
	- Macro Precision: 0.9706
	- Macro Recall: 0.9694
	- Macro F1: 0.9695
	- Macro AUC (OvR): 0.9982

### Best Model Classification Report (ResNet18 Pretrained)

- Glioma: Precision 1.00, Recall 0.96, F1 0.98
- Meningioma: Precision 0.93, Recall 0.99, F1 0.95
- Pituitary Tumor: Precision 0.99, Recall 0.96, F1 0.97
- Overall test accuracy: 0.9693

### Generated Outputs

- `outputs/1_1_data_audit.txt`
- `outputs/1_2_sample_grid.png`
- `outputs/1_2_class_distribution.png`
- `outputs/1_3_augmentation.png`
- `outputs/1_3_augmentation_note.txt`
- `outputs/2_3_training_curves.png`
- `outputs/2_3_training_curve_note.txt`
- `outputs/3_1_transfer_learning_note.txt`
- `outputs/3_2_pretrained_training_curves.png`
- `outputs/3_2_scratch_training_curves.png`
- `outputs/3_3_model_comparison_note.txt`
- `outputs/4_1_confusion_matrix.png`
- `outputs/4_1_evaluation_note.txt`
- `outputs/4_2_roc_pr_curves.png`
- `outputs/4_3_error_analysis.png`
- `outputs/5_1_filters.png`
- `outputs/5_2_gradcam.png`

### Notes

- Training on CPU is reproducible but slow.
- Transfer learning with pretrained weights clearly outperformed both the baseline CNN and scratch ResNet.