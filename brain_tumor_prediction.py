"""
FINAL PROJECT - MODEL TO PREDICT BRAIN TUMOR PROBABILITY

Run with: uv run python brain_tumor_prediction.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score, ConfusionMatrixDisplay,
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
)
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Plot settings
plt.rcParams['figure.figsize'] = (10, 6)


# =============================================================================
# Part 1: Data Loading & Exploration (15 points)
# =============================================================================

def task_1_1_load_data():
    """
    1.1 Load the Dataset 
    
    Load BrainTumor Datasets from Kaggel: https://www.kaggle.com/datasets/orvile/brain-cancer-mri-dataset

    Returns:
        train_loader, val_loader, test_loader, train_dataset, test_dataset
    """
    print("=" * 60)
    print("Task 1.1: Load the Dataset")
    print("=" * 60)

    # YOUR CODE HERE
    # Define transforms (basic for now, augmentation in 1.3)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
    ])

    # Load datasets
    dataset = None
    train_dataset = None
    val_dataset = None
    test_dataset = None

    # Create DataLoaders
    batch_size = 64
    train_loader = # DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = # DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = # DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Image shape: {train_dataset[0][0].shape}")
    print(f"Number of classes: {len(train_dataset.info['label'])}")

    return train_loader, val_loader, test_loader, train_dataset, test_dataset


def task_1_2_visualize_data(train_dataset):
    """
    1.2 Visualize the Data

    Display sample images and class distribution.
    """
    print("\n" + "=" * 60)
    print("Exercise 1.2: Visualize the Data")
    print("=" * 60)

    # YOUR CODE HERE
    # Display a grid of sample images (e.g., 4x4 grid)
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    for i, ax in enumerate(axes.flat):
        ax.imshow(train_dataset[i][0].permute(1, 2, 0).cpu().numpy(), cmap='gray')
        ax.set_title(f"Class: {train_dataset[i][1]}")
        ax.axis('off')

    # Show plot for samples from each class
    # Class: Brain_Glioma, Brain_Menin, Brain Tumor

    # Class distribution bar plot in percentage
    

    # Plot class distribution


def task_1_3_data_augmentation():
    """
    1.3 Data Augmentation (5 points)

    Implement augmentation transforms for training.

    Returns:
        augmented train_loader
    """
    print("\n" + "=" * 60)
    print("Exercise 1.3: Data Augmentation")
    print("=" * 60)

    # YOUR CODE HERE
    # Define augmentation transforms
        # Add augmentations:
        # - RandomHorizontalFlip
        # - RandomRotation(10)
        # - RandomAffine or ColorJitter for brightness/contrast
        # - etc.
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Recreate train dataset with augmentation
    train_dataset_aug = None
    

    # Define individual transformations to visualize their effects on the same image
    index = 0  
    img, label = train_dataset[index]

 # Define transforms for demonstration
    # p=1.0 so the flip definitely happens
    flip_tf = transforms.RandomHorizontalFlip(p=1.0)
    rotate_tf = transforms.RandomRotation(10)
    jitter_tf = transforms.ColorJitter(brightness=0.2, contrast=0.2)
    norm_tf = transforms.Normalize(mean=[0.5], std=[0.5])

    # Apply to the SAME tensor image
    img_original = img.clone()
    img_flip = flip_tf(img.clone())
    img_rotate = rotate_tf(img.clone())
    img_jitter = jitter_tf(img.clone())
    img_normalized = norm_tf(img.clone())

    # De-normalize only for display
    img_normalized_show = img_normalized * 0.5 + 0.5
    img_normalized_show = torch.clamp(img_normalized_show, 0, 1)

    images = [
        img_original,
        img_flip,
        img_rotate,
        img_jitter,
        img_normalized_show
    ]

    titles = [
        "Original",
        "RandomHorizontalFlip",
        "RandomRotation",
        "ColorJitter",
        "Normalize"
    ]

    plt.figure(figsize=(15, 3))
    for i, (im, title) in enumerate(zip(images, titles), 1):
        plt.subplot(1, len(images), i)
        plt.imshow(im.squeeze().cpu().numpy(), cmap="gray")
        plt.title(title)
        plt.axis("off")
    
    plt.suptitle('Data Augmentation Examples')
    plt.tight_layout()
    plt.savefig('outputs/1_3_augmentation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: outputs/1_3_augmentation.png")

    print("\n[YOUR ANSWER HERE]")
    print("Q: Why is augmentation important for medical imaging?")
    print("A: Data augmentation is crucial for medical imaging as it helps to increase the diversity of the training dataset,"
          "allowing the model to generalize better and become more robust to variations in the data."
          "This is particularly important in medical imaging, where acquiring large labeled datasets can be challenging.")

    train_loader_aug = DataLoader(train_dataset_aug, batch_size=32, shuffle=True)
    
    return train_loader_aug


# =============================================================================
# Part 2: Build a Simple CNN (25 points)
# =============================================================================

class SimpleCNN(nn.Module):
    """
    2.1 Define the Architecture (10 points)

    A simple CNN for binary classification.
    """
    def __init__(self):
        super(SimpleCNN, self).__init__() # Feel free to create a deeper model

        # YOUR CODE HERE
        # Define layers:
        # - Conv layers (e.g., 1->16->32 channels)
        # - MaxPool layers
        # - Fully connected layers
        # - Output layer (1 neuron for binary classification)

        self.features = nn.Sequential(  # nn.Sequential(...)
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # YOUR CODE HERE
        x = self.features(x)
        x = self.classifier(x)
        return x


def task_2_2_training_loop(model, train_loader, val_loader, epochs):
    

    """
    2.2 Training Loop 

    Train the model with early stopping.

    Returns:
        model: Trained model
        history: Dict with training history
    """
    print("\n" + "=" * 60)
    print("Exercise 2.2: Training Loop")
    print("=" * 60)

    model = model.to(device)

    # YOUR CODE HERE
    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()   # BCEWithLogitsLoss or CrossEntropyLoss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = None  # Try: optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }

    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            # YOUR CODE HERE
            # Forward pass, loss, backward pass, optimizer step
            images = images.to(device)
            labels = labels.squeeze().float().to(device)
            
            if labels.dim() == 1:
                    labels = labels.unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            
            probs = torch.sigmoid(outputs)
            predicted = (probs >= 0.5).float()
            
            correct += (predicted == labels).sum().item()
            total += labels.numel()

        if scheduler is not None:
            scheduler.step()



        # Validation phase
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.float().view(-1, 1).to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss_sum += loss.item() * images.size(0)

                probs = torch.sigmoid(outputs)
                predicted = (probs >= 0.5).float()

                val_correct += (predicted == labels).sum().item()
                val_total += labels.numel()
                

        # Calculate epoch metrics
        # YOUR CODE HERE
        train_loss = running_loss / total
        train_acc = correct / total
        
        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Early stopping check
        # YOUR CODE HERE
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered")
            break
    
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")

    return model, history

 
def task_2_3_training_curves(history):
    """
    2.3 Training Curves (5 points)

    Plot loss and accuracy curves.
    """
    print("\n" + "=" * 60)
    print("Exercise 2.3: Training Curves")
    print("=" * 60)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))


    # Loss curves
    ax1 = axes[0]
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid()


    # Accuracy curves
    ax2 = axes[1]
    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid()

    plt.tight_layout()
    plt.savefig('outputs/2_3_training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: outputs/2_3_training_curves.png")

    print("\n[YOUR ANALYSIS HERE]")
    print("Q: Is your model overfitting? How can you tell?")
    print("A: The model is slightly overfitting for the higher ephocs number."
          "The curve shows that the validation loss reach plateu after 6 epochs, while the training loss continues to decrease. "
          "The same pattern is observed in the accuracy curve, where the validation accuracy reach plateu after 6 epochs,"
          "while the training accuracy continues to increase."
          "It suggest extra training mostly helps training performance, not generalization")


# =============================================================================
# Part 3: Transfer Learning (25 points)
# =============================================================================

def task_3_1_pretrained_model():
    """
    3.1 Load Pretrained Model (8 points)
    

    Adapt a pretrained ResNet18 for our task.

    Returns:
        model: Modified ResNet18
    """
    print("\n" + "=" * 60)
    print("Exercise 3.1: Load Pretrained Model")
    print("=" * 60)

    # YOUR CODE HERE
    # Load pretrained ResNet18
    model = models.resnet18(weights='IMAGENET1K_V1')

    # Modify first conv layer for 1-channel input
    # Original: Conv2d(3, 64, kernel_size=7, ...)
    # We need: Conv2d(1, 64, kernel_size=7, ...)
    # Hint: Average the pretrained weights across input channels
    pretrained_weights = model.conv1.weight.data
    new_weights = pretrained_weights.mean(dim=1, keepdim=True)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.conv1.weight.data = new_weights

    # Replace final fully connected layer
    # Original: fc = Linear(512, 1000)
    # We need: fc = Linear(512, 1) for binary classification
    model.fc = nn.Linear(512, 1)

    print("Modified ResNet18 architecture:")
    print(f"First conv: {model.conv1}")
    print(f"Final fc: {model.fc}")

    print("\n[YOUR ANSWER HERE]")
    print("Q: Why might ImageNet pretrained weights help for medical images?")
    print("A: Pretrained weights can help the model learn relevant features from medical images more effectively, as they have already captured general patterns from a large and diverse dataset.")

    return model


def task_3_2_fine_tuning(model, train_loader, val_loader):
    """
    3.2 Fine-Tuning Strategy (8 points)

    Implement two-stage fine-tuning.

    Returns:
        model: Fine-tuned model
        history: Training history
    """
    print("\n" + "=" * 60)
    print("Exercise 3.2: Fine-Tuning Strategy")
    print("=" * 60)

    model = model.to(device)

    # Stage 1: Freeze all layers except final classifier
    print("\nStage 1: Training classifier only...")
    # YOUR CODE HERE
    # Freeze parameters
    # Train for ~5 epochs with higher learning rate
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
        
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
    scheduler = None  # Try: optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    history_stage1 = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    for epoch in range(5): # epochs = 5
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.squeeze().float().to(device)
            
            if labels.dim() == 1:
                    labels = labels.unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            
            probs = torch.sigmoid(outputs)
            predicted = (probs >= 0.5).float()
            
            correct += (predicted == labels).sum().item()
            total += labels.numel()

        if scheduler is not None:
            scheduler.step()

        # Validation phase
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.float().view(-1, 1).to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss_sum += loss.item() * images.size(0)

                probs = torch.sigmoid(outputs)
                predicted = (probs >= 0.5).float()

                val_correct += (predicted == labels).sum().item()
                val_total += labels.numel()
                

        # Calculate epoch metrics
        train_loss = running_loss / total
        train_acc = correct / total
        
        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total

        history_stage1["train_loss"].append(train_loss)
        history_stage1["val_loss"].append(val_loss)
        history_stage1["train_acc"].append(train_acc)
        history_stage1["val_acc"].append(val_acc)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered")
            break
    
        print(f"Epoch {epoch+1}/5 - "
              f"Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}")

    # Stage 2: Unfreeze all layers, train with lower learning rate
    print("\nStage 2: Fine-tuning full model...")
    # YOUR CODE HERE
    # Unfreeze parameters
    # Train for ~10 epochs with lower learning rate
    
    for param in model.parameters():
        param.requires_grad = True
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = None  # Try: optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    history_stage2 = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }

    best_val_loss = float('inf')
    patience_counter = 0
    for epoch in range(10): # epochs = 10
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.squeeze().float().to(device)
            
            if labels.dim() == 1:
                    labels = labels.unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            
            probs = torch.sigmoid(outputs)
            predicted = (probs >= 0.5).float()
            
            correct += (predicted == labels).sum().item()
            total += labels.numel()

        if scheduler is not None:
            scheduler.step()

        # Validation phase
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.float().view(-1, 1).to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss_sum += loss.item() * images.size(0)

                probs = torch.sigmoid(outputs)
                predicted = (probs >= 0.5).float()

                val_correct += (predicted == labels).sum().item()
                val_total += labels.numel()
                

        # Calculate epoch metrics
        train_loss = running_loss / total
        train_acc = correct / total
        
        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total

        history_stage2["train_loss"].append(train_loss)
        history_stage2["val_loss"].append(val_loss)
        history_stage2["train_acc"].append(train_acc)
        history_stage2["val_acc"].append(val_acc)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered")
            break
    
        print(f"Epoch {epoch+1}/10 - "
              f"Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}")
            
    # Combine histories from both stages
    history = {
        'train_loss': history_stage1['train_loss'] + history_stage2['train_loss'],
        'val_loss': history_stage1['val_loss'] + history_stage2['val_loss'],
        'train_acc': history_stage1['train_acc'] + history_stage2['train_acc'],
        'val_acc': history_stage1['val_acc'] + history_stage2['val_acc']
    }  

    return model, history


def task_3_3_compare_models(simple_cnn, resnet_scratch, resnet_pretrained, test_loader):
    """
    3.3 Compare Models (9 points)

    Create comparison table of model performance.
    """
    print("\n" + "=" * 60)
    print("Exercise 3.3: Compare Models")
    print("=" * 60)

    # YOUR CODE HERE
    # Evaluate each model on test set
    # Calculate: Accuracy, AUC, Precision, Recall, F1
    def evaluate_model(model, data_loader):
        """
        Evaluate the model on the given data loader and return true and predicted labels.

        Args:
            model: Trained model.
            data_loader: DataLoader for evaluation.

        Returns:
            y_true: True labels.
            y_pred: Predicted labels.
        """
        model.eval()
        y_true = []
        y_pred = []

        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                probs = torch.sigmoid(outputs)
                predictions = (probs >= 0.5).float()

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predictions.cpu().numpy())

        return np.array(y_true).flatten(), np.array(y_pred).flatten()

    metrics = {}
    metrics = {}
    for model_name, model in zip(["Simple CNN", "ResNet Scratch", "ResNet Pretrained"],
                                  [simple_cnn, resnet_scratch, resnet_pretrained]):
        y_true, y_pred = evaluate_model(model, test_loader)
        metrics[model_name] = {
            'accuracy': accuracy_score(y_true, y_pred),
            'auc': roc_auc_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred)
        }

    print("\nModel Comparison:")
    print("-" * 70)
    print(f"{'Model':<25} {'Accuracy':<10} {'AUC':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print("-" * 70)
    for model_name, model_metrics in metrics.items():
        print(f"{model_name:<25} {model_metrics['accuracy']:<10.4f} {model_metrics['auc']:<10.4f} "
              f"{model_metrics['precision']:<10.4f} {model_metrics['recall']:<10.4f} {model_metrics['f1']:<10.4f}")

    print("\n[YOUR ANALYSIS HERE]")
    print("Q: Which model performs best? Why?")
    print("A: The evaluation matrices suggest that the Simple CNN performs best, with the highest accuracy, AUC, precision, and F1 score."
          "This may be due to the fact that the Simple CNN is less complex and may be better suited for the relatively small dataset,"
          "while the ResNet models may be overfitting or not fully leveraging their capacity given the limited data.")


# =============================================================================
# Part 4: Model Evaluation (20 points)
# =============================================================================

def task_4_1_confusion_matrix(model, test_loader):
    """
    4.1 Confusion Matrix & Metrics (8 points)

    Evaluate best model with detailed metrics.
    """
    print("\n" + "=" * 60)
    print("Exercise 4.1: Confusion Matrix & Metrics")
    print("=" * 60)

    model.eval()

    # YOUR CODE HERE
    # Get predictions on test set
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.sigmoid(outputs)
            predictions = (probs >= 0.5).float()

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()
    all_probs = np.array(all_probs).flatten()
    

    # Confusion matrix
    plt.figure(figsize=(6, 5))
    # YOUR CODE HERE - ConfusionMatrixDisplay
    cm=confusion_matrix(all_labels, all_preds)
    im = plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.xticks(ticks=[0, 1], labels=['Normal', 'Pneumonia'])
    plt.yticks(ticks=[0, 1], labels=['Normal', 'Pneumonia'])
    plt.grid(False)
    plt.savefig('outputs/4_1_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: outputs/4_1_confusion_matrix.png")

    # Classification report
    print("\nClassification Report:")
    # YOUR CODE HERE
    report = classification_report(all_labels, all_preds, target_names=['Normal', 'Pneumonia'],
                                   output_dict=True, zero_division=0)
    
    for label in ['Normal', 'Pneumonia']:
        print(f"{label}: Precision: {report[label]['precision']:.4f}, "
              f"Recall: {report[label]['recall']:.4f}, "
              f"F1-score: {report[label]['f1-score']:.4f}")
    

    print("\n[YOUR ANALYSIS HERE]")
    print("Q: Given the clinical context (pneumonia detection), which metric matters most?")
    print("A: Recall (sensitivity) is the most critical metric in this context, as it measures the model's ability to correctly identify pneumonia cases. "
          "Missing a pneumonia case (false negative) can have severe consequences for patient care, so maximizing recall is essential to ensure that as many true pneumonia cases as possible are detected, even if it means accepting a higher false positive rate.")

    return all_labels, all_probs


def task_4_2_roc_pr_curves(labels, probs):
    """
    4.2 ROC and PR Curves (6 points)

    Plot ROC and Precision-Recall curves.
    """
    print("\n" + "=" * 60)
    print("Exercise 4.2: ROC and PR Curves")
    print("=" * 60)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ROC Curve
    ax1 = axes[0]
    # YOUR CODE HERE
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    ax1.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax1.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend(loc='lower right')

    # Precision-Recall Curve
    ax2 = axes[1]
    # YOUR CODE HERE
    precision, recall, _ = precision_recall_curve(labels, probs)
    pr_auc = auc(recall, precision)
    ax2.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.4f})')
    ax2.plot([0, 1], [0, 0], 'k--', label='Random Guess')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend(loc='lower left')

    plt.tight_layout()
    plt.savefig('outputs/4_2_roc_pr_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: outputs/4_2_roc_pr_curves.png")


def task_4_3_error_analysis(model, test_dataset):
    """
    4.3 Error Analysis (6 points)

    Display examples of TP, TN, FP, FN.
    """
    print("\n" + "=" * 60)
    print("Exercise 4.3: Error Analysis")
    print("=" * 60)

    model.eval()

    # YOUR CODE HERE
    # Find examples of each category
    # TP: label=1, pred=1 (correctly identified pneumonia)
    # TN: label=0, pred=0 (correctly identified normal)
    # FP: label=0, pred=1 (normal misclassified as pneumonia)
    # FN: label=1, pred=0 (pneumonia misclassified as normal)
    tp_images, tn_images, fp_images, fn_images = [], [], [], []
    tp_labels, tn_labels, fp_labels, fn_labels = [], [], [], []
    tp_preds, tn_preds, fp_preds, fn_preds = [], [], [], []
    tp_probs, tn_probs, fp_probs, fn_probs = [], [], [], []

    
    # Use DataLoader for efficient iteration
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            prob = torch.sigmoid(outputs).item()
            pred = 1 if prob >= 0.5 else 0
            true = labels.item()

            # remove batch dim only, keep channel dim
            img_cpu = images.cpu().squeeze(0)

            if true == 1 and pred == 1 and len(tp_images) < 4:
                tp_images.append(img_cpu)
                tp_labels.append(true)
                tp_preds.append(pred)
                tp_probs.append(prob)

            elif true == 0 and pred == 0 and len(tn_images) < 4:
                tn_images.append(img_cpu)
                tn_labels.append(true)
                tn_preds.append(pred)
                tn_probs.append(prob)

            elif true == 0 and pred == 1 and len(fp_images) < 4:
                fp_images.append(img_cpu)
                fp_labels.append(true)
                fp_preds.append(pred)
                fp_probs.append(prob)

            elif true == 1 and pred == 0 and len(fn_images) < 4:
                fn_images.append(img_cpu)
                fn_labels.append(true)
                fn_preds.append(pred)
                fn_probs.append(prob)

            if (len(tp_images) == 4 and len(tn_images) == 4 and
                len(fp_images) == 4 and len(fn_images) == 4):
                break

                    
    # Visualize examples
    def plot_img(ax, img, label, pred, prob):
        img_np = img.numpy()

        if img_np.ndim == 3 and img_np.shape[0] == 1:
            ax.imshow(img_np[0], cmap='gray')
        elif img_np.ndim == 3 and img_np.shape[0] == 3:
            ax.imshow(np.transpose(img_np, (1, 2, 0)))
        elif img_np.ndim == 2:
            ax.imshow(img_np, cmap='gray')
        else:
            ax.text(0.5, 0.5, "Bad shape\n{}".format(img_np.shape),
                    ha='center', va='center')
        ax.set_title("T:{} P:{}\nProb:{:.2f}".format(label, pred, prob), fontsize=9)
        ax.axis('off')

    fig, axes = plt.subplots(2, 8, figsize=(20, 6))
    fig.suptitle("Error Analysis: TP, TN, FP, FN Examples")

    # Row 1: TP, TN examples
    for i in range(3):
        if i < len(tp_images):
            plot_img(axes[0, i], tp_images[i], tp_labels[i], tp_preds[i], tp_probs[i])
        else:
            axes[0, i].axis('off')

    for i in range(3):
        if i < len(tn_images):
            plot_img(axes[0, i + 4], tn_images[i], tn_labels[i], tn_preds[i], tn_probs[i])
        else:
            axes[0, i + 4].axis('off')

    # Row 2: FP, FN examples
    for i in range(3):
        if i < len(fp_images):
            plot_img(axes[1, i], fp_images[i], fp_labels[i], fp_preds[i], fp_probs[i])
        else:
            axes[1, i].axis('off')

    for i in range(3):
        if i < len(fn_images):
            plot_img(axes[1, i + 4], fn_images[i], fn_labels[i], fn_preds[i], fn_probs[i])
        else:
            axes[1, i + 4].axis('off')
    
    
    plt.tight_layout()
    plt.savefig('outputs/4_3_error_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: outputs/4_3_error_analysis.png")

    print("\n[YOUR ANALYSIS HERE]")
    print("Q: Can you identify patterns in the errors?")
    print("A: FP occurs in normal images that look busy, hazy, low-contrast, or underexpanded"
          "while FN happen in pneumonia cases with mild or less obvious opacities."
          "The false positives are also predicted with high confidence in several cases, "
          "while the false negatives are mostly lower-confidence misses. "
          "That suggests the model is sometimes confidently fooled by certain visual patterns, "
          "but misses pneumonia when the pattern is weak or ambiguous.")

# =============================================================================
# Part 5: Model Interpretability (15 points)
# =============================================================================

def task_5_1_visualize_filters(model):
    """
    5.1 Visualize Filters (5 points)

    Display learned filters from first conv layer.
    """
    print("\n" + "=" * 60)
    print("Exercise 5.1: Visualize Filters")
    print("=" * 60)

    # YOUR CODE HERE
    # Get weights from first conv layer
    # For SimpleCNN: model.features[0].weight
    # For ResNet: model.conv1.weight
    
    if isinstance(model, SimpleCNN):
        filters = model.features[0].weight.data.cpu().numpy()
    elif isinstance(model, models.ResNet):
        filters = model.conv1.weight.data.cpu().numpy()
    else:
        print("Unsupported model type for filter visualization.")
        return
    
    # Display filters in a grid
    num_filters = filters.shape[0]
    num_cols = 8
    num_rows = (num_filters + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols, num_rows))
    for i in range(num_filters):
        ax = axes[i // num_cols, i % num_cols]
        filter_img = filters[i, 0]  # Get the single channel
        ax.imshow(filter_img, cmap='gray')
        ax.axis('off')
    plt.suptitle('Learned Filters from First Conv Layer')
    plt.tight_layout()  
    plt.savefig('outputs/5_1_filters.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: outputs/5_1_filters.png")

    print("\n[YOUR ANALYSIS HERE]")
    print("Q: What patterns do these filters detect?")
    print("A: It appears to learn basic edge, contrast, and texture detectors. "
          "These low-level filters likely help the network identify local boundaries "
          "and density changes in the chest radiographs, which are then combined by deeper "
          "layers into more task-relevant features.")



class BinaryLogitTarget:
    def __init__(self, target_class):
        self.target_class = target_class  # 0 or 1

    def __call__(self, model_output):
        # model_output shape: [B, 1]
        logit = model_output[:, 0]
        if self.target_class == 1:
            return logit
        else:
            return -logit

def task_5_2_gradcam(model, test_dataset):
    """
    5.2 Grad-CAM Visualization (10 points)

    Generate and visualize Grad-CAM heatmaps.
    """
    print("\n" + "=" * 60)
    print("Exercise 5.2: Grad-CAM Visualization")
    print("=" * 60)

    # YOUR CODE HERE
    # Option 1: Implement Grad-CAM from scratch
    # Option 2: Use pytorch-grad-cam library
    
    torch.set_grad_enabled(True)
    
    model = model.to(device)
    model.eval()
    
    # Use pytorch-grad-cam library
    
    # Make sure gradients are enabled for CAM
    for param in model.parameters():
        param.requires_grad = True
        
    # Recommended target layer
    target_layers = [model.layer4[-1]]

    # Use a few test examples only
    num_examples = 3
    fig, axes = plt.subplots(3, num_examples, figsize=(12, 10))
    fig.suptitle("Grad-CAM Visualizations", fontsize=16)

    with GradCAM(model=model, target_layers=target_layers) as cam:
        shown = 0

        for idx in range(len(test_dataset)):
            img, label = test_dataset[idx]
        
            
            # 1. Get the image and true label
            input_tensor = img.unsqueeze(0).to(device)   # [1, 1, H, W]
            #input_tensor.requires_grad_(True)
            true_label = int(label.item())
            
            # Do NOT use torch.no_grad() here
            

            # 2. Forward pass to get prediction
            output = model(input_tensor)
            prob = torch.sigmoid(output).item()
            pred_label = 1 if prob >= 0.5 else 0

            # 3. Compute Grad-CAM for the predicted class
            targets = [BinaryClassifierOutputTarget(0)]

            #model.zero_grad()
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            grayscale_cam = grayscale_cam[0]
            
            cam_max = grayscale_cam.max()
            if cam_max > 0:
                grayscale_cam = grayscale_cam / cam_max

            # 4. Overlay heatmap on original image
            # de-normalize image from [-1, 1] back to [0, 1]
            img_for_display = img.cpu().numpy()
            img_for_display = (img_for_display * 0.5) + 0.5
            img_for_display = np.clip(img_for_display, 0, 1)
            
            #img_for_display = img.cpu().numpy()   # [1, H, W]
            #img_for_display = (img_for_display * 0.5) + 0.5
            #img_for_display = np.clip(img_for_display, 0, 1)

            # Convert grayscale image to RGB for overlay
            rgb_img = np.transpose(img_for_display, (1, 2, 0))   # [H, W, 1]
            rgb_img = np.repeat(rgb_img, 3, axis=2)              # [H, W, 3]

            cam_overlay = show_cam_on_image(
                rgb_img.astype(np.float32),
                grayscale_cam,
                use_rgb=True,
                image_weight=0.7
            )

            # Top row: original image
            #axes[0, shown].imshow(img_for_display.squeeze(), cmap='gray', vmin=0, vmax=1)
            #axes[0, shown].set_title("True: {}".format(true_label))
            #axes[0, shown].axis('off')
            
            axes[0, shown].imshow(img_for_display.squeeze(), cmap='gray')
            axes[0, shown].set_title("True Label: {}".format(true_label))
            axes[0, shown].axis('off')


            # Mid row: Grad-CAM only
            axes[1, shown].imshow(grayscale_cam, cmap='jet')
            axes[1, shown].set_title("Grad-CAM Heatmap")
            axes[1, shown].axis('off')
            
            # Bottom row: original + Grad-CAM overlay
            axes[2, shown].imshow(cam_overlay)
            axes[2, shown].set_title(f"Pred: {pred_label} (Prob: {prob:.2f})")
            axes[2, shown].axis('off')

            shown += 1
            if shown >= num_examples:
                break
        
        # Plot original and CAM images
        
    # For 4-6 test images:
    # 1. Get the image and true label
    # 2. Forward pass to get prediction
    # 3. Compute Grad-CAM for the predicted class
    # 4. Overlay heatmap on original image

    # Show original images in top row


    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('outputs/5_2_gradcam.png', dpi=150, bbox_inches='tight')
    print("Saved: outputs/5_2_gradcam.png")
    plt.show()
    
    plt.close()
    print("\n[YOUR ANALYSIS HERE]")
    print("Q: Is the model looking at clinically relevant regions?")
    print("A: ...")



# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MPHY 6120 - Homework 5: Deep Learning for Medical Imaging")
    print("=" * 60 + "\n")

    # Create outputs directory
    os.makedirs('outputs', exist_ok=True)

    # =================
    # Part 1: Data Loading
    # =================
    train_loader, val_loader, test_loader, train_dataset, test_dataset = exercise_1_1_load_data()

    if train_dataset is not None:
        task_1_2_visualize_data(train_dataset)
        train_loader_aug = task_1_3_data_augmentation()

        # =================
        # Part 2: Simple CNN
        # =================
        simple_cnn = SimpleCNN()
        print(f"\nSimpleCNN architecture:\n{simple_cnn}")

        if train_loader is not None:
            simple_cnn, history = task_2_2_training_loop(
                simple_cnn, train_loader, val_loader, epochs=20 #epochs = 20
            )
            task_2_3_training_curves(history)

            # =================
            # Part 3: Transfer Learning
            # =================
            resnet_pretrained = task_3_1_pretrained_model()
            resnet_pretrained, pretrained_history = exercise_3_2_fine_tuning(
                resnet_pretrained, train_loader, val_loader
            )

            # Train ResNet from scratch for comparison
            resnet_scratch = models.resnet18(weights=None)
            # Modify for grayscale input and binary output...
            resnet_scratch.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            resnet_scratch.fc = nn.Linear(512, 1)
            resnet_scratch, scratch_history = task_2_2_training_loop(
                resnet_scratch, train_loader, val_loader, epochs=10 #epochs = 10
            )
            

            task_3_3_compare_models(
                simple_cnn, resnet_scratch, resnet_pretrained, test_loader
            )

            # =================
            # Part 4: Evaluation
            # =================
            labels, probs = task_4_1_confusion_matrix(resnet_pretrained, test_loader)
            task_4_2_roc_pr_curves(labels, probs)
            etask_4_3_error_analysis(resnet_pretrained, test_dataset)

            # =================
            # Part 5: Interpretability
            # =================
            task_5_1_visualize_filters(simple_cnn)
            task_5_2_gradcam(resnet_pretrained, test_dataset)

  

    print("\n" + "=" * 60)
    print("FINAL PROJECT COMPELET!")
    print("=" * 60)
