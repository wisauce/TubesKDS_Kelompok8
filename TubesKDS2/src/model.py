"""
CNN Model — Layer 1 of the Cell Cycle Intelligence System.
Fine-tuned ResNet-18 for cell cycle phase classification.
"""

import os
import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm

from src.config import NUM_PHASES, DEVICE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, MODEL_DIR


class CellCycleCNN(nn.Module):
    """ResNet-18 backbone with a custom classification head for 7 cell cycle phases."""

    def __init__(self, num_classes: int = NUM_PHASES, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)

        # Freeze early layers for faster convergence
        for name, param in self.backbone.named_parameters():
            if "layer3" not in name and "layer4" not in name and "fc" not in name:
                param.requires_grad = False

        # Replace final FC with a richer head
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)

    def get_feature_extractor(self):
        """Return all layers except the final FC (for Grad-CAM)."""
        return nn.Sequential(*list(self.backbone.children())[:-2])


def train_model(model, train_loader, val_loader,
                num_epochs=NUM_EPOCHS, lr=LEARNING_RATE, device=DEVICE):
    """
    Full training loop with validation, early stopping, and checkpoint saving.
    Returns (model, history_dict).
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0

    print(f"\n{'='*60}")
    print(f"  Training CNN on {device}")
    print(f"  Epochs: {num_epochs}  |  LR: {lr}  |  Batch: {train_loader.batch_size}")
    print(f"{'='*60}\n")

    for epoch in range(1, num_epochs + 1):
        # ── Train ──
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"  Epoch {epoch:02d}/{num_epochs}", ncols=80, leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / total
        train_acc = correct / total

        # ── Validate ──
        val_loss, val_acc = _evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        scheduler.step()

        print(f"  Epoch {epoch:02d}  "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.3f}  │  "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.3f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(MODEL_DIR, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_model.pth"))

    print(f"\n  ✓ Best validation accuracy: {best_val_acc:.4f}")
    print(f"  ✓ Model saved to {MODEL_DIR}/best_model.pth\n")

    # Load best weights
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_model.pth"),
                                     map_location=device, weights_only=True))
    return model, history


def _evaluate(model, loader, criterion, device):
    """Evaluate model on a DataLoader. Returns (loss, accuracy)."""
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)
    return running_loss / total, correct / total


def evaluate_test(model, test_loader, device=DEVICE):
    """
    Full test evaluation returning predictions & ground truth.
    Returns (all_preds, all_labels, all_probs).
    """
    model = model.to(device)
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


# numpy import at module level for evaluate_test
import numpy as np
