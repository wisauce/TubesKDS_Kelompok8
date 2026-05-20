"""
Grad-CAM — Gradient-weighted Class Activation Mapping
=====================================================
Generates heatmaps showing which spatial regions of a cell image
the CNN attends to when predicting each cell cycle phase.
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.config import IMG_SIZE, IMG_MEAN, IMG_STD, DEVICE, IDX_TO_PHASE


class GradCAM:
    """
    Grad-CAM for ResNet-18.
    Hooks into the last convolutional layer (layer4) to capture
    feature maps and gradients.
    """

    def __init__(self, model, target_layer=None):
        self.model = model.eval().to(DEVICE)
        self.gradients = None
        self.activations = None

        # Default: last conv block of ResNet-18
        if target_layer is None:
            target_layer = self.model.backbone.layer4[-1].conv2

        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap for a single image tensor.

        Args:
            input_tensor: (1, C, H, W) normalized image tensor
            target_class: class index (None = use predicted class)

        Returns:
            heatmap: (H, W) numpy array in [0, 1]
            predicted_class: int
            confidence: float
        """
        input_tensor = input_tensor.to(DEVICE)
        self.model.zero_grad()

        output = self.model(input_tensor)
        probs = F.softmax(output, dim=1)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        confidence = probs[0, target_class].item()

        # Backprop the target class score
        target_score = output[0, target_class]
        target_score.backward()

        # Pool gradients over spatial dimensions → channel weights
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)

        # Weighted sum of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)  # ReLU to keep only positive influence

        # Upsample to input resolution
        cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE),
                            mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam, target_class, confidence


def generate_gradcam_batch(model, dataset, n_samples=5, seed=42):
    """
    Generate Grad-CAM visualizations for a sample of images.

    Returns:
        list of dicts: [{
            "original": np.ndarray (H,W,3) uint8,
            "heatmap": np.ndarray (H,W) float,
            "predicted": str,
            "true_label": str,
            "confidence": float,
        }, ...]
    """
    np.random.seed(seed)
    gradcam = GradCAM(model)
    results = []

    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMG_MEAN, IMG_STD),
    ])

    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)

    for idx in indices:
        path, label = dataset.dataset.samples[dataset.indices[idx]]
        pil_img = Image.open(path).convert("RGB")
        original = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE)))

        tensor = eval_tf(pil_img).unsqueeze(0)
        heatmap, pred_cls, conf = gradcam.generate(tensor)

        results.append({
            "original": original,
            "heatmap": heatmap,
            "predicted": IDX_TO_PHASE[pred_cls],
            "true_label": IDX_TO_PHASE[label],
            "confidence": conf,
        })

    return results
