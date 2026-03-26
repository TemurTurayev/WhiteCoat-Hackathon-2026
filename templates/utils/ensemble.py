"""
Ensemble, TTA и GradCAM утилиты.
Ensemble: average predictions from multiple models -> +1-3% accuracy
TTA: predict on augmented versions -> +1-2% accuracy
GradCAM: visualize model attention -> interpretability for judges
"""
import cv2
import numpy as np
import torch
import torch.nn as nn


class ModelEnsemble:
    """Ensemble of multiple models with weighted averaging."""

    def __init__(self, models: list[nn.Module], weights: list[float] | None = None):
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)

    @torch.no_grad()
    def predict_classification(self, images: torch.Tensor) -> torch.Tensor:
        all_probs = []
        for model in self.models:
            model.eval()
            probs = torch.softmax(model(images), dim=1)
            all_probs.append(probs)

        weighted = torch.zeros_like(all_probs[0])
        for probs, w in zip(all_probs, self.weights):
            weighted += probs * w
        return weighted

    @torch.no_grad()
    def predict_segmentation(self, images: torch.Tensor) -> torch.Tensor:
        all_masks = []
        for model in self.models:
            model.eval()
            all_masks.append(model(images))

        weighted = torch.zeros_like(all_masks[0])
        for mask, w in zip(all_masks, self.weights):
            weighted += mask * w
        return weighted


def apply_tta_classification(model, image, transforms_list, device):
    """Test-Time Augmentation for classification. Returns averaged probs."""
    model.eval()
    all_probs = []
    for transform in transforms_list:
        tensor = transform(image=image)["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1).cpu().numpy()[0]
            all_probs.append(probs)
    return np.mean(all_probs, axis=0)


def apply_tta_segmentation(model, image, image_size, device):
    """TTA for segmentation: original + horizontal flip, averaged."""
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    model.eval()
    masks = []

    tensor = transform(image=image)["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        masks.append(model(tensor).squeeze().cpu().numpy())

    flipped = np.fliplr(image).copy()
    tensor = transform(image=flipped)["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        mask = model(tensor).squeeze().cpu().numpy()
        masks.append(np.fliplr(mask))

    return np.mean(masks, axis=0)


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.
    Shows which image regions the model considers important.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input_val, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        """Generate GradCAM heatmap. Returns (H, W) array normalized 0-1."""
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, target_class].backward()

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1).squeeze())

        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.cpu().numpy()

    @staticmethod
    def overlay_heatmap(image, heatmap, alpha=0.4):
        """Overlay GradCAM heatmap on RGB image."""
        h, w = image.shape[:2]
        resized = cv2.resize(heatmap, (w, h))
        colored = cv2.applyColorMap(np.uint8(255 * resized), cv2.COLORMAP_JET)
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        return cv2.addWeighted(image, 1 - alpha, colored, alpha, 0)


def get_gradcam_target_layer(model, model_name):
    """Auto-detect target layer for GradCAM based on model architecture."""
    if "efficientnet" in model_name:
        return model.conv_head if hasattr(model, "conv_head") else list(model.children())[-3]
    elif "resnet" in model_name:
        return model.layer4[-1]
    elif "densenet" in model_name:
        return model.features[-1]
    else:
        layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
        if layers:
            return layers[-1]
        raise ValueError(f"Cannot find target layer for {model_name}")
