"""
Module: Explainability — Visual Gradient-Weighted Class Activation Mapping (Grad-CAM).

Generates localized visual heatmaps highlighting the anatomical and pathological
regions in the CT scan that the 2D CNN focused on when making its diagnostic prediction.
"""
from typing import Tuple, Optional
import io
import base64
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from imaging.models.classical_cnn import get_cnn_model, LungCancerCNN
from imaging.config import IDX_TO_CLASS


class GradCAM:
    """Computes Grad-CAM heatmaps for a specified PyTorch CNN model and target layer."""

    def __init__(self, model: LungCancerCNN, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register forward and backward hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """
        Generates a 2D normalized [0, 1] Grad-CAM heatmap for the input tensor.
        """
        self.model.eval()
        self.model.zero_grad()

        input_var = input_tensor.clone().requires_grad_(True)
        logits = self.model(input_var)

        if target_class is None:
            target_class = int(torch.argmax(logits, dim=-1).item())

        score = logits[0, target_class]
        score.backward(retain_graph=True)

        gradients = self.gradients[0]     # (C, H_feat, W_feat)
        activations = self.activations[0] # (C, H_feat, W_feat)

        # Global average pooling of gradients to obtain importance weights alpha_k
        weights = torch.mean(gradients, dim=(1, 2), keepdim=True) # (C, 1, 1)

        # Weighted combination of feature activation maps
        cam = torch.sum(weights * activations, dim=0) # (H_feat, W_feat)

        # ReLU: only features with positive influence on the target class
        cam = F.relu(cam)

        # Upsample to input dimensions (224, 224)
        cam = cam.unsqueeze(0).unsqueeze(0) # (1, 1, H, W)
        cam = F.interpolate(cam, size=(input_tensor.shape[2], input_tensor.shape[3]), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        c_min, c_max = cam.min(), cam.max()
        if c_max - c_min > 1e-6:
            cam = (cam - c_min) / (c_max - c_min)
        else:
            cam = np.zeros_like(cam)

        return cam


def apply_jet_colormap(cam: np.ndarray) -> np.ndarray:
    """Applies a smooth JET/Turbo-style heatmap (R, G, B) to a [0, 1] float array."""
    cam_flat = np.clip(cam, 0.0, 1.0)
    # JET formula approximation:
    r = np.clip(1.5 - np.abs(4.0 * cam_flat - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * cam_flat - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * cam_flat - 1.0), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def generate_gradcam_overlay(
    raw_image_2d: np.ndarray,
    input_tensor: torch.Tensor,
    target_class: Optional[int] = None,
    alpha: float = 0.52,
) -> Tuple[str, str]:
    """
    Computes Grad-CAM for the CT input tensor, blends the resulting heatmap
    with the original grayscale CT image, and encodes as a Base64 PNG.

    Returns:
      - overlay_base64: Base64 data URL
      - attention_summary: Descriptive clinical narrative of the model's focus
    """
    model = get_cnn_model()
    target_layer = model.get_last_conv_layer()
    gradcam = GradCAM(model, target_layer)

    cam = gradcam.generate(input_tensor, target_class=target_class)

    # Resize raw image to match CAM size
    h, w = cam.shape
    pil_raw = Image.fromarray((np.clip(raw_image_2d, 0, 1) * 255).astype(np.uint8), mode="L")
    pil_raw = pil_raw.resize((w, h), Image.Resampling.BILINEAR)
    base_ct = np.array(pil_raw, dtype=np.float32) / 255.0
    base_ct_rgb = np.stack([base_ct] * 3, axis=-1)

    # Generate color heatmap
    heatmap_rgb = apply_jet_colormap(cam)

    # Blend original CT with heatmap
    # Highlight areas where cam > 0.15 with higher weight
    weight_map = np.clip(cam[..., None] * 1.3, 0.0, 0.85) * alpha
    blended = (1.0 - weight_map) * base_ct_rgb + weight_map * heatmap_rgb
    blended = np.clip(blended, 0.0, 1.0)

    # Convert to Base64 PNG
    blended_uint8 = (blended * 255).astype(np.uint8)
    pil_blended = Image.fromarray(blended_uint8, mode="RGB")
    buffer = io.BytesIO()
    pil_blended.save(buffer, format="PNG")
    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    overlay_base64 = f"data:image/png;base64,{b64_str}"

    # Determine peak activation centroid
    y_coords, x_coords = np.where(cam >= 0.70 * cam.max())
    if len(y_coords) > 0:
        cy_norm = float(np.mean(y_coords) / h)
        cx_norm = float(np.mean(x_coords) / w)
        vert_loc = "Upper" if cy_norm < 0.4 else ("Lower" if cy_norm > 0.6 else "Middle")
        horiz_loc = "Right (image left)" if cx_norm < 0.45 else ("Left (image right)" if cx_norm > 0.55 else "Central")
        loc_str = f"{vert_loc} {horiz_loc} lung field"
    else:
        loc_str = "lung parenchyma"

    label_str = IDX_TO_CLASS.get(target_class, "target") if target_class is not None else "predicted"
    attention_summary = (
        f"Grad-CAM attention peaks over the {loc_str}. High gradient weighting in this region "
        f"corresponds to the focal parenchymal density patterns driving the {label_str} classification."
    )

    return overlay_base64, attention_summary
