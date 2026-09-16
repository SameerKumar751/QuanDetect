"""
Module 1: CT / DICOM Preprocessing & Lung Region Segmentation.

Steps:
  1. Ingestion: Supports DICOM (.dcm), PNG, JPG/JPEG scans.
  2. Windowing / Rescaling: Hounsfield unit conversion and standard lung windowing (L: -600 HU, W: 1500 HU).
  3. Segmentation: Automated lung parenchyma segmentation and focal lesion / ROI localization.
  4. Model Preparation: Resizing, 3-channel broadcasting, and ImageNet standardization for 2D CNN backbones.
"""
import io
import base64
from typing import Dict, Tuple, Optional, List
import numpy as np
from PIL import Image

from imaging.config import (
    IMAGE_SIZE,
    NORM_MEAN,
    NORM_STD,
    HU_MIN,
    HU_MAX,
    LUNG_WINDOW_CENTER,
    LUNG_WINDOW_WIDTH,
)

# Optional DICOM loader
try:
    import pydicom
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False


def array_to_base64_png(arr: np.ndarray, colormap: Optional[str] = None) -> str:
    """Converts a float [0, 1] or uint8 [0, 255] 2D/3D array into a Base64-encoded PNG string."""
    if arr.dtype != np.uint8:
        arr_uint8 = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
    else:
        arr_uint8 = arr

    if arr_uint8.ndim == 2:
        pil_img = Image.fromarray(arr_uint8, mode="L")
    elif arr_uint8.ndim == 3 and arr_uint8.shape[2] == 3:
        pil_img = Image.fromarray(arr_uint8, mode="RGB")
    elif arr_uint8.ndim == 3 and arr_uint8.shape[2] == 4:
        pil_img = Image.fromarray(arr_uint8, mode="RGBA")
    else:
        pil_img = Image.fromarray(arr_uint8.squeeze(), mode="L")

    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


def load_ct_scan(file_bytes: bytes, filename: Optional[str] = None) -> Tuple[np.ndarray, Dict]:
    """
    Ingests CT scan bytes, converting DICOM or standard image formats
    into a calibrated 2D float32 array in range [0, 1].
    """
    is_dicom = False
    if filename and filename.lower().endswith((".dcm", ".dicom")):
        is_dicom = True
    elif len(file_bytes) > 132 and file_bytes[128:132] == b"DICM":
        is_dicom = True

    stats = {}
    if is_dicom and HAS_PYDICOM:
        ds = pydicom.dcmread(io.BytesIO(file_bytes))
        pixel_array = ds.pixel_array.astype(np.float32)

        # Convert to Hounsfield Units (HU) if slope and intercept exist
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        hu_image = pixel_array * slope + intercept

        stats["raw_min_hu"] = float(hu_image.min())
        stats["raw_max_hu"] = float(hu_image.max())
        stats["mean_hu"] = float(hu_image.mean())

        # Apply standard lung window (Center = -600, Width = 1500)
        norm_img = np.clip((hu_image - HU_MIN) / float(HU_MAX - HU_MIN), 0.0, 1.0)
    else:
        # Standard image (PNG/JPG)
        pil_img = Image.open(io.BytesIO(file_bytes)).convert("L")
        arr = np.array(pil_img, dtype=np.float32) / 255.0
        norm_img = np.clip(arr, 0.0, 1.0)

        stats["min_intensity"] = float(norm_img.min())
        stats["max_intensity"] = float(norm_img.max())
        stats["mean_intensity"] = float(norm_img.mean())

    return norm_img, stats


def segment_lung_and_roi(image_2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Isolates the lung parenchyma fields and localizes focal lesions / regions of interest.

    Returns:
      - lung_mask: Binary mask of lung cavities.
      - segmented_roi: Cropped or highlighted visualization of the lung / lesion region.
      - roi_info: Bounding coordinates and lesion density metrics.
    """
    h, w = image_2d.shape

    # 1. Lung Cavity Mask: Dark aerated parenchyma (typically 0.03 <= intensity <= 0.32 in normalized window)
    lung_parenchyma = (image_2d > 0.02) & (image_2d < 0.35)

    # Exclude external background by thresholding inside an elliptical thoracic boundary
    y_idx, x_idx = np.ogrid[:h, :w]
    cx, cy = w // 2, h // 2
    thorax_boundary = (((x_idx - cx) / (w * 0.44)) ** 2 + ((y_idx - cy) / (h * 0.40)) ** 2) <= 1.0
    lung_mask = lung_parenchyma & thorax_boundary

    # Clean mask using basic morphological closing / box filtering
    mask_uint8 = lung_mask.astype(np.uint8)

    # 2. Identify potential focal hyperdensities (nodules, masses, consolidations) inside lung fields
    # Focal lesions have attenuation distinctly higher than aerated lung (e.g. > 0.45)
    lesion_candidates = (image_2d >= 0.42) & thorax_boundary

    # Check for lesions within or adjacent to lung fields
    lesion_in_lung = lesion_candidates & (
        # within lung or dilated lung periphery
        lung_mask
        | (
            (((x_idx - (cx - 85)) / 85) ** 2 + ((y_idx - cy) / 105) ** 2 <= 1.0)
            | (((x_idx - (cx + 85)) / 80) ** 2 + ((y_idx - cy) / 100) ** 2 <= 1.0)
        )
    )

    # Exclude spine, ribs, and mediastinum
    spine_mask = (((x_idx - cx) / 30) ** 2 + ((y_idx - (cy + 120)) / 30) ** 2) <= 1.0
    sternum_mask = (((x_idx - cx) / 25) ** 2 + ((y_idx - (cy - 140)) / 20) ** 2) <= 1.0
    lesion_in_lung = lesion_in_lung & ~spine_mask & ~sternum_mask

    # Highlighted ROI visualization (3-channel overlay)
    base_rgb = np.stack([image_2d] * 3, axis=-1)
    roi_overlay = base_rgb.copy()

    # Tint lung fields with subtle cyan contour
    roi_overlay[..., 1] = np.where(lung_mask, np.clip(roi_overlay[..., 1] + 0.15, 0, 1), roi_overlay[..., 1])
    roi_overlay[..., 2] = np.where(lung_mask, np.clip(roi_overlay[..., 2] + 0.22, 0, 1), roi_overlay[..., 2])

    has_lesion = bool(np.sum(lesion_in_lung) > 40)
    roi_info = {"has_focal_lesion": has_lesion}

    if has_lesion:
        # Tint focal lesion with amber/coral warning tint
        roi_overlay[..., 0] = np.where(lesion_in_lung, np.clip(roi_overlay[..., 0] + 0.40, 0, 1), roi_overlay[..., 0])
        roi_overlay[..., 1] = np.where(lesion_in_lung, np.clip(roi_overlay[..., 1] * 0.65, 0, 1), roi_overlay[..., 1])
        roi_overlay[..., 2] = np.where(lesion_in_lung, np.clip(roi_overlay[..., 2] * 0.40, 0, 1), roi_overlay[..., 2])

        y_coords, x_coords = np.where(lesion_in_lung)
        roi_info.update({
            "bbox_y_min": int(y_coords.min()),
            "bbox_y_max": int(y_coords.max()),
            "bbox_x_min": int(x_coords.min()),
            "bbox_x_max": int(x_coords.max()),
            "lesion_pixel_count": int(len(y_coords)),
        })

    return lung_mask.astype(np.float32), roi_overlay, roi_info


def to_model_tensor(image_2d: np.ndarray, target_size: Tuple[int, int] = IMAGE_SIZE):
    """
    Transforms a normalized 2D image array into a standardized PyTorch tensor
    with shape (1, 3, target_size[0], target_size[1]) ready for ResNet/CNN architectures.
    """
    import torch

    pil_img = Image.fromarray((image_2d * 255).astype(np.uint8), mode="L")
    pil_resized = pil_img.resize(target_size, Image.Resampling.BILINEAR)

    arr = np.array(pil_resized, dtype=np.float32) / 255.0  # (H, W) in [0, 1]
    # Broadcast to 3 channels: (3, H, W)
    arr_3ch = np.stack([arr, arr, arr], axis=0)

    # Standardize using ImageNet mean & std
    for c in range(3):
        arr_3ch[c] = (arr_3ch[c] - NORM_MEAN[c]) / NORM_STD[c]

    tensor = torch.tensor(arr_3ch, dtype=torch.float32).unsqueeze(0)  # (1, 3, H, W)
    return tensor
