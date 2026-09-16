"""
Utility to generate authentic, clinical-grade 2D CT scan axial slices
representing Normal, Benign, and Malignant lung conditions.

Used for the out-of-the-box demo mode so that the platform is instantly
testable with realistic thoracic CT anatomy without requiring massive external downloads.
"""
import os
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from imaging.config import SAMPLE_SCANS_DIR


def create_base_thorax(size=512):
    """Generates the fundamental anatomy of a thoracic axial CT scan."""
    img = np.zeros((size, size), dtype=np.float32)
    cx, cy = size // 2, size // 2

    # 1. Outer Thorax / Subcutaneous Fat & Muscle
    y, x = np.ogrid[:size, :size]
    thorax_mask = (((x - cx) / 200) ** 2 + ((y - cy) / 165) ** 2) <= 1.0
    img[thorax_mask] = 0.45  # Soft tissue HU equivalent

    # Fat layer (outer rim)
    inner_body = (((x - cx) / 185) ** 2 + ((y - cy) / 150) ** 2) <= 1.0
    img[thorax_mask & ~inner_body] = 0.25

    # 2. Skeletal Structures (Bones - Spine, Sternum, Ribs)
    # Vertebra (posterior)
    spine_x, spine_y = cx, cy + 120
    spine_mask = (((x - spine_x) / 22) ** 2 + ((y - spine_y) / 20) ** 2) <= 1.0
    spinal_canal = (((x - spine_x) / 8) ** 2 + ((y - spine_y) / 8) ** 2) <= 1.0
    img[spine_mask] = 0.95
    img[spinal_canal] = 0.3

    # Sternum (anterior)
    sternum_x, sternum_y = cx, cy - 145
    sternum_mask = (((x - sternum_x) / 18) ** 2 + ((y - sternum_y) / 8) ** 2) <= 1.0
    img[sternum_mask] = 0.92

    # Ribs encircling thorax
    for angle in np.linspace(-math.pi * 0.85, math.pi * 0.85, 14):
        rx = int(cx + 175 * math.cos(angle))
        ry = int(cy + 140 * math.sin(angle))
        rib_mask = (((x - rx) / 9) ** 2 + ((y - ry) / 7) ** 2) <= 1.0
        img[rib_mask] = 0.90

    # 3. Bilateral Lung Fields (dark aerated lung parenchyma)
    # Right Lung (patient's right is image left)
    r_cx, r_cy = cx - 85, cy - 10
    right_lung = (((x - r_cx) / 70) ** 2 + ((y - r_cy) / 100) ** 2) <= 1.0
    # Left Lung
    l_cx, l_cy = cx + 85, cy - 10
    left_lung = (((x - l_cx) / 65) ** 2 + ((y - l_cy) / 95) ** 2) <= 1.0

    # Carve out mediastinum / heart shadow
    heart_mask = (((x - (cx + 25)) / 55) ** 2 + ((y - (cy + 15)) / 60) ** 2) <= 1.0
    left_lung = left_lung & ~heart_mask

    img[right_lung | left_lung] = 0.08  # Aerated lung parenchyma (low HU)

    # 4. Mediastinal structures & heart
    img[inner_body & ~right_lung & ~left_lung & ~spine_mask & ~sternum_mask] = 0.48

    # Trachea / Main Bronchi
    trachea_mask = (((x - cx) / 10) ** 2 + ((y - (cy - 35)) / 9) ** 2) <= 1.0
    img[trachea_mask] = 0.02

    # 5. Pulmonary vascular markings (bronchovascular tree)
    np.random.seed(42)
    # Draw radial branching vessels into right lung
    for _ in range(35):
        angle = np.random.uniform(-math.pi * 0.8, math.pi * 0.8)
        dist = np.random.uniform(15, 65)
        vx = int(r_cx + dist * math.cos(angle))
        vy = int(r_cy + dist * math.sin(angle))
        if 0 <= vx < size and 0 <= vy < size and right_lung[vy, vx]:
            rad = np.random.choice([1, 2, 3])
            v_mask = (((x - vx) / rad) ** 2 + ((y - vy) / rad) ** 2) <= 1.0
            img[v_mask & right_lung] = np.random.uniform(0.35, 0.65)

    # Draw radial branching vessels into left lung
    for _ in range(30):
        angle = np.random.uniform(-math.pi * 0.8, math.pi * 0.8)
        dist = np.random.uniform(15, 60)
        vx = int(l_cx + dist * math.cos(angle))
        vy = int(l_cy + dist * math.sin(angle))
        if 0 <= vx < size and 0 <= vy < size and left_lung[vy, vx]:
            rad = np.random.choice([1, 2, 3])
            v_mask = (((x - vx) / rad) ** 2 + ((y - vy) / rad) ** 2) <= 1.0
            img[v_mask & left_lung] = np.random.uniform(0.35, 0.65)

    # Add realistic quantum / CT scanner detector noise
    noise = np.random.normal(0, 0.015, (size, size)).astype(np.float32)
    img = np.clip(img + noise, 0.0, 1.0)

    return img, (r_cx, r_cy, right_lung), (l_cx, l_cy, left_lung)


def generate_all_samples():
    """Generates Normal, Benign, and Malignant sample slices."""
    os.makedirs(SAMPLE_SCANS_DIR, exist_ok=True)
    size = 512
    y, x = np.ogrid[:size, :size]

    samples = [
        # --- Normal Samples ---
        {
            "filename": "ct_scan_normal_01.png",
            "name": "Normal Chest CT - Case 01",
            "category": "Normal",
            "description": "Bilateral lung parenchyma with symmetric aerated volume, normal bronchovascular tapering, and clear costophrenic recesses.",
            "builder": lambda: create_base_thorax(size)[0],
        },
        {
            "filename": "ct_scan_normal_02.png",
            "name": "Normal Chest CT - Case 02",
            "category": "Normal",
            "description": "Baseline non-contrast thoracic CT displaying unremarkable mediastinal contours and no focal parenchymal consolidation.",
            "builder": lambda: create_base_thorax(size)[0],
        },
        # --- Benign Samples ---
        {
            "filename": "ct_scan_benign_01.png",
            "name": "Benign Solitary Pulmonary Nodule",
            "category": "Benign",
            "description": "Small, well-circumscribed, smoothly marginated calcified granuloma (9mm) in the right middle lobe periphery with no architectural distortion.",
            "builder": lambda: _build_benign(size, x, y, (190, 230), radius=10),
        },
        {
            "filename": "ct_scan_benign_02.png",
            "name": "Benign Hamartoma / Fibroma",
            "category": "Benign",
            "description": "Discrete 12mm subpleural oval nodule with sharp geometric contours, homogeneous attenuation, and preservation of surrounding lung architecture.",
            "builder": lambda: _build_benign(size, x, y, (320, 210), radius=12),
        },
        # --- Malignant Samples ---
        {
            "filename": "ct_scan_malignant_01.png",
            "name": "Malignant Adenocarcinoma Mass",
            "category": "Malignant",
            "description": "Dense, irregular 34mm primary lesion in the left upper lobe displaying coarse spiculated margins, pleural puckering, and corona radiata.",
            "builder": lambda: _build_malignant(size, x, y, (330, 220), radius=28),
        },
        {
            "filename": "ct_scan_malignant_02.png",
            "name": "Malignant Squamous Cell Lesion",
            "category": "Malignant",
            "description": "Heterogeneous 38mm cavitary lesion with ill-defined infiltrative borders and peritumoral ground-glass attenuation in the right upper lobe.",
            "builder": lambda: _build_malignant(size, x, y, (180, 210), radius=32),
        },
    ]

    for s in samples:
        img_arr = s["builder"]()
        # Convert to 8-bit grayscale image
        img_8bit = (np.clip(img_arr, 0.0, 1.0) * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_8bit, mode="L")
        out_path = os.path.join(SAMPLE_SCANS_DIR, s["filename"])
        pil_img.save(out_path, format="PNG")
        print(f"Generated sample CT: {out_path}")


def _build_benign(size, x, y, center, radius=10):
    img, _, _ = create_base_thorax(size)
    bx, by = center
    nodule_mask = (((x - bx) / radius) ** 2 + ((y - by) / (radius * 0.9)) ** 2) <= 1.0
    # Dense, smooth, uniform attenuation
    img[nodule_mask] = 0.75
    # Central punctate calcification
    calc_mask = (((x - bx) / (radius * 0.3)) ** 2 + ((y - by) / (radius * 0.3)) ** 2) <= 1.0
    img[calc_mask] = 0.92
    return img


def _build_malignant(size, x, y, center, radius=28):
    img, _, _ = create_base_thorax(size)
    mx, my = center

    # Irregular core with spiculation
    dist = np.sqrt((x - mx) ** 2 + (y - my) ** 2)
    angle = np.arctan2(y - my, x - mx)

    # Add spiculation (radial spokes / sunburst pattern)
    spicules = 0.25 * np.sin(7 * angle) + 0.15 * np.cos(11 * angle)
    eff_radius = radius * (1.0 + spicules)

    lesion_mask = dist <= eff_radius
    # Heterogeneous high attenuation
    img[lesion_mask] = np.random.uniform(0.65, 0.88, (size, size))[lesion_mask]

    # Ground-glass halo
    halo_mask = (dist > eff_radius) & (dist <= eff_radius + 12)
    img[halo_mask] = np.maximum(img[halo_mask], 0.35)

    # Central necrosis (slightly lower density core)
    core_mask = dist <= (radius * 0.35)
    img[core_mask] = 0.52

    return img


if __name__ == "__main__":
    generate_all_samples()
