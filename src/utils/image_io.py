"""Image I/O utilities and synthetic radiographic sample generator for NDT testing."""

import io
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image


def load_radiograph(source: Union[str, Path, bytes, io.BytesIO]) -> np.ndarray:
    """Load a radiographic X-ray image from a file path or in-memory byte buffer.
    
    Args:
        source: File path (str/Path), raw bytes, or BytesIO buffer.
        
    Returns:
        Grayscale uint8 numpy array representing the radiograph.
    """
    if isinstance(source, (bytes, io.BytesIO)):
        data = source.getvalue() if isinstance(source, io.BytesIO) else source
        nparr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Fallback to PIL in case of exotic format
            pil_img = Image.open(io.BytesIO(data)).convert("L")
            img = np.array(pil_img, dtype=np.uint8)
        return img

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Radiograph image not found: {path}")

    # Read with OpenCV
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        pil_img = Image.open(path).convert("L")
        img = np.array(pil_img, dtype=np.uint8)

    return img


def save_radiograph(image: np.ndarray, output_path: Union[str, Path]) -> bool:
    """Save an image to disk.
    
    Args:
        image: Numpy array image.
        output_path: Target save destination.
        
    Returns:
        True if successfully saved.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(path), image))


def generate_synthetic_weld_sample(
    defect_type: str = "crack",
    width: int = 640,
    height: int = 360,
    random_seed: Optional[int] = 42,
) -> np.ndarray:
    """Synthesize a realistic radiographic X-ray weld inspection image.
    
    Simulates:
    1. Base metal plate with radiographic transmission attenuation (darker gray).
    2. Central weld seam / weld bead with increased metal thickness (lighter band)
       and weave ripples.
    3. Heat-Affected Zone (HAZ) intensity gradients.
    4. Film/detector quantum grain noise.
    5. Realistic defect types:
       - 'crack' (CR): Jagged, branching high-aspect-ratio dark fissure.
       - 'porosity' (PO): Clustered spherical dark gas pore cavities.
       - 'lack_of_penetration' (LP): Continuous dark straight linear root defect.
       - 'clean' (ND): Sound weld without rejectable indications.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # 1. Base metal background: base transmission value around 70-85
    base_val = 80.0
    image = np.full((height, width), base_val, dtype=np.float32)

    # Base metal thickness slight gradient
    y_coords, x_coords = np.indices((height, width))
    image += 3.0 * np.sin(x_coords / 70.0)

    # 2. Central weld seam (horizontal band in middle)
    weld_center_y = height // 2
    weld_width = 80
    dist_from_weld_center = np.abs(y_coords - weld_center_y)
    
    # Gaussian bell curve profile for weld crown reinforcement (thicker steel = more X-ray absorption = brighter image)
    weld_profile = np.exp(-0.5 * (dist_from_weld_center / (weld_width / 2.8)) ** 2)
    image += weld_profile * 75.0  # Weld peak brightness around 155

    # Weld bead ripple texture (freeze marks from manual or GMAW welding)
    ripples = np.sin(x_coords / 4.0 + 0.05 * (y_coords - weld_center_y) ** 2) * 4.0
    image += ripples * (weld_profile > 0.1)

    # 3. Defect injection (Lower absorption / lower density metal = darker radiographic indication)
    defect_type_lower = defect_type.lower()

    if "crack" in defect_type_lower or defect_type_lower == "cr":
        # Draw a jagged, branching transverse or longitudinal crack
        cx, cy = width // 2 - 40, weld_center_y - 10
        pts = [(cx, cy)]
        curr_x, curr_y = cx, cy
        for _ in range(35):
            curr_x += np.random.randint(2, 5)
            curr_y += np.random.randint(-2, 3)
            pts.append((curr_x, curr_y))
            # Occasional small micro-branch
            if np.random.rand() > 0.8:
                branch_x, branch_y = curr_x, curr_y
                for _ in range(8):
                    branch_x += np.random.randint(1, 4)
                    branch_y += np.random.randint(-3, 1)
                    pts.append((branch_x, branch_y))

        # Render crack onto attenuation map
        for x, y in pts:
            if 0 <= y < height and 0 <= x < width:
                # Main crack trough
                image[max(0, y-1):min(height, y+2), max(0, x-1):min(width, x+2)] -= 42.0

    elif "porosity" in defect_type_lower or defect_type_lower == "po":
        # Cluster of gas pores
        cluster_center_x, cluster_center_y = width // 2, weld_center_y
        num_pores = np.random.randint(9, 15)
        for _ in range(num_pores):
            px = cluster_center_x + np.random.randint(-60, 60)
            py = cluster_center_y + np.random.randint(-22, 22)
            radius = np.random.uniform(2.5, 6.0)
            
            # Sub-pixel spherical cavity attenuation
            y_min, y_max = max(0, int(py - radius - 2)), min(height, int(py + radius + 3))
            x_min, x_max = max(0, int(px - radius - 2)), min(width, int(px + radius + 3))
            
            for yy in range(y_min, y_max):
                for xx in range(x_min, x_max):
                    d = np.sqrt((xx - px)**2 + (yy - py)**2)
                    if d <= radius:
                        depth_factor = np.sqrt(max(0.0, 1.0 - (d / radius)**2))
                        image[yy, xx] -= 55.0 * depth_factor

    elif "penetration" in defect_type_lower or defect_type_lower == "lp":
        # Continuous linear lack of root penetration right at joint centerline
        root_y = weld_center_y
        lp_start_x = width // 4
        lp_end_x = (3 * width) // 4
        for x in range(lp_start_x, lp_end_x):
            # slight jitter
            y = root_y + int(np.random.normal(0, 0.4))
            image[y-1:y+2, x] -= 48.0

    elif "clean" in defect_type_lower or defect_type_lower == "nd":
        # Sound weld - no defect added
        pass

    # 4. Realistic radiographic quantum mottle / grain noise
    quantum_noise = np.random.normal(0, 3.8, (height, width))
    image += quantum_noise

    # Clip to valid 8-bit range
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def ensure_sample_radiographs(samples_dir: Union[str, Path]) -> Dict[str, Path]:
    """Generate default test radiographs if they do not exist."""
    dir_path = Path(samples_dir)
    dir_path.mkdir(parents=True, exist_ok=True)

    samples = {
        "crack_defect.png": ("crack", 101),
        "porosity_cluster.png": ("porosity", 202),
        "lack_of_penetration.png": ("lack_of_penetration", 303),
        "sound_weld_no_defect.png": ("clean", 404),
    }

    generated_paths = {}
    for filename, (def_type, seed) in samples.items():
        file_path = dir_path / filename
        if not file_path.exists():
            img = generate_synthetic_weld_sample(defect_type=def_type, random_seed=seed)
            save_radiograph(img, file_path)
        generated_paths[filename] = file_path

    return generated_paths
