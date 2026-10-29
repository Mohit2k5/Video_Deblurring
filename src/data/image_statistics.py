from pathlib import Path
import pandas as pd
import cv2
import numpy as np
from PIL import Image
import random

def compute_statistics(raw_dir: Path, supported_exts: list, sample_limit: int = 100, seed: int = 42):
    random.seed(seed)
    stats = []
    
    if not raw_dir.exists():
        return pd.DataFrame(stats)
        
    all_files = [p for p in raw_dir.rglob('*') if p.is_file() and p.suffix.lower() in supported_exts]
    if sample_limit and len(all_files) > sample_limit:
        all_files = random.sample(all_files, sample_limit)
        
    for item in all_files:
        try:
            with Image.open(item) as img:
                width, height = img.size
                mode = img.mode
                
            img_cv = cv2.imread(str(item), cv2.IMREAD_GRAYSCALE)
            if img_cv is not None:
                mean_bright = np.mean(img_cv)
                std_contrast = np.std(img_cv)
                laplacian_var = cv2.Laplacian(img_cv, cv2.CV_64F).var()
            else:
                mean_bright, std_contrast, laplacian_var = None, None, None
                
            stats.append({
                'dataset': item.parent.parent.name,
                'sequence': item.parent.name,
                'filename': item.name,
                'width': width,
                'height': height,
                'mode': mode,
                'size_bytes': item.stat().st_size,
                'mean_brightness': mean_bright,
                'std_contrast': std_contrast,
                'sharpness_variance': laplacian_var
            })
        except Exception:
            pass
            
    return pd.DataFrame(stats)
