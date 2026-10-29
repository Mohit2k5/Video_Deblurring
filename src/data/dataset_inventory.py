import os
from pathlib import Path
import pandas as pd

def discover_datasets(raw_dir: Path, supported_exts: list):
    inventory = []
    ext_counts = {}
    
    if not raw_dir.exists():
        return pd.DataFrame(inventory), pd.DataFrame(list(ext_counts.items()), columns=['Extension', 'Count'])

    for dataset_dir in raw_dir.iterdir():
        if not dataset_dir.is_dir():
            continue
        dataset_name = dataset_dir.name
        
        for seq_dir in dataset_dir.iterdir():
            if not seq_dir.is_dir():
                continue
            seq_name = seq_dir.name
            
            image_count = 0
            for item in seq_dir.iterdir():
                if item.is_file():
                    ext = item.suffix.lower()
                    if ext in supported_exts:
                        image_count += 1
                        ext_counts[ext] = ext_counts.get(ext, 0) + 1
                        
            inventory.append({
                'dataset': dataset_name,
                'sequence': seq_name,
                'image_count': image_count
            })
            
    df_inv = pd.DataFrame(inventory) if inventory else pd.DataFrame(columns=['dataset', 'sequence', 'image_count'])
    df_ext = pd.DataFrame(list(ext_counts.items()), columns=['Extension', 'Count'])
    return df_inv, df_ext
