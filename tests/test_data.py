import pytest
from pathlib import Path
import pandas as pd
from src.data.dataset_inventory import discover_datasets
from src.data.sequence_analysis import analyze_sequences

def test_discover_datasets_empty(tmp_path):
    df_inv, df_ext = discover_datasets(tmp_path, ['.png'])
    assert df_inv.empty

def test_analyze_sequences():
    df = pd.DataFrame({
        'dataset': ['train', 'train'],
        'sequence': ['000', '001'],
        'image_count': [100, 100]
    })
    res = analyze_sequences(df)
    assert not res.empty
    assert res['total_sequences'].iloc[0] == 2
