import pandas as pd

def analyze_sequences(df_inv: pd.DataFrame):
    if df_inv.empty:
        return pd.DataFrame()
        
    summary = df_inv.groupby('dataset').agg(
        total_sequences=('sequence', 'count'),
        min_frames=('image_count', 'min'),
        max_frames=('image_count', 'max'),
        mean_frames=('image_count', 'mean'),
        median_frames=('image_count', 'median')
    ).reset_index()
    
    return summary
