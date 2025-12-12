"""
Functions for generating target labels from glucose data.
"""

import pandas as pd


def generate_target_labels_aligned(
    chunks_with_time: list[tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]], 
    glucose_df: pd.DataFrame, 
    threshold: float = 3.9, 
    forecast_window: pd.Timedelta = pd.Timedelta(minutes=30)
) -> list[int]:
    """
    Generates binary target labels for each chunk based on aligned glucose data.
    
    Args:
        chunks_with_time: List of tuples (chunk_df, start_time, end_time)
        glucose_df: DataFrame with glucose readings and datetime index
        threshold: Glucose level threshold for hypoglycemia (mmol/L)
        forecast_window: Time window after chunk end to check for hypoglycemia
        
    Returns:
        List of binary labels (1 = hypoglycemia predicted, 0 = no hypoglycemia)
    """
    target_labels = []
    hypo_times = glucose_df[glucose_df['glucose'] <= threshold].index
    
    for chunk, start_time, end_time in chunks_with_time:
        label = 0
        for onset_time in hypo_times:
            if end_time < onset_time <= end_time + forecast_window:
                label = 1
                break
        target_labels.append(label)
    return target_labels