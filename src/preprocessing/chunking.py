"""
Functions for chunking ECG data into time windows.
"""

import pandas as pd


def chunkify_with_time(
    df: pd.DataFrame, 
    chunk_size: int, 
    step_size: int
) -> list[tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]]:
    """
    Splits sensor data into overlapping chunks and returns chunks with their time ranges.
    
    Args:
        df: DataFrame with datetime index containing ECG data
        chunk_size: Number of samples per chunk
        step_size: Number of samples to step between chunks
        
    Returns:
        List of tuples (chunk_df, start_time, end_time)
    """
    chunks = []
    for start in range(0, df.shape[0], step_size):
        end = start + chunk_size
        if end < df.shape[0]:
            chunk = df.iloc[start:end]
        else:
            chunk = df.iloc[start:]
        chunks.append((chunk, chunk.index[0], chunk.index[-1]))
    return chunks