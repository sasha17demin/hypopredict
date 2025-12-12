"""
Functions for extracting ECG and HRV features from chunks.
"""

import pandas as pd
import neurokit2 as nk


def extract_features(chunks: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Extracts statistical features (mean, std, min, max, etc.) from each chunk.
    
    Args:
        chunks: List of DataFrames containing ECG data
        
    Returns:
        DataFrame with statistical features for each chunk
    """
    feature_rows = []
    for chunk in chunks:
        feats = pd.concat([
            chunk.mean().add_suffix("_mean"),
            chunk.std().add_suffix("_std"),
            chunk.min().add_suffix("_min"),
            chunk.max().add_suffix("_max"),
            chunk.quantile(0.25).add_suffix("_q25"),
            chunk.median().add_suffix("_median"),
            chunk.quantile(0.75).add_suffix("_q75"),
            chunk.skew().add_suffix("_skew"),
            chunk.kurtosis().add_suffix("_kurtosis")
        ])
        feature_rows.append(feats)
    return pd.DataFrame(feature_rows)


def extract_ecg_features(
    chunks: list[pd.DataFrame], 
    ecg_column: str = 'EcgWaveform',
    sampling_rate: int = 250
) -> pd.DataFrame:
    """
    Extracts ECG waveform features (time-domain statistics) from each chunk.
    
    Args:
        chunks: List of DataFrames containing ECG data
        ecg_column: Name of the ECG column
        sampling_rate: Sampling rate of the ECG signal (Hz)
        
    Returns:
        DataFrame with ECG features for each chunk
    """
    feature_rows = []
    for chunk in chunks:
        ecg_signal = chunk[ecg_column]
        
        stats = pd.Series({
            "mean_ecg": ecg_signal.mean(),
            "std_ecg": ecg_signal.std(),
            "min_ecg": ecg_signal.min(),
            "max_ecg": ecg_signal.max(),
        })
        feature_rows.append(stats)

    return pd.DataFrame(feature_rows)


def extract_hrv_features(
    chunks: list[pd.DataFrame], 
    ecg_column: str = 'EcgWaveform',
    sampling_rate: int = 250
) -> list[pd.DataFrame]:
    """
    Extracts HRV features (time-domain, frequency-domain, and nonlinear) from each chunk.
    
    Args:
        chunks: List of DataFrames containing ECG data
        ecg_column: Name of the ECG column
        sampling_rate: Sampling rate of the ECG signal (Hz)
        
    Returns:
        List of DataFrames with HRV features for each chunk
    """
    feature_rows = []
    for chunk in chunks:
        ecg_signal = chunk[ecg_column]
        ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)
        rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate)[1]
        hrv_features = nk.hrv(rpeaks, sampling_rate=sampling_rate, show=False)
        feature_rows.append(hrv_features)

    return feature_rows


def process_chunk(
    chunk: pd.DataFrame, 
    ecg_column: str = 'EcgWaveform',
    sampling_rate: int = 250
) -> pd.Series:
    """
    Process a single chunk to extract ECG and HRV features.
    
    Args:
        chunk: DataFrame containing ECG data
        ecg_column: Name of the ECG column
        sampling_rate: Sampling rate of the ECG signal (Hz)
        
    Returns:
        Series with combined ECG and HRV features
    """
    ecg_signal = chunk[ecg_column]
    ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)
    
    # Extract R-peaks (only once)
    rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate)[1]
    
    # Extract ECG time-domain statistics
    ecg_stats = pd.Series({
        "mean_ecg": ecg_signal.mean(),
        "std_ecg": ecg_signal.std(),
        "min_ecg": ecg_signal.min(),
        "max_ecg": ecg_signal.max(),
    })
    
    # Extract HRV features
    try:
        hrv_features = nk.hrv(rpeaks, sampling_rate=sampling_rate, show=False)
        hrv_features = hrv_features.iloc[0]  # Convert single-row DataFrame to Series
    except Exception:
        hrv_features = pd.Series(dtype=float)
    
    # Combine features
    combined = pd.concat([ecg_stats, hrv_features])
    return combined


def extract_combined_features_sequential(
    chunks: list[pd.DataFrame],
    ecg_column: str = 'EcgWaveform',
    sampling_rate: int = 250,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Extract combined ECG and HRV features sequentially.
    
    Args:
        chunks: List of DataFrames containing ECG data
        ecg_column: Name of the ECG column
        sampling_rate: Sampling rate of the ECG signal (Hz)
        verbose: Whether to print progress
        
    Returns:
        DataFrame with combined features for each chunk
    """
    results = []
    for i, chunk in enumerate(chunks):
        if verbose:
            print(f"Processing chunk {i+1}/{len(chunks)}")
        try:
            result = process_chunk(chunk, ecg_column, sampling_rate)
            results.append(result)
        except Exception as e:
            if verbose:
                print(f"Error processing chunk {i}: {e}")
            results.append(pd.Series(dtype=float))
    return pd.DataFrame(results).reset_index(drop=True)