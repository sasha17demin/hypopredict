"""
Functions for splitting data into train and test sets.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def train_test_split_chunks(
    chunks: list[pd.DataFrame], 
    target_labels: list[int], 
    test_size: float = 0.2
) -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[int], list[int]]:
    """
    Splits chunks and labels into train and test sets (chronological split).
    
    Args:
        chunks: List of DataFrames containing ECG data
        target_labels: List of binary labels
        test_size: Proportion of data to use for testing
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    total_chunks = len(chunks)
    num_test_chunks = int(np.ceil(total_chunks * test_size))
    split_index = total_chunks - num_test_chunks
    
    X_train = chunks[:split_index]
    y_train = target_labels[:split_index]
    X_test = chunks[split_index:]
    y_test = target_labels[split_index:]
    
    return X_train, X_test, y_train, y_test


def train_test_split_chunks_stratified(
    chunks: list[pd.DataFrame], 
    target_labels: list[int], 
    test_size: float = 0.2, 
    random_state: int = 42
) -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[int], list[int]]:
    """
    Splits chunks and labels into train and test sets using stratified sampling.
    This ensures both sets have similar proportions of positive/negative labels.
    
    Args:
        chunks: List of DataFrames containing ECG data
        target_labels: List of binary labels
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    indices = list(range(len(chunks)))
    
    train_indices, test_indices = train_test_split(
        indices, 
        test_size=test_size, 
        stratify=target_labels,
        random_state=random_state
    )
    
    X_train = [chunks[i] for i in train_indices]
    X_test = [chunks[i] for i in test_indices]
    y_train = [target_labels[i] for i in train_indices]
    y_test = [target_labels[i] for i in test_indices]
    
    return X_train, X_test, y_train, y_test