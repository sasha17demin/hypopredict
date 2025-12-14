"""
Module for preprocessing chunkified CV data before feature extraction.
"""

import numpy as np

def filter_and_stack(split_chunkified, split_labels):
    """Filter out days with no chunks and stack all chunks and labels into arrays.

    Args:
        split_chunkified: dict of day -> chunks
        split_labels: dict of day -> chunk labels

    Returns:
        valid_chunks: list of (chunk_start_time, chunk_end_time) for all days
        valid_labels: np.array of labels for all chunks
    """

    valid_chunks = []
    valid_labels = []

    for day in split_chunkified.keys():
        chunks = split_chunkified[day]
        labels = split_labels[day]

        # keep those chunks and labels with labels >= 0
        mask = np.array(labels) >= 0
        # by index to preserve order
        chunks = [chunk for i, chunk in enumerate(chunks) if mask[i]]
        labels = [label for i, label in enumerate(labels) if mask[i]]
        # stack them into the final arrays
        valid_chunks.extend(chunks)
        valid_labels.extend(labels)

    return valid_chunks, np.array(valid_labels)
