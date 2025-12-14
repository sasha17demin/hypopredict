"""
DEPRECATED: This module has been renamed to features.chunking

This file provides backward compatibility imports.
Please update your imports to use:
    from hypopredict.features import chunking
"""
import warnings

warnings.warn(
    "hypopredict.train_test_split is deprecated. "
    "Use 'from hypopredict.features import chunking' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import from new location
from hypopredict.features.chunking import (
    chunkify,
    chunkify_day,
    chunkify_df,
    get_HG_onset_times,
    generate_target_labels,
    train_test_split_chunks
)


def hello():
    """Backward compatibility function."""
    return "Hello from hypopredict.train_test_split module (DEPRECATED)!!"


__all__ = [
    'hello',
    'chunkify',
    'chunkify_day',
    'chunkify_df',
    'get_HG_onset_times',
    'generate_target_labels',
    'train_test_split_chunks',
]
