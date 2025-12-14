"""
DEPRECATED: This module has been split into data.loaders and data.labels

This file provides backward compatibility imports.
Please update your imports to use:
    from hypopredict.data import loaders, labels
"""
import warnings

warnings.warn(
    "hypopredict.compressor is deprecated. "
    "Use 'from hypopredict.data import loaders, labels' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import from new locations
from hypopredict.data.loaders import (
    gdrive_to_pandas,
    parse_compress_csv,
    save_to_feather,
    GLUCOSE_ID_LINKS
)

from hypopredict.data.labels import (
    identify_hg_events,
    plot_hg_events,
    day_count_HG
)

__all__ = [
    'gdrive_to_pandas',
    'parse_compress_csv',
    'save_to_feather',
    'GLUCOSE_ID_LINKS',
    'identify_hg_events',
    'plot_hg_events',
    'day_count_HG',
]
