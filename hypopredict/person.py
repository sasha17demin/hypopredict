"""
DEPRECATED: This module has been moved to hypopredict.core.person

This file provides backward compatibility imports.
Please update your imports to use:
    from hypopredict.core import Person
"""
import warnings

warnings.warn(
    "hypopredict.person is deprecated. Use 'from hypopredict.core import Person' instead.",
    DeprecationWarning,
    stacklevel=2
)

from hypopredict.core.person import Person

__all__ = ['Person']
