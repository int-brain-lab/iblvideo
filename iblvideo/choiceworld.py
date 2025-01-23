import logging
import warnings

from iblvideo.pose_dlc import dlc  # Import the new module

_logger = logging.getLogger('ibllib')

# Emit a DeprecationWarning
warnings.warn(
    """
    The `iblvideo.choiceworld` module will be deprecated in a future release.
    Replace:
        from iblvideo.choiceworld import dlc
    with:
        from iblvideo.pose_dlc import dlc
    """,
    DeprecationWarning,
    stacklevel=2,
)

# Log a warning for visibility
_logger.warning(
    """
    Deprecation Warning: The `iblvideo.choiceworld` module will be deprecated in a future release.
    Update your imports to:
        from iblvideo.pose_dlc import dlc
    """
)

# Maintain full compatibility with the new module
__all__ = ['dlc']
