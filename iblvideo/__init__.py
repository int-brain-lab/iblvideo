
__version__ = '3.1.0'     # This is the only place where the version is hard coded, only adapt here

__dlc_version__ = 'v2.2'  # versioning for DLC weights/test data
__lp_version__ = 'v2.1'   # versioning for LP weights/test data
__la_version__ = 'v1.2'   # versioning for LA weights/test data

import warnings

from iblvideo.weights import (
    download_la_models as download_la_models,
    download_lp_models as download_lp_models,
    download_weights as download_weights,
)

try:
    from iblvideo.pose_dlc import dlc as dlc
except ImportError:
    warnings.warn(
        'deeplabcut not installed; dlc() unavailable. Install with pip install deeplabcut.'
    )

try:
    from iblvideo.pose_lp import lightning_pose as lightning_pose
except ImportError:
    warnings.warn(
        'lightning-pose not installed; lightning_pose() unavailable. '
        'Install with pip install -e ".[pose]".'
    )

try:
    from iblvideo.segmentation_la import lightning_action as lightning_action
except ImportError:
    warnings.warn(
        'lightning-action not installed; lightning_action() unavailable. '
        'Install with pip install -e ".[action]".'
    )
