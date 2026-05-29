
__version__ = '3.1.0'     # This is the only place where the version is hard coded, only adapt here

__dlc_version__ = 'v2.2'  # versioning for DLC weights/test data
__lp_version__ = 'v2.1'   # versioning for LP weights/test data
__la_version__ = 'v1.2'   # versioning for LA weights/test data

from iblvideo.pose_dlc import dlc
from iblvideo.pose_lp import lightning_pose
from iblvideo.segmentation_la import lightning_action
from iblvideo.weights import download_lp_models, download_la_models, download_weights
