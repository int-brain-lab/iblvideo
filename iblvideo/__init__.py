
__version__ = '3.0.2'  # This is the only place where the version is hard coded, only adapt here

__dlc_version__ = 'v2.2'  # versioning for DLC weights/test data
__lp_version__ = 'v2.1'  # versioning for LP weights/test data

from iblvideo.run import run_session, run_queue
from iblvideo.weights import download_lp_models, download_weights
