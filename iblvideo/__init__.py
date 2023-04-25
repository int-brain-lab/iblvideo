
__version__ = 'iblvideo_2.2.1'  # This is the only place where the version is hard coded, only adapt here

import deeplabcut

from iblvideo.run import run_session, run_queue
from iblvideo.pose_dlc import dlc
from iblvideo.weights import download_weights
