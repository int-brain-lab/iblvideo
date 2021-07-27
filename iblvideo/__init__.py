
__version__ = '1.2.0'  # This is the only place where the version is hard coded, only adapt here

import deeplabcut

from iblvideo.run import run_session, run_queue
from iblvideo.choiceworld import dlc
from iblvideo.weights import download_weights
