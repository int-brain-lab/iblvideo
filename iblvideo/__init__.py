
__version__ = '3.0.0'  # This is the only place where the version is hard coded, only adapt here

# import deeplabcut

from iblvideo.run import run_session, run_queue
# from iblvideo.pose_dlc import dlc
from iblvideo.pose_lit import lightning_pose
from iblvideo.weights import download_lit_model  # , download_weights
