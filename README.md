# iblvideo
Scripts for video recording on the dedicated video computer that comes with every recording rig.
Prerequisites are python 3 and PyCapture2 from FLIR/PointGrey. To record from multiple PointGrey cameras, download this repository, open capture/CameraFunctions.py and adjust the path to your disk where the videos should be stored (savePath) as well as the duration of the intended video and frame rates of the cameras, then run capture/MasterVideo.py inside the iblvideo/capture folder to start video acquisition. In addition to compressed videos, time stamps are saved.

Images are compressed on the fly using MJPG. Post acquisition, videos can further be compresed across frames using h.264 (mp4) compression. Find a command for doing so via ffmpeg in ffmpeg_commands. 
