
import time
from CameraFunctions import ConfigureCameras
from subprocess import *


# Configure camera settings
frameRate = 50
numCams = ConfigureCameras(frameRate)

# Start subprocesses for each camera
cam1 = Popen(['gnome-terminal', '-e', 'python /home/vid/iblvideo/CaptureCamera_1.py'], stdout=PIPE, stderr=PIPE)
cam2 = Popen(['gnome-terminal', '-e', 'python /home/vid/iblvideo/CaptureCamera_2.py'], stdout=PIPE, stderr=PIPE)
cam3 = Popen(['gnome-terminal', '-e', 'python /home/vid/iblvideo/CaptureCamera_3.py'], stdout=PIPE, stderr=PIPE)


