
import time
from CameraFunctions import ConfigureCameras
from subprocess import *


# Configure camera settings
frameRate = 50
numCams = ConfigureCameras(frameRate)

cam1 = Popen(['gnome-terminal', '-e', 'python /home/vid/iblvideo/CaptureCamera_1.py'], stdout=PIPE, stderr=PIPE)
stdout = cam1.stdout.read()
stderr = cam1.stderr.read()
if stdout:
	print(stdout)
if stderr:
	print(stderr)
cam2 = Popen(['gnome-terminal', '-e', 'python /home/vid/iblvideo/CaptureCamera_2.py'], stdout=PIPE, stderr=PIPE)
stdout = cam1.stdout.read()
stderr = cam1.stderr.read()
if stdout:
	print(stdout)
if stderr:
	print(stderr)
cam3 = Popen(['gnome-terminal', '-e', 'python /home/vid/iblvideo/CaptureCamera_3.py'], stdout=PIPE, stderr=PIPE)
stdout = cam1.stdout.read()
stderr = cam1.stderr.read()
if stdout:
	print(stdout)
if stderr:
	print(stderr)

