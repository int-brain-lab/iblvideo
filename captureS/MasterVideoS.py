"""
Master script which spawns seperate consoles for each camera to run them in parallel
"""

from CameraFunctionsS import ConfigureCameras
from subprocess import *

# Configure camera settings
numCams = ConfigureCameras()

# Start subprocesses for each camera
print('Cameras configured, starting %s cameras'%numCams)
for i in range(numCams):
    Popen(['gnome-terminal','-e','python CaptureCameraS.py ' + str(i)])