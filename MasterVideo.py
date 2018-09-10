
import time
from CameraFunctions import ConfigureCameras
from subprocess import *


# Configure camera settings
numCams = ConfigureCameras()

# Start subprocesses for each camera
print('Cameras configured, starting %s cameras'%numCams)
for i in range(numCams):
    eval('Popen([\'gnome-terminal\', \'-e\', \'python /home/vid/iblvideo/CaptureCamera_'+ str(i+1) + '.py\'], stdout=PIPE, stderr=PIPE)')


