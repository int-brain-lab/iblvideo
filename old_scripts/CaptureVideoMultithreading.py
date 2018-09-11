"""
Capture from multiple cameras using PyCapture2
"""
import PyCapture2
from CameraFunctions import ConfigureCameras
import threading
import time
import os
import numpy as np

# Path to save the video
save_path = '/media/vid/547f640f-3dbc-4419-a8f2-e4e715b088ba/test'
os.chdir(save_path)

# Settings
duration = 10
frameRate = 60

# Define multi threading class
class camThread(threading.Thread):
    def __init__(self, camIndex):
        threading.Thread.__init__(self)
        self.camIndex = camIndex
    def run(self):
        camCapture(self.camIndex)

# Define capture fuction
def camCapture(camIndex):
    # Initialize camera
    bus = PyCapture2.BusManager()
    cam = PyCapture2.Camera()
    avi = PyCapture2.AVIRecorder()
    cam.connect(bus.getCameraFromIndex(camIndex))
    fRateProp = cam.getProperty(PyCapture2.PROPERTY_TYPE.FRAME_RATE)
    frameRateCam = fRateProp.absValue
    cam.setConfiguration(numBuffers = 300, grabMode = 1)
    camInfo = cam.getCameraInfo()
    camID = str(camInfo.serialNumber)

    # Start camera capture
    print('Starting camera ', camID)
    TS = []
    Pin = []
    cam.startCapture()
    avi.MJPGOpen(camID.encode('utf-8'), frameRate, 75) # Save to AVI with JPEG compression
    while (time.time()-startTime) < duration:
        # Get image from camera buffer
        try:
            image = cam.retrieveBuffer()
        except PyCapture2.Fc2error as fc2Err:
            print("Error retrieving buffer : ", fc2Err)
            continue

        # Get timestamp for this frame
        timeStamp = image.getTimeStamp()
        TS.append(int(((timeStamp.seconds-startTime)*1000000 + timeStamp.microSeconds)/1000))   

        # Get pin status
        #imageData = image.getData()
        #Pin.append(imageData[7])     
        
        # Add frame to AVI
        avi.append(image)

    # Save timestamps
    np.save('TimeStamps_' + camID, TS)
    np.save('SyncTrigger_' + camID, Pin)

    # Disconnect cameras
    cam.stopCapture()
    cam.disconnect()
    print('Camera ' + camID + ' is done')

# Configure camera settings
numCams = ConfigureCameras(frameRate)

bus = []
startTime = time.time()
for j in range(numCams):
   thread = camThread(numCams)
   thread.start()



