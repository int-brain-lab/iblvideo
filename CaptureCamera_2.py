"""
Capture from multiple cameras using PyCapture2
"""
import PyCapture2
from CameraFunctions import ConfigureCameras
import time
import os
import numpy as np

# Path to save the video
save_path = '/media/vid/547f640f-3dbc-4419-a8f2-e4e715b088ba/test'
os.chdir(save_path)

# Settings
camIndex = 1
duration = 30
frameRate = 50

# Initialize camera
bus = PyCapture2.BusManager()
cam = PyCapture2.Camera()
avi = PyCapture2.AVIRecorder()
cam.connect(bus.getCameraFromIndex(camIndex))
cam.setProperty(type = 16, onOff = True, autoManualMode = False, absValue = frameRate)
fRateProp = cam.getProperty(PyCapture2.PROPERTY_TYPE.FRAME_RATE)
frameRateCam = fRateProp.absValue
cam.setConfiguration(numBuffers = 500, grabMode = 1)
camInfo = cam.getCameraInfo()
camID = str(camInfo.serialNumber)

# Start camera capture
print('Starting camera ', camID)
TS = []
Pin = []
cam.startCapture()
camTime = time.time()
startTime = time.time()
frameCount = 0
avi.MJPGOpen(camID.encode('utf-8'), frameRate, 75) # Save to AVI with JPEG compression
while (time.time()-startTime) < duration:
    try:
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
        imageData = image.getData()
        Pin.append(imageData[0])
        
        # Add frame to AVI
        avi.append(image)
        
        # Print output
        if time.time()-camTime > 10:    
           print('Elapsed time ' + str(int((time.time()-startTime)/60)) + ' minutes and ' + str(int(np.mod(time.time()-startTime, 60))) + ' seconds') 
           print('Recorded %s frames with a frame rate of '%frameCount + str(round(frameCount/(time.time()-startTime), 1)))
           camTime = time.time()
        frameCount = frameCount + 1

    except Exception as e:
        # Save timestamps
        np.save('TimeStamps_' + camID, TS)
        np.save('SyncTrigger_' + camID, Pin)

        # Disconnect cameras
        cam.stopCapture()
        cam.disconnect()
        print('Camera ' + camID + ' has crashed')
        print(e)
        time.sleep(60)

# Save timestamps
np.save('TimeStamps_' + camID, TS)
np.save('SyncTrigger_' + camID, Pin)

# Disconnect cameras
cam.stopCapture()
cam.disconnect()
print('Camera ' + camID + ' is done')
time.sleep(60)




