"""
Capture from multiple cameras using PyCapture2
"""
import PyCapture2
from CameraFunctions import ConfigureCameras, CaptureSettings
import time
import os
import numpy as np

# Path to save the video
savePath, frameRates, duration = CaptureSettings()
os.chdir(savePath)

# Settings
camIndex = 2
frameRate = frameRates[camIndex]

# PyCapture
try:
    bus = PyCapture2.BusManager()
    cam = PyCapture2.Camera()
    avi = PyCapture2.AVIRecorder()

    #Check if camera is available
    numCams = bus.getNumOfCameras()
    if camIndex > numCams:
        print('Camera not found')
        exit()

    #Connect and configure camera
    cam.connect(bus.getCameraFromIndex(camIndex))
    cam.setProperty(type = 16, onOff = True, autoManualMode = False, absValue = frameRate)
    #cam.setConfiguration(numBuffers = 500, grabMode = PyCapture2.GRAB_MODE.BUFFER_FRAMES)
    camInfo = cam.getCameraInfo()
    camID = str(camInfo.serialNumber)

    # Start camera capture
    print('Started camera ', camID)
    TS = []
    Pin = []
    cam.startCapture()
    camTimer = time.time()
    pulseTimer = time.time()
    startTime = time.time()
    frameCount = 0
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
        imageData = image.getData()
        Pin.append(imageData[0])
        
        # Add frame to AVI
        avi.append(image)
        
        # Print output
        if time.time()-camTimer > 10:    
           print('[' + str(int((time.time()-startTime)/60)) + ':' + str(int(np.mod(time.time()-startTime, 60))) + '] Recorded %s frames with a frame rate of '%frameCount + str(round(frameCount/(time.time()-startTime), 1)))
           camTimer = time.time()
        frameCount = frameCount + 1

        if (imageData[0] == 48) & (time.time()-pulseTimer > 2):
            print('[' + str(int((time.time()-startTime)/60)) + ':' + str(int(np.mod(time.time()-startTime, 60))) + '] Received TTL pulse from Bpod')
            pulseTimer = time.time()

except Exception as e:
    # Save data and error
    np.save('TimeStamps_' + camID, TS)
    np.save('SyncPulses_' + camID, Pin)
    errorFile = open('ErrorLog_' + camID + '.txt', 'w')        
    errorFile.write(str(e))
    errorFile.close()
    print('Camera ' + camID + ' has crashed')
    print('Error message: ' + str(e))
    print(cam.isConnected)    

    # Disconnect cameras
    if cam.isConnected == True:  
        cam.stopCapture()
        cam.disconnect()
    time.sleep(10)

# Save timestamps
np.save('TimeStamps_' + camID, TS)
np.save('SyncPulses_' + camID, Pin)

# Disconnect cameras
cam.stopCapture()
cam.disconnect()
print('Camera ' + camID + ' is done')
time.sleep(10)




