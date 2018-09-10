"""
Configure all connected cameras and save to camera memory
"""
import PyCapture2

def CaptureSettings():
    savePath = '/media/vid/547f640f-3dbc-4419-a8f2-e4e715b088ba/test'
    #savePath = '/home/vid/Videos'
    frameRates = [50,50,50]
    duration = 30
    return savePath, frameRates, duration
    
def ConfigureCameras():
   bus = PyCapture2.BusManager()
   numCams = bus.getNumOfCameras()
   print('Number of cameras detected: ', numCams)
   for j in range(numCams):
       cam = PyCapture2.Camera()
       cam.connect(bus.getCameraFromIndex(j))
       camInfo = cam.getCameraInfo()
       print('Configuring camera ', str(camInfo.serialNumber))
       #cam.restoreFromMemoryChannel(0)     
       #cam.setConfiguration(numBuffers = 1000, grabMode = 1)
       cam.setEmbeddedImageInfo(GPIOPinState = True, timestamp = False, gain = False, shutter = False, brightness = False, exposure = False, whiteBalance = False, frameCounter = False, strobePattern = False, ROIPosition = False)
       cam.setGPIOPinDirection(pin = 3, direction = 1)
       #cam.setProperty(type = 16, onOff = True, autoManualMode = False, absValue = frameRate)
       cam.setTriggerMode(onOff = False)
       cam.setGPIOPinDirection(3, 0)
       #cam.saveToMemoryChannel(1)
       cam.disconnect()
   return numCams

def CloseCameras(frameRate):
   bus = PyCapture2.BusManager()
   numCams = bus.getNumOfCameras()
   print('Number of cameras detected: ', numCams)
   for j in range(numCams):
       cam = PyCapture2.Camera()
       cam.connect(bus.getCameraFromIndex(j))
       camInfo = cam.getCameraInfo()
       cam.disconnect()
       print('Closed camera ', str(camInfo.serialNumber))
