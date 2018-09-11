import PyCapture2
frameRate = 50

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
   cam.setEmbeddedImageInfo(timestamp = False, GPIOPinState = True, gain = False, shutter = False, brightness = False, exposure = False, whiteBalance = False, frameCounter = False, strobePattern = False)
   cam.setGPIOPinDirection(pin = 3, direction = 1)
   cam.setProperty(type = 16, onOff = True, autoManualMode = False, absValue = frameRate)
   cam.setTriggerMode(onOff = False)
   cam.setGPIOPinDirection(3, 0)
   #cam.saveToMemoryChannel(1)
   cam.disconnect()
