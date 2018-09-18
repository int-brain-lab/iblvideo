"""
Capture from multiple cameras using PySpin
"""
import PySpin
from CameraFunctionsS import ConfigureCameras, CaptureSettings
import time
import os
import numpy as np
from sys import argv

# Path to save the video
savePath, frameRates, duration = CaptureSettings()
os.chdir(savePath)

# Settings
camIndex = int(argv[1])
frameRate = frameRates[camIndex]

system = PySpin.System.GetInstance()
cam_list = system.GetCameras()

#Check if camera is available
num_cameras = cam_list.GetSize()

if camIndex >= num_cameras:
    print('Camera not found')
    exit()

cam=cam_list[camIndex]
cam.Init()
nodemap = cam.GetNodeMap()
nodemap_tldevice = cam.GetTLDeviceNodeMap()
node_serial = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))
device_serial_number = node_serial.GetValue()

# Set acquisition mode to continuous
node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()
node_acquisition_mode.SetIntValue(acquisition_mode_continuous)

#set framerate
cam.AcquisitionFrameRate.SetValue(frameRate)

cam.BeginAcquisition() 
print('Started camera %s, ' %camIndex, 'Serial: %s' %device_serial_number, 'frameRate set: %s' %frameRate)
images = []
TS= []

startTime = time.time()

while (time.time()-startTime) < duration:
 image_result = cam.GetNextImage()
 images.append(image_result.Convert(PySpin.PixelFormat_Mono8, PySpin.HQ_LINEAR))
 TS.append(time.time()-startTime)
 image_result.Release()

cam.EndAcquisition()
cam.DeInit()
del cam
cam_list.Clear()
system.ReleaseInstance()

print('%s acquired in %s sec' %(len(images),duration))
avi_recorder = PySpin.SpinVideo()
avi_filename = '%s/%s.avi' %(savePath,device_serial_number)

option = PySpin.MJPGOption()
option.frameRate = frameRate 
option.quality = 75
avi_recorder.Open(avi_filename, option)
for image in images:
 avi_recorder.Append(image)
print('avi saved for cam %s, ' %camIndex, 'Serial: %s' %device_serial_number)
np.save('%s/%s_TS' %(savePath,device_serial_number),TS)


