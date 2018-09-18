"""
Configure all connected cameras and save to camera memory
"""
import PySpin

def CaptureSettings():
    savePath = '/media/angelakilab/64aefdd9-efac-4b64-b86b-bb77547d1967/test' #path to video storage disc  
    frameRates = [150]*10 #frame rates in Hz
    duration = 5 #Duration of the recording in sec
    return savePath, frameRates, duration
    
def ConfigureCameras():

   # Retrieve singleton reference to system object
   system = PySpin.System.GetInstance()

   # Retrieve list of cameras from the system
   cam_list = system.GetCameras()
   num_cameras = cam_list.GetSize()

   print('Number of cameras detected: ', num_cameras)
   for i, cam in enumerate(cam_list):
       nodemap_tldevice = cam.GetTLDeviceNodeMap()
       cam.Init()
       nodemap = cam.GetNodeMap()
       node_serial = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))
       device_serial_number = node_serial.GetValue()
       print('Configuring camera ', str(device_serial_number))

       '''
       trigger mode to be activated here for TTL pulse
 
       NOT IMPLEMENTED
       '''
       cam.DeInit()       
       del cam


   cam_list.Clear()
   system.ReleaseInstance()
   return num_cameras
