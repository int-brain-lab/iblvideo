import os, fnmatch
from pylab import *
import cv2
import numpy as np
ion()

def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result

def get_example_frame(video_path):

 cap = cv2.VideoCapture(video_path)
 fps = cap.get(cv2.CAP_PROP_FPS) 
 frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
 duration = frameCount/fps
# print(video_path)
# print('fps = ' + str(fps))
# print('number of frames = ' + str(frameCount))
# print('duration (s) = ' + str(duration))
# minutes = int(duration/60)
# seconds = duration%60
# print('duration (min:s) = ' + str(minutes) + ':' + str(seconds))

 #get example frame in middle of video
 cap.set(1,frameCount/2)
 ret, frame = cap.read()
 cap.release()

 return frame

#get paths of all example videos
find('*.mp4', '/home/mic/Downloads/FlatIron')


def plot_example_frames():

 nrows=ceil((len(find('*.mp4', '/home/mic/Downloads/FlatIron')))**0.5)
 ncols=floor((len(find('*.mp4', '/home/mic/Downloads/FlatIron')))**0.5)

 fig=figure(figsize=(14,10))
 k=1
 for vid in find('*.mp4', '/home/mic/Downloads/FlatIron'):

  ax=subplot(nrows,ncols,k)  
  ax.imshow(get_example_frame(vid))
  title(vid.split('/')[5]+', '+vid.split('/')[7]+', \n '+vid.split('/')[8]+', '+vid.split('/')[9])
  ax.axis('off')
  k+=1



#Set grayscale colorspace for the frame. 
#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)



