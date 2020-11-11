import os, fnmatch
from pylab import *
import cv2
import numpy as np
from shutil import copyfile
from oneibl.one import ONE
one = ONE()
ion()

def get_latest_n_videos_from_all_labs(n):

 #n=4 #number of most recent videos to download 
 
 for lab in one.list(keyword='lab'):#['mainenlab']:# 
  
  sessions, details = one.search(dataset_types='_iblrig_leftCamera.raw', lab=lab, 
  details=True,date_range=['2019-05-20', '2030-04-10'])

  try:
   sessions_with_vids=[]
   for i in range(len(sessions)): #check if there are downloadable videos
     
    # just get "trained" mice:    
    if (details[i]['n_correct_trials']/details[i]['n_trials'])>0.65:    
     urls=[d['data_url'] for d in details[i]['data_dataset_session_related']]
     urls_=[x for x in urls if x is not None]
     vids=[x for x in urls_ if '_iblrig_leftCamera.raw' in x]
     if vids!=[]:
      sessions_with_vids.append([sessions[i],details[i]['end_time']])
    
   print('%s vids to download for %s' %(len(sessions_with_vids),lab))
   #download only the n most recent videos
   s=list(reversed(sorted(sessions_with_vids,key=lambda x: x[1])))
   print(sessions_with_vids[:n])

   for j in s[:n]:
    one.load(j[0],dataset_types='_iblrig_leftCamera.raw')   
   
  except:
   print('no vids for %s' %(lab))


def Find(pattern, path):
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
 cap.set(1,frameCount/2)
 ret, frame = cap.read()
 cap.release()

 return frame

def plot_and_save_example_frames():
 '''
 save individual images
 '''


 flatiron='/home/mic/Downloads/FlatIron'

 nrows=ceil((len(Find('*.mp4', flatiron)))**0.5)
 ncols=floor((len(Find('*.mp4', flatiron)))**0.5)


 if not os.path.exists(flatiron+'/example_frames/'):
  os.makedirs(flatiron+'/example_frames/')

 for vid in Find('*.mp4', flatiron):
  fig=figure(figsize=(14,10))
  imshow(get_example_frame(vid))
  title(vid.split('/')[5]+', '+vid.split('/')[7]+', \n '+vid.split('/')[8]+', '+vid.split('/')[9])
  axis('off')
  plt.tight_layout()
  fig.savefig(flatiron+'/example_frames/%s.png' %(vid.split('/')[5]+', '+vid.split('/')[7]+', \n '+vid.split('/')[8]+', '+vid.split('/')[9]))


def tile_example_frames():

 '''
 plot a tile of images
 '''

 flatiron='/home/mic/Downloads/FlatIron'

 nrows=ceil((len(Find('*.mp4', flatiron)))**0.5)
 ncols=ceil((len(Find('*.mp4', flatiron)))**0.5)

 fig=figure(figsize=(14,10))
 
 k=1
 for vid in Find('*.mp4', flatiron):
  ax=subplot(nrows,ncols,k)
  imshow(get_example_frame(vid))
  title(vid.split('/')[5]+', '+vid.split('/')[7]+', \n '+vid.split('/')[8]+', '+vid.split('/')[9])
  axis('off')
  plt.tight_layout()
  k+=1
