from pylab import *
import os, fnmatch,pandas,deeplabcut 
from datetime import datetime
ion()
from shutil import copyfile

def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result

#for i in find('*.mp4', '/home/mic/Downloads/FlatIron'):
# copyfile(i,'/home/mic/DLC/videos/'+i.split('/')[-1])

def Get_short_sample_vids(vid_folder):

 '''
 Cut from all videos a 15 sec sample (processing speed: 200 fps)
 '''
 if vid_folder[-1]!='/':
  print('the last character of vid-folder string must be /')
  return
 #vid_folder='/home/mic/DLC/videos/'

 if not os.path.exists(vid_folder+'short_samples/'):
  os.makedirs(vid_folder+'short_samples/')

 for vid in [v for v in os.listdir(vid_folder) if v.endswith('.mp4')]:
#  os.system('ffmpeg -i %s -c:v libx264 -crf 23 -ss 00:00:00 -t 00:00:15 -c:a copy %s' %(vid_folder+vid,vid_folder+'short_samples/'+vid))

  os.system('ffmpeg -i %s -c:v copy -ss 00:00:00 -t 00:00:15 -c:a copy %s' %(vid_folder+vid,vid_folder+'short_samples/'+vid))

 print('created short sample videos in %s' %(vid_folder+'short_samples/'))


def DLC_vids_coarsly(vid_folder):

 '''
 apply coarse DLC to side video, 20 fsp for 1280x1024 (70 frames/sec for vids 640x512)
 '''

 if vid_folder[-1]!='/':
  print('the last character of vid-folder string must be /')
  return

 short_vids=[vid_folder+'short_samples/'+vid for vid in os.listdir(vid_folder+'short_samples/')]

 config_path='/home/mic/DLC/trainingRig-mic-2019-02-11/config.yaml'
 deeplabcut.analyze_videos(config_path,short_vids)


def get_coordinates_of_average_pivot(ROI,vid_folder,h5):

 '''
 h5 is local file name only; get average position of a pivot point, e.g. eye center, for autocropping
 '''

 #choose parts to find pivot point which is used to crop around a ROI

 Q={'eye':['pupil_top_r','pupil_left_r','pupil_bottom_r','pupil_right_r'],
 'nostril':['nose_tip'],
 'tongue':['tube_top','tube_bottom'],
 'paws':['nose_tip']}

 
 parts=Q[ROI]


 df=pandas.read_hdf(vid_folder+h5)

 XYs=[]
 for part in parts:
  x_values=df[(df.keys()[0][0], part, 'x')].values
  y_values=df[(df.keys()[0][0], part, 'y')].values
  likelyhoods=df[(df.keys()[0][0], part, 'likelihood')].values
  
  mx = ma.masked_where(likelyhoods<0.9, x_values)
  x=ma.compressed(mx)
  my = ma.masked_where(likelyhoods<0.9, y_values)
  y=ma.compressed(my)

  XYs.append([nanmean(x),nanmean(y)])

 return mean(XYs,axis=0)


def crop_video(ROI,XY,vid,vid_folder,cropped_vid_folder):
 '''
 vid local file name only
 '''
 
 if vid_folder[-1]!='/':
  print('the last character of vid-folder string must be /')
  return

 P={'eye': 'ffmpeg -i %s -vf "crop=%s:%s:%s:%s" -c:v libx264 -crf 17 -c:a copy %s' %(vid_folder+vid,100,100,XY[0]-50,XY[1]-50,cropped_vid_folder+vid[:-4]+'_eye.mp4'),
 'nostril':'ffmpeg -i %s -vf "crop=%s:%s:%s:%s" -c:v libx264 -crf 17 -c:a copy %s' %(vid_folder+vid,100,100,XY[0]-10,XY[1]-40,cropped_vid_folder+vid[:-4]+'_nostril.mp4'),
 'tongue':'ffmpeg -i %s -vf "crop=%s:%s:%s:%s" -ss 00:00:00 -t 00:10:00 -c:v libx264 -crf 17 -c:a copy %s' %(vid_folder+vid,160,160,XY[0]-60,XY[1]-100,cropped_vid_folder+vid[:-4]+'_tongue.mp4'),
 'paws':'ffmpeg -i %s -vf "crop=%s:%s:%s:%s" -c:v libx264 -crf 23 -c:a copy %s' %(vid_folder+vid,900,750,XY[0],XY[1]-100,cropped_vid_folder+vid[:-4]+'_paws.mp4')}

 os.system(P[ROI])


def for_all_videos_crop(vid_folder):
 '''
 1600 fps 
 ''' 
 if vid_folder[-1]!='/':
  print('the last character of vid-folder string must be /')
  return

 ROIs=['eye','nostril','tongue','paws'] 
 
 for ROI in ROIs:
  #vid_folder='/home/mic/DLC/videos/'
  cropped_vid_folder=vid_folder+ROI+'/'

  if not os.path.exists(vid_folder+ROI+'/'):
   os.makedirs(vid_folder+ROI+'/')

  vids=[v for v in os.listdir(vid_folder) if v.endswith('.mp4')]
  #vids=[v for v in os.listdir(vid_folder) if v.endswith('.avi')]

  h5s=[v for v in os.listdir(vid_folder+'short_samples/') if v.endswith('.h5')] 

  d={}
  for vid in vids:
   d[vid]=[x for x in h5s if vid[:-4] in x][0]

  for vid in d:
   print(vid)
   XY=get_coordinates_of_average_pivot(ROI,vid_folder+'short_samples/',d[vid])
   crop_video(ROI,XY,vid,vid_folder,cropped_vid_folder)

  # for paws consider spatial downsampling after cropping in order to speed up processing (with 900x750 DLC is 30 fps while downsampled it is 120 fsp! Downsizing the video runs at 500 fsp)):
  if ROI=='paws':
    cropped_vids=[v for v in os.listdir(cropped_vid_folder) if v.endswith('.mp4')] 
    for vid in cropped_vids:
 #'ffmpeg -i %s -vf scale=450:375 -c:v libx264 -crf 23 -c:a copy %s' %(vid_folder+vid,cropped_vid_folder+vid[:-4]+'_paws.mp4')
     os.system('ffmpeg -i %s -vf scale=450:374 -c:v libx264 -crf 23 -c:a copy %s' %(cropped_vid_folder+vid,cropped_vid_folder+vid[:-4]+'_small.mp4'))
     os.remove(cropped_vid_folder+vid)

def DLC_ROIs(vid_folder):

 '''
 520 fps
 '''
 
 NNs={'eye':'/home/mic/DLC/eye-mic-2019-04-16/config.yaml',
 'nostril':'/home/mic/DLC/nostril-mic-2019-04-22/config.yaml',
 'paws':'/home/mic/DLC/paws-mic-2019-04-26/config.yaml',
 'tongue':'/home/mic/DLC/tongue-mic-2019-04-26/config.yaml'}

 for ROI in NNs: 

  cropped_vid_folder=vid_folder+ROI+'/'
  vids=[cropped_vid_folder+v for v in os.listdir(cropped_vid_folder) if v.endswith('.mp4')]
  #vids=[cropped_vid_folder+v for v in os.listdir(cropped_vid_folder) if v.endswith('.avi')]

  #choose network weights according to ROI 
  config_path=NNs[ROI]
  deeplabcut.analyze_videos(config_path,vids)

def get_time_series(vid_folder):
 '''
 vid_folder is path to folder with videos (with / at end)
 total processing time with F being total number of all frames of all videos to be processed
 and f is all frames worth 15 sec (short samples for cropping):  
 (f/200+f/20+F/1600+F/520) sec
 '''
 #vid_folder='/home/mic/DLC/videos/'
  
 print('getting short samples')
 Get_short_sample_vids(vid_folder) # f/200 [sec]
 print('DLC coarsely on short samples')
 DLC_vids_coarsly(vid_folder) # f/20 [sec]
 print('for all videos and ROIs crop')
 for_all_videos_crop(vid_folder) # F/1600 [sec]
 print('DLC_ROIs')
 DLC_eye(vid_folder) # F/520 [sec]
 




