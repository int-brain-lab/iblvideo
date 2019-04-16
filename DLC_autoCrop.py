from pylab import *
import pandas 
import os
import deeplabcut


def DLC_vids_coarsly(vid_folder):

 short_vids=[vid_folder+'short_samples/'+vid for vid in os.listdir(vid_folder+'short_samples/')]

 config_path='/home/mic/DLC/trainingRig-mic-2019-02-11/config.yaml'
 deeplabcut.analyze_videos(config_path,short_vids)


def Get_short_sample_vids(vid_folder):

 '''
 Cut from all videos a 10 sec sample, then run DLC-coarse on it
 '''

 #vid_folder='/home/mic/DLC/videos/'

 if not os.path.exists(vid_folder+'short_samples/'):
  os.makedirs(vid_folder+'short_samples/')

 for vid in os.listdir(vid_folder):
  os.system('ffmpeg -i %s -c:v libx264 -crf 23 -ss 00:00:00 -t 00:00:15 -c:a copy %s' %(vid_folder+vid,vid_folder+'short_samples/'+vid))

 print('created short sample videos in %s' %(vid_folder+'short_samples/'))


def get_coordinates_of_average_pupil(vid_folder,h5):

 '''
 h5 is local file name only
 '''

 parts=['pupil_top_r','pupil_left_r','pupil_bottom_r','pupil_right_r']

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


def crop_video(XY,vid,vid_folder,cropped_vid_folder):
 '''
 vid local file name only
 '''

 #crop square of 100x100 px, centered on the XY point (average pupil center) 
 os.system('ffmpeg -i %s -vf "crop=%s:%s:%s:%s" -c:v libx264 -crf 23 -c:a copy %s' %(vid_folder+vid,100,100,XY[0]-50,XY[1]-50,cropped_vid_folder+vid[:-4]+'_eye.mp4'))


def for_all_videos_crop_eye():
 '''
 this assumes DLC has already found
 ''' 
 vid_folder='/home/mic/DLC/videos/'
 cropped_vid_folder=vid_folder+'eye/'

 if not os.path.exists(vid_folder+'eye/'):
  os.makedirs(vid_folder+'eye/')

 vids=[v for v in os.listdir(vid_folder) if v.endswith('.mp4')]
 h5s=[v for v in os.listdir(vid_folder+'short_samples/') if v.endswith('.h5')] 

 d={}
 for vid in vids:
  d[vid]=[x for x in h5s if vid[:-4] in x][0]

 for vid in d:
  print(vid)
  XY=get_coordinates_of_average_pupil(vid_folder+'short_samples/',d[vid])
  crop_video(XY,vid,vid_folder,cropped_vid_folder)

