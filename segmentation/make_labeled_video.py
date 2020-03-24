import numpy as np
import alf.io
import matplotlib.pyplot as plt
import ibllib.plots as iblplt
from oneibl.one import ONE
from pathlib import Path
import cv2
import csv
from scipy.signal import resample

trial_range = [0,1000]

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

alf_path = Path('/home/mic/transfer_DLC/witten_right/alf')
eid = 'short_test'

video_data = alf_path.parent / 'raw_video_data'
video_path = video_data / '_iblrig_rightCamera.raw.mp4'

# that gives cam time stamps and DLC output
cam_right = alf.io.load_object(alf_path, '_ibl_rightCamera')

# set where to read and save video and get video info
cap = cv2.VideoCapture(video_path.as_uri())
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
size = (int(cap.get(3)), int(cap.get(4)))
out = cv2.VideoWriter('vid_%s_trials_%s_%s.mp4' %(eid, trial_range[0], trial_range[-1]),cv2.VideoWriter_fourcc(*'mp4v'), fps, size)# put , 0 if grey scale
#assert length < len(cam_right['times']), '#frames > #stamps'


#trial_range = [50,60] # pick trial range for which to display stuff

#trials = alf.io.load_object(alf_path, '_ibl_trials')
#num_trials = len(trials['intervals'])
#if trial_range[-1] > num_trials - 1: print('There are only %s trials' %num_trials)

#frame_start = find_nearest(cam_right['times'], [trials['intervals'][trial_range[0]][0]]) 
#frame_stop = find_nearest(cam_right['times'], [trials['intervals'][trial_range[-1]][1]])

frame_start = 0
frame_stop = 1000

#############
# Get average point of fingers for each paw
#############

fingers2 = ['paw_l','paw_r']
# Set values to nan if likelyhood is too low
XYs = {}
for part in fingers2:
    x = np.ma.masked_where(cam_right[part+'_likelihood'] < 0.9, cam_right[part+'_x'])
    x = x.filled(np.nan)
    y = np.ma.masked_where(cam_right[part+'_likelihood'] < 0.9, cam_right[part+'_y'])
    y = y.filled(np.nan)
    XYs[part] = np.array([x[frame_start:frame_stop], y[frame_start:frame_stop]])


# writing stuff on frames
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2

# Region of interest cropping?
# ROI = frame[y1:y2, x1:x2]

#set start frame
cap.set(1,frame_start)

dot_s = 3 # [px] for painting DLC dots 
r = [0,0,255]
g = [0,255,0]
block = np.ones((2*dot_s, 2*dot_s, 3))
cd = {'paw_l':r, 'paw_r':g}


k = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   
    gray = frame
    
#    cv2.putText(gray,'Wheel angle: ' + str(round(wheel_pos[k],2)), 
#        bottomLeftCornerOfText, 
#        font, 
#        fontScale,
#        fontColor,
#        lineType)

    for part in fingers2:
        X0 = XYs[part][0][k]
        Y0 = XYs[part][1][k] 
        # transform for opencv?
        X = Y0
        Y = X0
        if not np.isnan(X) and not np.isnan(Y):
            X = X.astype(int)
            Y = Y.astype(int)
            gray[X - dot_s:X + dot_s, Y - dot_s:Y + dot_s] = block * cd[part]    

    out.write(gray)
    cv2.imshow('frame',gray)
    cv2.waitKey(1)
    k += 1
    if k == (frame_stop - frame_start) - 1: break

out.release()
cap.release()
cv2.destroyAllWindows()



