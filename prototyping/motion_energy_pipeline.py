import cv2 
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import alf.io
from oneibl.one import ONE
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import os
import time


'''
For a session where there is DLC already computed,
load DLC traces to cut video ROIs and then
compute motion energy for these ROIS.

bodyCamera: cut ROI such that mouse body
            but not wheel motion is in ROI
            
left(right)Camera: cut whisker pad region
'''


def get_dlc_XYs(eid, video_type):

    #video_type = 'left'    
    one = ONE() 
    dataset_types = ['camera.dlc',                     
                     'camera.times',
                     'trials.intervals']
                     
    a = one.list(eid,'dataset-types')   
    if not all([x['dataset_type'] for x in a]):
        print('not all data available')    
        return   
    
                 
    one.load(eid, dataset_types = dataset_types)
    local_path = one.path_from_eid(eid)  
    alf_path = local_path / 'alf'   
    
    cam0 = alf.io.load_object(
        alf_path,
        '%sCamera' %
        video_type,
        namespace='ibl')

    Times = cam0['times']

    cam = cam0['dlc']
    points = np.unique(['_'.join(x.split('_')[:-1]) for x in cam.keys()])


    # Set values to nan if likelyhood is too low # for pqt: .to_numpy()
    XYs = {}
    for point in points:
        x = np.ma.masked_where(
            cam[point + '_likelihood'] < 0.9, cam[point + '_x'])
        x = x.filled(np.nan)
        y = np.ma.masked_where(
            cam[point + '_likelihood'] < 0.9, cam[point + '_y'])
        y = y.filled(np.nan)
        XYs[point] = np.array(
            [x, y])    

    return Times, XYs  


def run_command(command):
    """
    Runs a shell command using subprocess.

    :param command: command to run
    :return: dictionary with keys: process, stdout, stderr
    """
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    info, error = process.communicate()
    return {
        'process': process,
        'stdout': info.decode(),
        'stderr': error.decode()}


def motion_energy(video_path):
    '''
    make a motion energy video (subtracting subsequent frames), 
    save the average across pixels
    '''
    
    make_video = True
    
    cap = cv2.VideoCapture(video_path)
    total_frames = cap.get(7) 
    size = (int(cap.get(3)), int(cap.get(4)))
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    D = np.zeros(frameCount) 
    if make_video:  
        output_path = video_path.replace('.mp4', '_ME.mp4')
        out = cv2.VideoWriter(output_path,cv2.VideoWriter_fourcc(*'mp4v'), 30.0, size)
    ret0, frame0 = cap.read()
    # turn into grayscale to avoid RGB artefacts
    frame0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    k = 0
    while cap.isOpened():     
        ret1, frame1 = cap.read()     
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)   
        if (ret0==True) and (ret1==True):
            difference = cv2.absdiff(frame0, frame1) 
            D[k] = difference.median()
            k +=1
            
            if make_video: 
                out.write(difference)
            frame0 = frame1 
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
    
    return D 


def get_mean_positions(XYs):
    mloc = {} # mean locations
    for point in XYs:
        mloc[point] = [int(np.nanmean(XYs[point][0])), int(np.nanmean(XYs[point][1]))]
    return mloc


def cut_whisker(file_in,XYs):

    file_out = file_in.replace('.mp4', '.whiskers.mp4')                
    
    mloc = get_mean_positions(XYs)
    p_nose = np.array(mloc['nose_tip'])
    p_pupil = np.array(mloc['pupil_top_r'])    

    # heuristic to find whisker area in side videos:
    # square with side length half the distance 
    # between nose and pupil and anchored on midpoint

    p_anchor = np.mean([p_nose,p_pupil],axis=0)
    squared_dist = np.sum((p_nose-p_pupil)**2, axis=0)
    dist = np.sqrt(squared_dist)
    whxy = [dist/2, dist/3, p_anchor[0] - dist/4, p_anchor[1]]

    crop_command = (
        'ffmpeg -nostats -y -loglevel 0  -i {file_in} -vf "crop={w[0]}:{w[1]}:'
        '{w[2]}:{w[3]}" -c:v libx264 -crf 11 -c:a copy {file_out}')
    pop = run_command(crop_command.format(file_in=file_in, file_out=file_out, w=whxy))

    return file_out, whxy



def cut_body(file_in,XYs):
    '''
    cut body without tail and wheel
    '''

    file_out = file_in.replace('.mp4', '.body_core.mp4')                    
    mloc = get_mean_positions(XYs)
    p_anchor = np.array(mloc['tail_start']) 
    dist = p_anchor[0] 
    whxy = [dist*3/5, 210, p_anchor[0] - dist*3/5, p_anchor[1] - 120]

    crop_command = (
        'ffmpeg -nostats -y -loglevel 0  -i {file_in} -vf "crop={w[0]}:{w[1]}:'
        '{w[2]}:{w[3]}" -c:v libx264 -crf 11 -c:a copy {file_out}')
    pop = run_command(crop_command.format(file_in=file_in, file_out=file_out, w=whxy))

    return file_out, whxy



def compute_ROI_ME(eid):
    start_T = time.time() 
    
    one = ONE()
    
    dataset_types = ['camera.dlc',                     
                     'camera.times',
                     '_iblrig_Camera.raw']                     
                     
    a = one.list(eid,'dataset-types')   
    if not all([x['dataset_type'] for x in a]):
        print('not all data available')    
        return    

    # download all three raw videos             
    one.load(eid, dataset_types)
    video_data = one.path_from_eid(eid) / 'raw_video_data'
    
    for video_type in ['body']:#['left','right',
        # get DLC for anchor points
        video_path = video_data / str('_iblrig_%sCamera.raw.mp4' % video_type)
        if not os.path.isfile(video_path):
            print(f'no {video_type} found at {video_path}')
            continue

        _, XYs = get_dlc_XYs(eid, video_type)
        
        # compute results
        if video_type == 'body':       
            file_out, whxy = cut_body(str(video_path),XYs)
        else:
            file_out, whxy = cut_whisker(str(video_path),XYs)
                   
        D = motion_energy(file_out)    
                
        # save ROI location
        p0 =  f'{video_type}ROIMotionEnergy.position.npy'
        np.save(Path(video_path).parent / p0, whxy)

        # save ME
        p1 =  f'{video_type}.ROIMotionEnergy.npy'    
        np.save(Path(video_path).parent / p1, D)    

        #os.remove(file_out)            
        print(eid, video_type, 'done')  
                    
    end_T = time.time() 
    print(eid, 'done in', np.round((end_T - start_T),2))
    
    
