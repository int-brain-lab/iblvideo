from __future__ import print_function
import urllib
import bz2
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import matplotlib.image as mpimg
import cv2
import plotly.graph_objs as go
import os
from utils.utils_IO import ordered_arr_3d_to_dict, refill_nan_array, arr_2d_to_list_of_dicts, read_image, make_image_array, revert_ordered_arr_2d_to_dict, save_object, write_video
from utils.utils_plotting import plot_image_labels, plot_3d_points, vector_plot, draw_circles, slope, drawLine, skew, plot_cams_and_points
from utils.utils_BA import fun, bundle_adjustment_sparsity, project
from anipose_BA import CameraGroup, Camera
from scipy.spatial.transform import Rotation as R

import alf.io
from oneibl.one import ONE
from pathlib import Path

import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import matplotlib.patches as mpatches

'''
adapted from Sunand Raghupathi, Paninski Lab, 51N84D/3D-Animal-Pose; 
this includes bundle adjustement taken from 
https://github.com/lambdaloop/aniposelib/blob/master/aniposelib/cameras.py
'''


'''
P_{X,Y}_{TOP,BOT}: (int) - {width / 2, height / 2} for camera {1,2}

-->For setting the offset terms in the camera matrix to the center of the image plane

pts_array_2d: (np.array) - Array of shape (num_cameras, num_points, 2) containing set of 2d points for each camera. This should be after cleaning NaNs i.e. removing rows with NaNs

info_dict: Dictionary with keys {'num_frames', 'num_analyzed_body_parts', 'num_cameras', 'num_points_all', 'clean_point_indices'}

--> 'num_frames' is the number of frames in the video

--> 'num_analyzed_body_parts' is the number of body parts / joints being modeled (i.e. one per keypoint)

--> 'num_cameras' is the number of cameras. In our case, it is 2

--> 'num_points_all' is the original number of points (including NaNs)

--> 'clean_point_indices' is a list of indices (with length = num_points in pts_array_2d) pointing to the clean (non-NaN) entries in the original data

path_images: (list) - List of sublists. Each sublist (one per camera / view) contains absolute paths to image frames.
'''



'''
Get IBL 2D points for a given trial
'''
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def GetXYs(eid, video_type, trial_range):
    '''
    eid: session id, e.g. '3663d82b-f197-4e8b-b299-7b803a155b84'
    video_type: one of 'left', 'right', 'body'
    trial_range: first and last trial number of range to be shown, e.g. [5,7]
    '''
 
    one = ONE()
    dataset_types = ['camera.times',
                     'trials.intervals',
                     'camera.dlc']

    a = one.list(eid, 'dataset-types')

    assert all([i in a for i in dataset_types]
               ), 'For this eid, not all data available'

    D = one.load(eid, dataset_types=dataset_types, dclass_output=True)
    alf_path = Path(D.local_path[0]).parent.parent / 'alf'

    video_data = alf_path.parent / 'raw_video_data'     
    video_path = list(video_data.rglob('_iblrig_%sCamera.raw.*' % video_type))[0] 
    print(video_path) 

    # that gives cam time stamps and DLC output (change to alf_path eventually)
    
    cam1 = alf.io.load_object(video_path.parent, '_ibl_%sCamera' % video_type)     
    try:
        cam0 = alf.io.load_object(alf_path, '_ibl_%sCamera' % video_type)          
    except:
        cam0 = {}    
    cam = {**cam0,**cam1}

    # just to read in times for newer data (which has DLC results in pqt format
    #cam = alf.io.load_object(alf_path, '_ibl_%sCamera' % video_type)

    # pick trial range for which to display stuff
    trials = alf.io.load_object(alf_path, '_ibl_trials')
    num_trials = len(trials['intervals'])
    if trial_range[-1] > num_trials - 1:
        print('There are only %s trials' % num_trials)

    frame_start = find_nearest(cam['times'],
                               [trials['intervals'][trial_range[0]][0]])
    frame_stop = find_nearest(cam['times'],
                              [trials['intervals'][trial_range[-1]][1]])

    '''
    DLC related stuff
    '''
    Times = cam['times'][frame_start:frame_stop] 
    del cam['times']      

#    dlc_name = '_ibl_%sCamera.dlc.pqt' % video_type
#    dlc_path = alf_path / dlc_name
#    cam=pd.read_parquet(dlc_path)    


    points = np.unique(['_'.join(x.split('_')[:-1]) for x in cam.keys()])
    

    if video_type != 'body':
        d = list(points) 
        d.remove('tube_top')
        d.remove('tube_bottom')   
        points = np.array(d)


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
            [x[frame_start:frame_stop], y[frame_start:frame_stop]])
            
    res_folder = '/home/mic/3D-Animal-Pose-master/IBL_example/%s_trials_%s_%s' %(eid, video_type, trial_range[0], trial_range[1])    
    
    Path(res_folder).mkdir(parents=True, exist_ok=True)    
            
    np.save('/home/mic/3D-Animal-Pose-master/IBL_example/%s_trials_%s_%s/XYs_%s.npy' %(eid, video_type, trial_range[0], trial_range[1]), XYs)
    np.save('/home/mic/3D-Animal-Pose-master/IBL_example/%s_trials_%s_%s/times_%s.npy' %(eid, video_type, trial_range[0], trial_range[1]), Times)
    #return XYs, Times


def get_3d_points_for_IBL_example():
    #bring IBL data in format for bundle_adjust
    # starting with one paw only, the one called left in video left
    XYs_left = np.load('/home/mic/3D-Animal-Pose-master/IBL_example/XYs_left.npy', allow_pickle=True).flatten()[0]
    XYs_right = np.load('/home/mic/3D-Animal-Pose-master/IBL_example/XYs_right.npy', allow_pickle=True).flatten()[0]

    times_left = np.load('/home/mic/3D-Animal-Pose-master/IBL_example/times_left.npy')
    times_right = np.load('/home/mic/3D-Animal-Pose-master/IBL_example/times_right.npy')

    # get closest stamps or right cam (150 Hz) for each stamp of left (60 Hz)
    idx_aligned = []
    for t in times_left:
        idx_aligned.append(find_nearest(times_right, t))
        
    # paw_l in video left = paw_r in video right
    # Divide left coordinates by 2 to get them in half resolution like right cam; 
    # reduce temporal resolution of right cam to that of left cam
    num_analyzed_body_parts = 3  # both paws and nose

    cam_right_paw1 = np.array([XYs_right['paw_r'][0][idx_aligned], XYs_right['paw_r'][1][idx_aligned]]) 
    cam_left_paw1 = np.array([XYs_left['paw_l'][0]/2,XYs_left['paw_l'][1]/2]) 

    cam_right_paw2 = np.array([XYs_right['paw_l'][0][idx_aligned], XYs_right['paw_l'][1][idx_aligned]]) 
    cam_left_paw2 = np.array([XYs_left['paw_r'][0]/2,XYs_left['paw_r'][1]/2]) 

    cam_right_nose = np.array([XYs_right['nose_tip'][0][idx_aligned], XYs_right['nose_tip'][1][idx_aligned]]) 
    cam_left_nose = np.array([XYs_left['nose_tip'][0]/2,XYs_left['nose_tip'][1]/2]) 

    # the format shall be such that points are concatenated, p1,p2,p3,p1,p2,p3, ... 
    cam1 = np.zeros((len(idx_aligned) * num_analyzed_body_parts, 2)) 
    cam1[0::3] = cam_right_paw1.T
    cam1[1::3] = cam_right_paw2.T
    cam1[2::3] = cam_right_nose.T

    cam2 = np.zeros((len(idx_aligned) * num_analyzed_body_parts, 2)) 
    cam2[0::3] = cam_left_paw1.T
    cam2[1::3] = cam_left_paw2.T
    cam2[2::3] = cam_left_nose.T

    pts_array_2d_with_nans = np.array([cam1,cam2])

    num_cameras, num_points_all, _ = pts_array_2d_with_nans.shape

    # remove nans (any of the x_r,y_r, x_l, y_l) and keep clean_point_indices
    non_nan_idc = ~np.isnan(pts_array_2d_with_nans).any(axis=2).any(axis=0)

    info_dict = {}
    info_dict['num_frames'] = len(times_left) 
    info_dict['num_cameras'] = num_cameras
    info_dict['num_analyzed_body_parts'] = num_analyzed_body_parts 
    info_dict['num_points_all'] = num_points_all
    info_dict['clean_point_indices'] = np.arange(num_points_all)[non_nan_idc]

    pts_array_2d = pts_array_2d_with_nans[:,info_dict['clean_point_indices']]

    IMG_WIDTH = 640
    IMG_HEIGHT = 512
    P_X_left = P_X_right = IMG_WIDTH // 2
    P_Y_left = P_Y_right = IMG_HEIGHT // 2
     

    # For IBL we give 3D coordinates in resolution of right camera (camera 1)
    # Thus offsets are the same, however the 2D points of the left cam must have been divided by 2 


    #--------CAMERA 1------------ (that's ibl_rightCamera)
    #Initialize camera 1
    camera_1 = Camera(rvec=[0,0,0], tvec=[0,0,0]) # right-handed coordinate system, e_z how cam 1 points; e_x perp to plane with both cams and target
    #Set offset
    camera_1.set_size((P_X_right, P_Y_right))

    cam1_init_params = np.abs(np.random.rand(8))
    #Set rotations [0:3] and translation [3:6] to 0
    cam1_init_params[0:6] = 0
    #Initialize focal length to image width
    cam1_init_params[6] = P_X_right * 2
    #Initialize distortion to 0
    cam1_init_params[7] = 0.0 
    #Set parameters
    camera_1.set_params(cam1_init_params)

    #--------CAMERA 2------------(that's ibl_leftCamera)
    #Set rotation vector w.r.t. camera 1
    # roration around y axis only, about 120 deg (2.0127 rad) from Guido's CAD
    rvec2 = np.array([0, 2.0127, 0])
    

    #Set translation vector w.r.t. camera 1, using CAD drawing [mm];
    # cameras are 292.8 mm apart; 
    #distance vector pointing from cam1 to the other cam: 
    tvec2 = [-15.664, 0, 24.738]
    #Initialize camera 2
    camera_2 = Camera(rvec=rvec2, tvec=tvec2)
    #Set offset 
    camera_1.set_size((P_X_left, P_Y_left))

    cam2_init_params = np.abs(np.random.rand(8))
    cam2_init_params[0:3] = rvec2
    cam2_init_params[3:6] = tvec2
    cam2_init_params[6] = P_X_left * 2
    cam2_init_params[7] = 0.0
    camera_2.set_params(cam2_init_params)

    #Group cameras
    cam_group = CameraGroup(cameras=[camera_1, camera_2])

    #Get error before Bundle Adjustment by triangulating using the initial parameters:
    f0, points_3d_init = cam_group.get_initial_error(pts_array_2d)
    print(points_3d_init.shape)

    #fig = plot_cams_and_points(cam_group=cam_group, points_3d=points_3d_init, title="3D Points Initialized")

    #Run Bundle Adjustment
    res, points_3d = cam_group.bundle_adjust(pts_array_2d)

    # do the pts_array_3d_clean
    array_3d_back = refill_nan_array(points_3d, 
                                   info_dict, 
                                   dimension = '3d')

    pts3d_dict = ordered_arr_3d_to_dict(array_3d_back, info_dict)
    
    L = []
    for i in range(num_analyzed_body_parts):
        r = np.array([pts3d_dict['x_coords'][:,i], pts3d_dict['y_coords'][:,i],pts3d_dict['z_coords'][:,i]]).T
        L.append(r)
    
    L = np.array(L)
    _, frames, _ = L.shape
    col_list = np.array([[0]*frames,[0.5]*frames,[1]*frames]).flatten()
    pts_3d = np.vstack(L)
                                                                      
    fig = plot_cams_and_points(cam_group=cam_group, points_3d=pts_3d, title="3D Points Bundle Adjusted", point_size = 10, color_list=col_list)
    
    np.save('/home/mic/3D-Animal-Pose-master/IBL_example/pts3d.npy', pts3d_dict)
    
    return pts3d_dict
















