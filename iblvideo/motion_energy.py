'''
For a session where there is DLC already computed,
load DLC traces to cut video ROIs and then
compute motion energy for these ROIS.

bodyCamera: cut ROI such that mouse body but not wheel motion is in ROI

left(right)Camera: cut whisker pad region
'''

import os
import time
import logging
from pathlib import Path
import numpy as np
import cv2

from oneibl.one import ONE
import alf.io
_logger = logging.getLogger('ibllib')


def get_dlc_midpoints(eid, side, one=None):

    if one is None:
        one = ONE()

    _ = one.load(eid, dataset_types='camera.dlc')
    local_path = one.path_from_eid(eid)
    alf_path = Path(local_path).joinpath('alf')

    dlc = alf.io.load_object(alf_path, f'{side}Camera', namespace='ibl')['dlc']

    # Set values to nan if likelihood is too low and calcualte midpoints
    targets = np.unique(['_'.join(col.split('_')[:-1]) for col in dlc.columns])
    mloc = {}
    for t in targets:
        idx = dlc.loc[dlc[f'{t}_likelihood'] < 0.9].index
        dlc.loc[idx, [f'{t}_x', f'{t}_y']] = np.nan
        mloc[t] = [int(np.nanmean(dlc[f'{t}_x'])), int(np.nanmean(dlc[f'{t}_y']))]

    return mloc


def crop_body(eid):
    '''
    Cut body without tail and wheel
    '''
    mloc = get_dlc_midpoints(eid, 'body')
    anchor = np.array(mloc['tail_start'])
    whxy = [int(anchor[0] * 3 / 5),  # width of output rectangle
            210,                # height of output rectangle
            int(anchor[0] - anchor[0] * 3 / 5),  # x position of upper left corner of output rectangle
            int(anchor[1] - 120)]  # y position of upper left corner of output rectangle
    # CROP AND STREAM
    # file_out = file_in.replace('.mp4', '.body_core.mp4')
    _logger.info(f'cropping ROI done, {file_in}')

    return file_out, whxy



def compute_ROI_ME(video_path, dlc):
    '''
    Compute ROI motion energy on a single video

    :param video_path: path to IBL video
    :param video_type: 'body', 'left' or 'right'
    :param XYs: dlc traces for this video

    '''
    start_T = time.time()
    side = [x for x in ['left', 'right', 'body'] if x in video_path.stem][0]

    if not os.path.isfile(video_path):
        print(f'{video_path} not found')
        return

    # Crop ROI
    _logger.info('{side}Camera Cropping ROI')
    if side == 'body':
        file_out, whxy = cut_body(video_path, dlc)
    else:
        file_out, whxy = cut_whisker(video_path, dlc)

    # save ROI
    roi_file = f'{side}ROIMotionEnergy.position.npy'
    np.save(Path(video_path).parent.joinpath(roi_file), whxy)

    # TODO: Check if
    _logger.info(f'{side}Camera computing motion energy')
    me = motion_energy(file_out)

    # save ME
    me_file = f'{side}.ROIMotionEnergy.npy'
    np.save(Path(video_path).parent.joinpath(me_file), me)

    # remove cropped video
    os.remove(file_out)
    _logger.info(f'Motion energy and ROI for {side}Camera computed and saved')

    end_T = time.time()
    print(f'{side}Camera computed in', np.round((end_T - start_T),2))
