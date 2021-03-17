'''
For a session where there is DLC already computed,
load DLC traces to cut video ROIs and then
compute motion energy for these ROIS.

bodyCamera: cut ROI such that mouse body but not wheel motion is in ROI

left(right)Camera: cut whisker pad region
'''

import os
import time
import numpy as np
import cv2

from oneibl.one import ONE
import alf.io
from ibllib.io.video import get_video_frames_preload, url_from_eid


def get_dlc_midpoints(eid, label, one=None):

    if one is None:
        one = ONE()

    # Download dlc data if not available locally
    _ = one.load(eid, dataset_types='camera.dlc', download_only=True)
    alf_path = one.path_from_eid(eid).joinpath('alf')

    dlc_df = alf.io.load_object(alf_path, f'{label}Camera', namespace='ibl')['dlc']

    # Set values to nan if likelihood is too low and calcualte midpoints
    targets = np.unique(['_'.join(col.split('_')[:-1]) for col in dlc_df.columns])
    mloc = {}
    for t in targets:
        idx = dlc_df.loc[dlc_df[f'{t}_likelihood'] < 0.9].index
        dlc_df.loc[idx, [f'{t}_x', f'{t}_y']] = np.nan
        mloc[t] = [int(np.nanmean(dlc_df[f'{t}_x'])), int(np.nanmean(dlc_df[f'{t}_y']))]

    return mloc


def motion_energy(eid, label, frame_numbers=None, one=None):
    '''
    Compute motion energy on cropped frames of a single video

    :param eid: Session ID
    :param label: 'body', 'left' or 'right'
    '''

    if one is None:
        one = ONE()

    start_T = time.time()
    video_path = one.path_from_eid(eid).joinpath('raw_video_data',
                                                 f'_iblrig_{label}Camera.raw.mp4')
    # Check if video available locally, else create url
    if not os.path.isfile(video_path):
        video_path = url_from_eid(eid, label=label, one=one)

    # Crop ROI
    mloc = get_dlc_midpoints(eid, label, one=one)
    if label == 'body':
        anchor = np.array(mloc['tail_start'])
        w, h = int(anchor[0] * 3 / 5), 210
        x, y = int(anchor[0] - anchor[0] * 3 / 5), int(anchor[1] - 120)
    else:
        anchor = np.mean([mloc['nose_tip'], mloc['pupil_top_r']], axis=0)
        dist = np.sqrt(np.sum((np.array(mloc['nose_tip']) - np.array(mloc['pupil_top_r']))**2,
                       axis=0))
        w, h = int(dist / 2), int(dist / 3)
        x, y = int(anchor[0] - dist / 4), int(anchor[1])

    # Note that x and y are flipped when loading with cv2, therefore:
    mask = np.s_[y:y + h, x:x + w]

    # Crop and grayscale frames.
    cropped_frames = get_video_frames_preload(video_path, frame_numbers=frame_numbers, mask=mask,
                                              func=cv2.cvtColor, code=cv2.COLOR_BGR2GRAY)

    # save ROI
    alf_path = one.path_from_eid(eid).joinpath('alf')
    np.save(alf_path.joinpath(f'{label}ROIMotionEnergy.position.npy'), np.asarray([w, h, x, y]))

    # Compute and save motion energy
    # Cast to float
    cropped_frames = np.asarray(cropped_frames, dtype=np.float32)
    me = np.mean(np.abs(cropped_frames[1:] - cropped_frames[:-1]), axis=(1, 2))
    # copy last value to make motion energy fit frame length
    me = np.append(me, me[-1])

    # save ME
    np.save(alf_path.joinpath(f'{label}.ROIMotionEnergy.npy'), me)

    end_T = time.time()
    print(f'{label}Camera computed in', np.round((end_T - start_T), 2))
