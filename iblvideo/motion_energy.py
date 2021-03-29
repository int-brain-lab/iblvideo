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
import pandas as pd
import cv2

from oneibl.one import ONE
from ibllib.io.video import get_video_frames_preload, url_from_eid, label_from_path
from ibllib.io.extractors.camera import get_video_length


def grayscale(x):
    return cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)


def get_dlc_midpoints(dlc_pqt):
    # Load dataframe
    dlc_df = pd.read_parquet(dlc_pqt)
    # Set values to nan if likelihood is too low and calcualte midpoints
    targets = np.unique(['_'.join(col.split('_')[:-1]) for col in dlc_df.columns])
    mloc = {}
    for t in targets:
        idx = dlc_df.loc[dlc_df[f'{t}_likelihood'] < 0.9].index
        dlc_df.loc[idx, [f'{t}_x', f'{t}_y']] = np.nan
        mloc[t] = [int(np.nanmean(dlc_df[f'{t}_x'])), int(np.nanmean(dlc_df[f'{t}_y']))]
    return mloc


def motion_energy(session_path, dlc_pqt, frames=None, one=None):
    '''
    Compute motion energy on cropped frames of a single video

    :param session_path: Path to session
    :param label: 'body', 'left' or 'right'
    '''

    one = one or ONE()
    start_T = time.time()

    # Get label from dlc_df
    label = label_from_path(dlc_pqt)
    video_path = session_path.joinpath('raw_video_data', f'_iblrig_{label}Camera.raw.mp4')
    # Check if video available locally, else create url
    if not os.path.isfile(video_path):
        eid = one.eid_from_path(session_path)
        video_path = url_from_eid(eid, label=label, one=one)

    # Crop ROI
    mloc = get_dlc_midpoints(dlc_pqt)
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
    # save ROI coordinates
    roi = np.asarray([w, h, x, y])
    alf_path = session_path.joinpath('alf')
    roi_file = alf_path.joinpath(f'{label}ROIMotionEnergy.position.npy')
    np.save(roi_file, roi)

    if frames:
        # Find how many frames in the video
        frame_count = get_video_length(video_path)
        # Initiate some variables
        n, me, keep_reading = 0, np.empty(0), True
        while keep_reading:
            # Set the frame numbers to the next #frames, with 1 frame overlap
            frame_numbers = range(n * (frames - 1), n * (frames - 1) + frames)
            # Make sure not to load empty frames
            if max(frame_numbers) >= frame_count:
                frame_numbers = range(frame_numbers.start, frame_count)
                keep_reading = False

            # Load, crop and grayscale frames.
            cropped_frames = np.asarray(get_video_frames_preload(video_path,
                                        frame_numbers=frame_numbers, mask=mask, func=grayscale),
                                        dtype=np.float32)
            # Calculate motion energy for those frames and append to big array
            me = np.append(me, np.mean(np.abs(cropped_frames[1:] - cropped_frames[:-1]),
                                       axis=(1, 2)))
            # Next set of frames
            n += 1
    else:
        # Compute on entire video at once
        cropped_frames = get_video_frames_preload(video_path, frame_numbers=None, mask=mask,
                                                  func=grayscale)
        cropped_frames = np.asarray(cropped_frames, dtype=np.float32)
        me = np.mean(np.abs(cropped_frames[1:] - cropped_frames[:-1]), axis=(1, 2))

    # copy last value to make motion energy fit frame length
    me = np.append(me, me[-1])

    # save ME
    me_file = alf_path.joinpath(f'{label}Camera.ROIMotionEnergy.npy')
    np.save(me_file, me)
    end_T = time.time()
    print(f'{label}Camera computed in', np.round((end_T - start_T), 2))

    return me_file, roi_file
