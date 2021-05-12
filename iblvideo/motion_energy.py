"""
For a session where there is DLC already computed,
load DLC traces to cut video ROIs and then
compute motion energy for these ROIS.

bodyCamera: cut ROI such that mouse body but not wheel motion is in ROI

left(right)Camera: cut whisker pad region
"""

import time
import numpy as np
import pandas as pd
import cv2
import logging

from oneibl.one import ONE
from ibllib.io.video import get_video_frames_preload, label_from_path
from ibllib.io.extractors.camera import get_video_length
from oneibl.stream import VideoStreamer

_log = logging.getLogger('ibllib')


def grayscale(x):
    return cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)


def get_dlc_midpoints(dlc_pqt, targets):
    # Load dataframe
    dlc_df = pd.read_parquet(dlc_pqt)
    mloc = {}
    for t in targets:
        # Set values to nan if likelihood is too low and calcualte midpoints
        idx = dlc_df.loc[dlc_df[f'{t}_likelihood'] < 0.9].index
        dlc_df.loc[idx, [f'{t}_x', f'{t}_y']] = np.nan
        if all(np.isnan(dlc_df[f'{t}_x'])) or all(np.isnan(dlc_df[f'{t}_y'])):
            raise ValueError(f'Failed to calculate midpoint, {t} all NaN in {dlc_pqt}')
        else:
            mloc[t] = [int(np.nanmean(dlc_df[f'{t}_x'])), int(np.nanmean(dlc_df[f'{t}_y']))]
    return mloc


def motion_energy(file_mp4, dlc_pqt, frames=10000):
    """
    Compute motion energy on cropped frames of a single video

    :param file_mp4: Video file to run motion energy for
    :param dlc_pqt: Path to dlc result in pqt file format.
    :param frames: Number of frames to load into memory at once. If None all frames are loaded.
    :return me_file: Path to numpy file contaiing motion energy.
    :return me_roi: Path to numpy file containing ROI coordinates.

    The frames parameter determines how many cropped frames per camera are loaded into memory at
    once and should be set depending on availble RAM. Some approximate numbers for orientation,
    assuming 90 min video and frames set to:
    1       : 152 KB (body),   54 KB (left),   15 KB (right)
    50000   : 7.6 GB (body),  2.7 GB (left), 0.75 GB (right)
    None    :  25 GB (body), 17.5 GB (left), 12.5 GB (right)
    """

    start_T = time.time()
    label = label_from_path(dlc_pqt)

    # Crop ROI
    if label == 'body':
        mloc = get_dlc_midpoints(dlc_pqt, targets=['tail_start'])
        anchor = np.array(mloc['tail_start'])
        w, h = int(anchor[0] * 3 / 5), 210
        x, y = int(anchor[0] - anchor[0] * 3 / 5), int(anchor[1] - 120)
    else:
        mloc = get_dlc_midpoints(dlc_pqt, targets=['nose_tip', 'pupil_top_r'])
        anchor = np.mean([mloc['nose_tip'], mloc['pupil_top_r']], axis=0)
        dist = np.sqrt(np.sum((np.array(mloc['nose_tip']) - np.array(mloc['pupil_top_r']))**2,
                       axis=0))
        w, h = int(dist / 2), int(dist / 3)
        x, y = int(anchor[0] - dist / 4), int(anchor[1])

    # Note that x and y are flipped when loading with cv2, therefore:
    mask = np.s_[y:y + h, x:x + w]
    # save ROI coordinates
    roi = np.asarray([w, h, x, y])
    alf_path = file_mp4.parent.parent.joinpath('alf')
    alf_path.mkdir(exist_ok=True)
    roi_file = alf_path.joinpath(f'{label}ROIMotionEnergy.position.npy')
    np.save(roi_file, roi)

    frame_count = get_video_length(file_mp4)
    me = np.zeros(frame_count,)

    cap = cv2.VideoCapture(str(file_mp4))
    if frames:
        n, keep_reading = 0, True
        while keep_reading:
            # Set the frame numbers to the next #frames, with 1 frame overlap
            frame_numbers = range(n * (frames - 1), n * (frames - 1) + frames)
            # Make sure not to load empty frames
            if np.max(frame_numbers) >= frame_count:
                frame_numbers = range(frame_numbers.start, frame_count)
                keep_reading = False
            # Load, crop and grayscale frames.
            cropped_frames = get_video_frames_preload(cap, frame_numbers=frame_numbers,
                                                      mask=mask, func=grayscale,
                                                      quiet=True).astype(np.float32)
            # Calculate motion energy for those frames and append to big array
            me[frame_numbers[:-1]] = np.mean(np.abs(np.diff(cropped_frames, axis=0)), axis=(1, 2))
            # Next set of frames
            n += 1
    else:
        # Compute on entire video at once
        cropped_frames = get_video_frames_preload(cap, frame_numbers=None, mask=mask,
                                                  func=grayscale, quiet=True).astype(np.float32)
        me[:-1] = np.mean(np.abs(np.diff(cropped_frames, axis=0)), axis=(1, 2))

    # copy last value to make motion energy fit frame length
    cap.release()
    me[-1] = me[-2]

    # save ME
    me_file = alf_path.joinpath(f'{label}Camera.ROIMotionEnergy.npy')
    np.save(me_file, me)
    end_T = time.time()
    print(f'{label}Camera computed in', np.round((end_T - start_T), 2))

    return me_file, roi_file
