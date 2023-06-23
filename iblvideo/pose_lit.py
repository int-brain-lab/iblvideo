"""Functions to run Lightning Pose on IBL data with existing networks."""

import cv2
from glob import glob
import logging
import numpy as np
import os
import pandas as pd
from pathlib import Path
import shutil
import sys

from iblvideo.params import BODY_FEATURES, SIDE_FEATURES, LEFT_VIDEO, RIGHT_VIDEO, BODY_VIDEO
from iblvideo.pose_lit_utils import analyze_video, collect_model_paths
from iblvideo.utils import _run_command

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
_logger = logging.getLogger('ibllib')


def _s00_subsample(file_in: Path, file_out: Path, force: bool = False) -> tuple:
    """Step 0 subsample video for detection. Put 500 uniformly sampled frames into new video.

    :param file_in: video file to subsample
    :param file_out: subsampled video
    :param force: whether to overwrite existing intermediate files
    :return file_out, force: subsampled video, updated force parameter
    """
    step = '00'
    action = f'Sparse video {file_out.name} for posture detection'

    if file_out.exists() and not force:
        _logger.info(f'STEP {step} {action} exists, not computing')
    else:
        _logger.info(f'STEP {step} START {action}')

        cap = cv2.VideoCapture(str(file_in))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # get from 20 to 500 samples linearly spaced throughout the session
        nsamples = min(max(20, int(frame_count / cap.get(cv2.CAP_PROP_FPS))), 500)
        samples = np.int32(np.round(np.linspace(0, frame_count - 1, nsamples)))

        size = (int(cap.get(3)), int(cap.get(4)))
        # fourcc = cv2.VideoWriter_fourcc(*'H264')  # new
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # old
        out = cv2.VideoWriter(str(file_out), fourcc, 5, size)
        for i in samples:
            cap.set(1, i)
            _, frame = cap.read()
            out.write(frame)
        out.release()
        _logger.info(f'STEP {step} END {action}')
        # set force to true to recompute all subsequent steps
        force = True

    return file_out, force


def _s01_detect_rois(
    tdir: Path,
    mp4_file: Path,
    model_path: Path,
    view: str,
    force: bool = False,
    create_labels: bool = False,
) -> tuple:
    """Step 1 run Lightning Pose to detect ROIS.

    :param tdir: temporary directory to store outputs
    :param mp4_file: temporally subsampled video
    :param model_path: path to model directory
    :param view: camera view; 'left' | 'right' | 'body'
    :param force: whether to overwrite existing intermediate files
    :param create_labels: create a labeled video for debugging purposes
    return file_out, force: Path to dataframe used to crop video, updated force parameter
    """
    step = '01'
    action = f'Posture detection for {mp4_file.name}'

    file_out = next(tdir.glob('*subsampled*.csv'), None)
    if file_out is None or force is True:
        _logger.info(f'STEP {step} START {action}')
        out = analyze_video(
            network='roi_detect',
            mp4_file=str(mp4_file),
            model_path=str(model_path),
            view=view,
            create_labels=create_labels,
        )
        file_out = next(tdir.glob(f'*{out}*.csv'), None)
        _logger.info(f'STEP {step} END {action}')
        # set force to true to recompute all subsequent steps
        force = True
    else:
        _logger.info(f'STEP {step} {action} exists, not computing.')
    return file_out, force


def _s02_run_specialized_networks(dlc_params, tfile, network, view, force=True):

    # Check if final result exists
    result = next(tfile.parent.glob(f'*{network}*filtered.h5'), None)
    if result and force is not True:
        _logger.info(f'STEP 02 lp feature for {tfile.name} already extracted, not computing.')
    else:
        _logger.info(f'STEP 02 START extract lp feature for {tfile.name}')
        analyze_videos(network=network, view=view, mp4_file=tfile)
        # TODO: add EKS!!!
        # deeplabcut.analyze_videos(str(dlc_params), [str(tfile)])
        # deeplabcut.filterpredictions(str(dlc_params), [str(tfile)])
        _logger.info(f'STEP 02 END extract lp feature for {tfile.name}')
        # Set force to true to recompute all subsequent steps
        force = True

    return force


def _s03_extract_pose_alf(tdir, file_label, networks, file_mp4, *args):
    """
    Output an ALF matrix.
    Column names contain the full DLC results [nframes, nfeatures]
    """
    _logger.info(f'STEP 06 START wrap-up and extract ALF files {file_label}')
    if 'bodyCamera' in file_label:
        video_params = BODY_VIDEO
    elif 'leftCamera' in file_label:
        video_params = LEFT_VIDEO
    elif 'rightCamera' in file_label:
        video_params = RIGHT_VIDEO

    # Create alf path to store final files
    alf_path = tdir.parent.parent.joinpath('alf')
    alf_path.mkdir(exist_ok=True, parents=True)

    for roi in networks:
        if networks[roi]['features'] is None:
            continue
        # we need to make sure we use filtered traces
        df = pd.read_hdf(next(tdir.glob(f'*{roi}*filtered.h5')))
        if roi == 'paws':
            whxy = [0, 0, 0, 0]
        else:
            whxy = np.load(next(tdir.glob(f'*{roi}*.whxy.npy')))

        # Simplify the indices of the multi index df
        columns = [f'{c[1]}_{c[2]}' for c in df.columns.to_flat_index()]
        df.columns = columns

        # translate and scale the specialized window in the full initial frame
        post_crop_scale = networks[roi]['postcrop_downsampling']
        pre_crop_scale = 1.0 / video_params['sampling']
        for ind in columns:
            if ind[-1] == 'x':
                df[ind] = df[ind].apply(lambda x: (x * post_crop_scale + whxy[2]) * pre_crop_scale)
                if video_params['flip']:
                    df[ind] = df[ind].apply(lambda x: video_params['original_size'][0] - x)
            elif ind[-1] == 'y':
                df[ind] = df[ind].apply(lambda x: (x * post_crop_scale + whxy[3]) * pre_crop_scale)

        # concatenate this in one dataframe for all networks
        if 'df_full' not in locals():
            df_full = df.copy()
        else:
            df_full = pd.concat([df_full.reset_index(drop=True),
                                 df.reset_index(drop=True)], axis=1)

    # save in alf path
    file_alf = alf_path.joinpath(f'_ibl_{file_label}.dlc.pqt')
    df_full.to_parquet(file_alf)

    _logger.info(f'STEP 06 END wrap-up and extract ALF files {file_label}')
    return file_alf


def lightning_pose(
    mp4_file: str,
    ckpts_path: str = None,
    force: bool = False,
    create_labels: bool = False,
) -> Path:
    """Analyse a leftCamera, rightCamera, or bodyCamera video with Lightning Pose.

    The process consists of 4 steps:
    0- temporally subsample video frames using ffmpeg for ROI network
    1- run Lightning Pose to detect ROIS for: 'eye', 'nose_tip', 'tongue', 'paws'
    2- run specialized networks on each ROI
    3- output ALF dataset for the raw Lightning Pose output

    :param mp4_file: video file to run
    :param ckpts_path: path to folder with Lightning Pose weights
    :param force: whether to overwrite existing intermediate files
    :param create_labels: create labeled videos for debugging
    :return out_file: path to Lightning Pose table in parquet file format
    """

    # initiate
    mp4_file = Path(mp4_file)  # e.g. '_iblrig_leftCamera.raw.mp4'
    file_label = mp4_file.stem.split('.')[0].split('_')[-1]  # e.g. 'leftCamera'
    if 'bodyCamera' in file_label:
        view = 'body'
        networks = BODY_FEATURES
    else:
        view = 'right' if 'rightCamera' in file_label else 'left'
        networks = SIDE_FEATURES

    # create a directory for temporary files
    raw_video_path = mp4_file.parent
    tdir = raw_video_path.joinpath(f'lp_tmp_{file_label}')
    tdir.mkdir(exist_ok=True)
    # temporary file for temporally subsampled video
    tfiles = {'mp4_sub': tdir / mp4_file.name.replace('.raw.', '.subsampled.')}

    # collect all model paths
    models_dict = collect_model_paths(view=view)

    # Run the processing steps in order
    file_sparse, force = _s00_subsample(
        file_in=Path(mp4_file),
        file_out=Path(tfiles['mp4_sub']),
        force=force,
    )

    file_df_crop, force = _s01_detect_rois(
        tdir=tdir,
        mp4_file=file_sparse,
        model_path=Path(models_dict['roi_detect']),
        view=view,
        force=force,
        create_labels=create_labels,
    )

    for k in networks:
        if networks[k]['features'] is None:
            continue
        if k == 'nose_tip':
            _s02_run_specialized_networks(ckpts_dict[k], network=k, view=view, force=force)

    # out_file = _s03_extract_pose_alf(tdir, file_label, networks, file_mp4)
    #
    # # at the end mop up the mess
    # shutil.rmtree(tdir)
    # # Back to home folder else there  are conflicts in a loop
    # os.chdir(Path.home())
    # print(file_label)

    out_file = 'test'
    return out_file


if __name__ == '__main__':

    from iblvideo.tests.download_test_data import _download_dlc_test_data
    test_dir = _download_dlc_test_data()
    out_file = lightning_pose(
        mp4_file=Path(test_dir) / 'input' / '_iblrig_leftCamera.raw.mp4',
        ckpts_path=None,
        force=False,
        # force=True,
        create_labels=True,
    )
