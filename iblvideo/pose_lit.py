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
from typing import Optional

from iblvideo.params import BODY_FEATURES, SIDE_FEATURES, LEFT_VIDEO, RIGHT_VIDEO, BODY_VIDEO
from iblvideo.pose_lit_utils import analyze_video, collect_model_paths, get_crop_window
from iblvideo.utils import _run_command

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
_logger = logging.getLogger('ibllib')


# TODO:
# - move networks to a single location, remove `get_data_dir` func in `analyze_video`
# - test on right video
# - compare left/right traces to DLC traces


def _subsample_video(file_in: Path, file_out: Path, force: bool = False) -> tuple:
    """Step 0: subsample video for detection - put 500 uniformly sampled frames into new video.

    :param file_in: video file to subsample
    :param file_out: subsampled video
    :param force: whether to overwrite existing intermediate files
    :return: subsampled video, updated force parameter
    """
    step = '00'
    action = f'Subsample video {file_out.name} for ROI detection'

    if file_out.exists() and not force:
        _logger.info(f'STEP {step}: {action} exists, not computing')
    else:
        _logger.info(f'STEP {step}: START {action}')

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
        _logger.info(f'STEP {step}: END {action}')
        # set force to true to recompute all subsequent steps
        force = True

    return file_out, force


def _run_network(
    tdir: Path,
    mp4_file: Path,
    model_path: Path,
    network: dict,
    view: str,
    force: bool = False,
    create_labels: bool = False,
    roi_df_file: Optional[Path] = None,
):
    """Step 1: run Lightning Pose networks.

    :param tdir: temporary directory to store outputs
    :param mp4_file: temporally subsampled video
    :param model_path: path to model directory
    :param view: camera view; 'left' | 'right' | 'body'
    :param force: whether to overwrite existing intermediate files
    :param create_labels: create a labeled video for debugging purposes
    return: path to dataframe used to crop video, updated force parameter
    """
    step = '01'
    action = f'Inference for {network["label"]} network on {mp4_file.name}'

    file_out = next(tdir.glob(f'*{network["label"]}*.csv'), None)
    if file_out and not force:
        _logger.info(f'STEP {step}: {action} exists, not computing.')
    else:
        _logger.info(f'STEP {step}: START {action}')
        if roi_df_file:
            crop_window = get_crop_window(file_df_crop=roi_df_file, network=network)
        else:
            crop_window = None
        analyze_video(
            network=network['label'],
            mp4_file=str(mp4_file),
            model_path=str(model_path),
            view=view,
            create_labels=create_labels,
            crop_window=crop_window,
            sequence_length=network['sequence_length'],
            save_dir=str(tdir),
        )
        file_out = next(tdir.glob(f'*{network["label"]}*.csv'), None)
        _logger.info(f'STEP {step}: END {action}')
        # set force to true to recompute all subsequent steps
        force = True

    return file_out, force


def _extract_pose_alf(
    tdir: Path, file_label: str, networks: dict, mp4_file: Path, roi_df_file: Path, *args
):
    """Step 2: collect all outputs into a single file.

    """
    step = '02'
    action = f'Extract ALF files for {file_label}'

    # Create alf path to store final files
    alf_path = tdir.parent.parent.joinpath('alf')
    alf_path.mkdir(exist_ok=True, parents=True)
    file_alf = alf_path.joinpath(f'_ibl_{file_label}.dlc.pqt')

    if file_alf.exists():
        _logger.info(f'STEP {step}: {action} exists, not computing')
    else:
        _logger.info(f'STEP {step}: START {action}')

        if 'bodyCamera' in file_label:
            video_params = BODY_VIDEO
        elif 'leftCamera' in file_label:
            video_params = LEFT_VIDEO
        elif 'rightCamera' in file_label:
            video_params = RIGHT_VIDEO
        else:
            raise NotImplementedError

        for k, v in networks.items():
            if v['features'] is None:
                # skip ROI network
                continue
            df = pd.read_csv(next(tdir.glob(f'*{k}*.csv')), header=[0, 1, 2], index_col=0)
            if k == 'paws':
                whxy = [0, 0, 0, 0]
            else:
                whxy = get_crop_window(file_df_crop=roi_df_file, network=v)

            # Simplify the indices of the multi index df
            columns = [f'{c[1]}_{c[2]}' for c in df.columns.to_flat_index()]
            df.columns = columns

            # translate and scale the specialized window in the full initial frame
            post_crop_scale = v['postcrop_downsampling']
            pre_crop_scale = 1.0 / video_params['sampling']
            for ind in columns:
                if ind[-1] == 'x':
                    df[ind] = df[ind].apply(
                        lambda x: (x * post_crop_scale + whxy[2]) * pre_crop_scale)
                    if video_params['flip']:
                        df[ind] = df[ind].apply(lambda x: video_params['original_size'][0] - x)
                elif ind[-1] == 'y':
                    df[ind] = df[ind].apply(
                        lambda x: (x * post_crop_scale + whxy[3]) * pre_crop_scale)

            # concatenate this in one dataframe for all networks
            if 'df_full' not in locals():
                df_full = df.copy()
            else:
                df_full = pd.concat([
                    df_full.reset_index(drop=True),
                    df.reset_index(drop=True)
                ], axis=1)

        # save in alf path
        df_full.to_parquet(file_alf)

        _logger.info(f'STEP {step}: END {action}')

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

    # subsample video for ROI network
    file_sparse, force = _subsample_video(
        file_in=Path(mp4_file),
        file_out=Path(tfiles['mp4_sub']),
        force=force,
    )

    # run ROI network
    proj_path = next(Path(ckpts_path).glob(networks['roi_detect']['weights']))
    roi_df_file, force = _run_network(
        tdir=tdir,
        mp4_file=file_sparse,
        model_path=next(proj_path.glob('*/*/*/*/*/*/*.ckpt')),
        network=networks['roi_detect'],
        view=view,
        force=force,
        create_labels=create_labels,
    )

    # run all other networks
    for k, v in networks.items():
        if v['features'] is None:
            continue
        _run_network(
            tdir=tdir,
            mp4_file=Path(mp4_file),
            model_path=Path(models_dict[k]),
            network=v,
            view=view,
            force=force,
            create_labels=False,  # need to make intermediate videos for this to work
            roi_df_file=roi_df_file,
        )

    # collect all outputs into a single file
    out_file = _extract_pose_alf(
        tdir=tdir,
        file_label=file_label,
        networks=networks,
        mp4_file=mp4_file,
        roi_df_file=roi_df_file,
    )

    # create final video
    if create_labels:
        from moviepy.editor import VideoFileClip
        from lightning_pose.utils.predictions import create_labeled_video
        mp4_file_labeled = Path(str(mp4_file).replace('.mp4', f'.labeled.mp4'))
        video_clip = VideoFileClip(str(mp4_file))
        preds_df = pd.read_parquet(out_file)
        print(preds_df.head())
        # transform df to numpy array
        keypoints_arr = np.reshape(preds_df.to_numpy(), [preds_df.shape[0], -1, 3])
        xs_arr = keypoints_arr[:, :, 0]
        ys_arr = keypoints_arr[:, :, 1]
        mask_array = keypoints_arr[:, :, 2] > 0.9
        # video generation
        create_labeled_video(
            clip=video_clip,
            xs_arr=xs_arr,
            ys_arr=ys_arr,
            mask_array=mask_array,
            filename=str(mp4_file_labeled),
        )

    # # at the end mop up the mess
    # shutil.rmtree(tdir)
    # # Back to home folder else there  are conflicts in a loop
    # os.chdir(Path.home())

    return out_file


if __name__ == '__main__':

    from iblvideo.tests.download_test_data import _download_dlc_test_data
    test_dir = _download_dlc_test_data()
    alf_file = lightning_pose(
        mp4_file=Path(test_dir).joinpath('input/_iblrig_leftCamera.raw.mp4'),
        ckpts_path='/media/mattw/ibl/tracking/current-lp-networks',  # TODO: remove hard-coding
        force=False,
        # force=True,
        create_labels=True,
    )
