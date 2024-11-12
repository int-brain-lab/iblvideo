"""Pipeline to run Lightning Pose on a single IBL video with trained networks."""

import cv2
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import shutil
import sys
from typing import Optional, Tuple

from iblvideo.params import LEFT_VIDEO, RIGHT_VIDEO, BODY_VIDEO
from iblvideo.pose_lit_utils import analyze_video, get_crop_window, run_eks
from iblvideo.weights import download_weights

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
_logger = logging.getLogger('ibllib')


def _subsample_video(
    file_in: Path,
    file_out: Path,
    force: bool = False
) -> Tuple[Path, bool]:
    """Step 0: subsample video for ROI detection using 500 uniformly sampled frames.

    :param file_in: path to video file to subsample
    :param file_out: path to subsampled video file
    :param force: whether to overwrite existing intermediate files
    :return: path subsampled video, updated force parameter
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

        # write out frames
        size = (int(cap.get(3)), int(cap.get(4)))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(file_out), fourcc, 5, size)
        for i in samples:
            cap.set(1, i)
            _, frame = cap.read()
            out.write(frame)
        out.release()

        # set force to true to recompute all subsequent steps
        force = True

        _logger.info(f'STEP {step}: END {action}')

    return file_out, force


def _run_network(
    tdir: Path,
    mp4_file: Path,
    model_path: Path,
    network_params: dict,
    camera_params: dict,
    ensemble_number: int = 0,
    force: bool = False,
    roi_df_file: Optional[Path] = None,
) -> Tuple[Path, bool]:
    """Step 1: run Lightning Pose networks.

    :param tdir: temporary directory to store outputs
    :param mp4_file: path to video
    :param model_path: path to model directory
    :param network_params: parameters for network, see SIDE_FEATURES and BODY_FEATURES in params.py
    :param camera_params: parameters for camera, see LEFT_VIDEO etc in params.py
    :param ensemble_number: unique integer to track predictions from different ensemble members
    :param force: whether to overwrite existing intermediate files
    :param roi_df_file: path to dataframe output by ROI network, for computing crop window
    return: path to dataframe with results, updated force parameter
    """
    step = '01'
    action = f'Inference for {network_params["label"]} network on {mp4_file.name}'

    file_out = next(tdir.glob(f'*{network_params["label"]}{ensemble_number}*.csv'), None)
    if file_out and not force:
        _logger.info(f'STEP {step}: {action} exists, not computing.')
    else:
        _logger.info(f'STEP {step}: START {action}')

        # get crop info
        if roi_df_file:
            crop_window = get_crop_window(roi_df_file=roi_df_file, network_params=network_params)
        else:
            crop_window = None

        # get batch size; can increase batch size with smaller frames
        sequence_length = network_params['sequence_length'] * camera_params['sampling']

        analyze_video(
            network=network_params['label'],
            mp4_file=str(mp4_file),
            model_path=str(model_path),
            camera_params=camera_params,
            crop_window=crop_window,
            ensemble_number=ensemble_number,
            sequence_length=sequence_length,
            save_dir=str(tdir),
        )
        file_out = next(tdir.glob(f'*{network_params["label"]}{ensemble_number}*.csv'), None)

        # set force to true to recompute all subsequent steps
        force = True

        _logger.info(f'STEP {step}: END {action}')

    return file_out, force


def _run_eks(
    tdir: Path,
    mp4_file: Path,
    network_params: dict,
    force: bool = False,
) -> Tuple[Path, bool]:
    """Step 1: run Lightning Pose networks.

    :param tdir: temporary directory to store outputs
    :param mp4_file: path to video
    :param network_params: parameters for network, see SIDE_FEATURES and BODY_FEATURES in params.py
    :param force: whether to overwrite existing intermediate files
    return: path to dataframe with results, updated force parameter
    """
    step = '02'
    action = f'EKS for {network_params["label"]} network on {mp4_file.name}'

    # important! no * before .csv, we assume the eks output does not have ensemble number attached
    file_out = next(tdir.glob(f'*{network_params["label"]}.csv'), None)
    if file_out and not force:
        _logger.info(f'STEP {step}: {action} exists, not computing.')
    else:
        _logger.info(f'STEP {step}: START {action}')

        # important! there is a * before .csv to capture ensemble number
        ens_files = list(tdir.glob(f'*{network_params["label"]}*.csv'))

        run_eks(
            network=network_params['label'],
            eks_params=network_params['eks_params'],
            mp4_file=str(mp4_file),
            csv_files=ens_files,
        )

        file_out = next(tdir.glob(f'*{network_params["label"]}.csv'), None)

        # set force to true to recompute all subsequent steps
        force = True

        _logger.info(f'STEP {step}: END {action}')

    return file_out, force


def _extract_pose_alf(
    tdir: Path,
    file_label: str,
    camera_params: dict,
    roi_df_file: Path,
    force: bool = False,
) -> Path:
    """Step 2: collect all outputs into a single file.

    :param tdir: temporary directory to store outputs
    :param file_label: name of video, used for naming alf file
    :param camera_params: parameters for camera, see LEFT_VIDEO etc in params.py
    :param roi_df_file: path to dataframe output by ROI network, for computing crop window
    :param force: whether to overwrite existing intermediate files
    return: path to dataframe with results, updated force parameter
    """
    step = '03'
    action = f'Extract ALF files for {file_label}'

    # Create alf path to store final files
    alf_path = tdir.parent.parent.joinpath('alf')
    alf_path.mkdir(exist_ok=True, parents=True)
    file_alf = alf_path.joinpath(f'_ibl_{file_label}.lightningPose.pqt')

    if file_alf.exists() and not force:
        _logger.info(f'STEP {step}: {action} exists, not computing')
    else:
        _logger.info(f'STEP {step}: START {action}')

        for net_name, net_params in camera_params['features'].items():
            if net_params['features'] is None:
                # skip ROI network
                continue

            # read df and simplify the indices of multi-index
            df = pd.read_csv(next(tdir.glob(f'*{net_name}*.csv')), header=[0, 1, 2], index_col=0)
            columns = [f'{c[1]}_{c[2]}' for c in df.columns.to_flat_index()]
            df.columns = columns

            # translate and scale the specialized window in the full initial frame
            if net_name in ['paws', 'tail_start']:
                whxy = [0, 0, 0, 0]
            else:
                whxy = get_crop_window(roi_df_file=roi_df_file, network_params=net_params)

            post_crop_scale = net_params['postcrop_downsampling']
            pre_crop_scale = 1.0 / camera_params['sampling']

            for ind in columns:
                if ind[-1] == 'x':
                    df[ind] = df[ind].apply(
                        lambda x: (x * post_crop_scale + whxy[2]) * pre_crop_scale)
                    if camera_params['flip']:
                        df[ind] = df[ind].apply(lambda x: camera_params['original_size'][0] - x)
                elif ind[-1] == 'y':
                    df[ind] = df[ind].apply(
                        lambda x: (x * post_crop_scale + whxy[3]) * pre_crop_scale)

            # concatenate this in one dataframe for all networks
            if 'df_full' not in locals():
                df_full = df.copy()
            else:
                df_full = pd.concat(
                    [df_full.reset_index(drop=True), df.reset_index(drop=True)], axis=1)

        # save in alf path
        df_full.to_parquet(file_alf)

        _logger.info(f'STEP {step}: END {action}')

    return file_alf


def lightning_pose(
    mp4_file: str,
    ckpts_path: Optional[Path] = None,
    force: bool = False,
    create_labels: bool = False,
    remove_files: bool = True,
) -> Path:
    """Analyse a leftCamera, rightCamera, or bodyCamera video with Lightning Pose.

    The process consists of 4 steps:
    0. temporally subsample video frames using ffmpeg for ROI network
    1. run Lightning Pose to detect ROIS for: 'eye', 'nose_tip', 'tongue', 'paws'
    2. run specialized networks on each ROI
    3. output ALF dataset for the raw Lightning Pose output

    :param mp4_file: video file to run
    :param ckpts_path: path to folder with Lightning Pose weights
    :param force: whether to overwrite existing intermediate files
    :param create_labels: create labeled videos for debugging
    :param remove_files: True (default) to remove temp files, False to leave (for debugging)
    :return out_file: path to Lightning Pose table in parquet file format
    """

    if ckpts_path is None:
        ckpts_path = download_weights()

    # initiate
    mp4_file = Path(mp4_file)  # e.g. '_iblrig_leftCamera.raw.mp4'
    # TODO: should use ibllib func for this, don't want circular import, maybe pass label in task?
    file_label = mp4_file.stem.split('.')[0].split('_')[-1]  # e.g. 'leftCamera'
    if 'bodyCamera' in file_label:
        camera_params = BODY_VIDEO
    elif 'rightCamera' in file_label:
        camera_params = RIGHT_VIDEO
    elif 'leftCamera' in file_label:
        camera_params = LEFT_VIDEO
    else:
        raise NotImplementedError

    # create a directory for temporary files
    raw_video_path = mp4_file.parent
    tdir = raw_video_path.joinpath(f'lp_tmp_iblrig_{file_label}.raw')
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
    roi_df_file, force = _run_network(
        tdir=tdir,
        mp4_file=file_sparse,
        model_path=next(Path(ckpts_path).glob(camera_params['features']['roi_detect']['weights'])),
        network_params=camera_params['features']['roi_detect'],
        camera_params=camera_params,
        force=force,
    )

    # run specialized networks
    for net_name, net_params in camera_params['features'].items():
        if net_params['features'] is None:
            continue
        # potentially loop over multiple networks (when ensembling)
        for m, model_path in enumerate(Path(ckpts_path).glob(net_params['weights'])):
            _run_network(
                tdir=tdir,
                mp4_file=Path(mp4_file),
                model_path=model_path,
                network_params=net_params,
                camera_params=camera_params,
                ensemble_number=m,
                force=force,
                roi_df_file=roi_df_file,
            )

    # run single-view eks on any ensembles
    for net_name, net_params in camera_params['features'].items():
        if net_params['features'] is None:
            continue
        net_weights = Path(ckpts_path).glob(net_params['weights'])
        if len(list(net_weights)) > 1:
            _run_eks(
                tdir=tdir,
                mp4_file=Path(mp4_file),
                network_params=net_params,
                force=force,
            )

    # collect all outputs into a single file
    out_file = _extract_pose_alf(
        tdir=tdir,
        file_label=file_label,
        camera_params=camera_params,
        roi_df_file=roi_df_file,
        force=force,
    )

    # create final video
    if create_labels:
        from moviepy.editor import VideoFileClip
        from lightning_pose.utils.predictions import create_labeled_video
        mp4_file_labeled = Path(str(mp4_file).replace('.mp4', f'.labeled.mp4'))
        video_clip = VideoFileClip(str(mp4_file))
        preds_df = pd.read_parquet(out_file)
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
            dotsize=5 if 'left' in file_label else 3,
        )

    # clean up temp files
    if remove_files:
        shutil.rmtree(tdir)

    return out_file


if __name__ == '__main__':

    from iblvideo.tests.download_test_data import _download_dlc_test_data

    # cam = 'left'
    # cam = 'right'
    cam = 'body'

    test_dir = _download_dlc_test_data()
    # ckpts_path_local = download_weights()
    ckpts_path_local = Path('/media/mattw/ibl/tracking/current-lp-networks')

    alf_file = lightning_pose(
        mp4_file=Path(test_dir).joinpath(f'input/_iblrig_{cam}Camera.raw.mp4'),
        ckpts_path=ckpts_path_local,
        force=False,
        create_labels=True,
    )
