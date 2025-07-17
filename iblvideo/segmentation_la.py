"""Pipeline to run Lightning Action on a single IBL video with trained networks."""

import logging
import shutil
import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
import torch

from iblvideo.segmentation_la_utils import analyze_video, run_ensembling
from iblvideo.params_lp import LEFT_VIDEO, RIGHT_VIDEO
from iblvideo.weights import download_la_models

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
_logger = logging.getLogger('ibllib')


def _run_network(
    tdir: Path,
    pose_file: Path,
    pose_timestamp_file: Path,
    wheel_file: Path,
    wheel_timestamp_file: Path,
    file_label: str,
    paw_label: str,
    model_path: Path,
    camera_params: dict,
    ensemble_number: int = 0,
    force: bool = False,
) -> Tuple[Path, bool]:
    """Step 1: run Lightning Action networks.

    :param tdir: temporary directory to store outputs
    :param pose_file: pose file
    :param pose_timestamp_file: timestamps associated with pose file
    :param wheel_file: wheel file
    :param wheel_timestamp_file: timestamps associated with wheel file
    :param file_label: which video to run network on
    :param paw_label: which paw to run network on
    :param model_path: path to model directory
    :param camera_params: parameters for camera, see LEFT_VIDEO etc in params_lp.py
    :param ensemble_number: unique integer to track predictions from different ensemble members
    :param force: whether to overwrite existing intermediate files
    return: path to dataframe with results, updated force parameter
    """
    step = '01'
    action = f'Inference for network {ensemble_number} on {paw_label} in {pose_file.name}'

    file_out = tdir.joinpath(f'{file_label}.{paw_label}.{ensemble_number}.csv')

    if file_out.exists() and not force:
        _logger.info(f'STEP {step}: {action} exists, not computing.')
    else:
        _logger.info(f'STEP {step}: START {action}')

        # get batch size
        sequence_length = 500

        analyze_video(
            tdir=tdir,
            pose_file=pose_file,
            pose_timestamp_file=pose_timestamp_file,
            wheel_file=wheel_file,
            wheel_timestamp_file=wheel_timestamp_file,
            paw_label=paw_label,
            ensemble_number=ensemble_number,
            model_path=model_path,
            flip=camera_params['flip'],
            original_dims=camera_params['original_size'],
            sequence_length=sequence_length,
            file_out=file_out,
        )

        # set force to true to recompute all subsequent steps
        force = True

        _logger.info(f'STEP {step}: END {action}')

    return file_out, force


def _run_ensembling(
    tdir: Path,
    pose_file: Path,
    file_label: str,
    paw_label: str,
    force: bool = False,
) -> Tuple[Path, bool]:
    """Step 2: run ensembling.

    :param tdir: temporary directory to store outputs
    :param pose_file: path to pose
    :param file_label: which video to run network on
    :param paw_label: which paw to run network on
    :param network_params: parameters for network, see SIDE_FEATURES and BODY_FEATURES in params_lp.py
    :param force: whether to overwrite existing intermediate files
    return: path to dataframe with results, updated force parameter
    """
    step = '02'
    action = f'Ensembling for {paw_label} network on {pose_file.name}'

    file_out = tdir.joinpath(f'{file_label}.{paw_label}.csv')

    if file_out.exists() and not force:
        _logger.info(f'STEP {step}: {action} exists, not computing.')
    else:
        _logger.info(f'STEP {step}: START {action}')

        # important! there is a * before .csv to capture ensemble number
        ens_files = list(tdir.glob(f'*{file_label}.{paw_label}*.csv'))

        run_ensembling(
            paw_label=paw_label,
            csv_files=ens_files,
            file_out=file_out,
        )

        # set force to true to recompute all subsequent steps
        force = True

        _logger.info(f'STEP {step}: END {action}')

    return file_out, force


def _extract_pose_alf(
    tdir: Path,
    file_label: str,
    paw_labels: list[str],
    force: bool = False,
) -> Path:
    """Step 3: collect all outputs into a single file.

    :param tdir: temporary directory to store outputs
    :param file_label: name of video, used for naming alf file
    :param paw_labels: names of paws model is run on
    :param force: whether to overwrite existing intermediate files
    return: path to dataframe with results, updated force parameter
    """
    step = '03'
    action = f'Extract ALF files for {file_label}'

    # Create alf path to store final files
    alf_path = tdir.parent.parent.joinpath('alf')
    alf_path.mkdir(exist_ok=True, parents=True)
    file_alf = alf_path.joinpath(f'_ibl_{file_label}.pawstates.pqt')

    if file_alf.exists() and not force:
        _logger.info(f'STEP {step}: {action} exists, not computing')
    else:
        _logger.info(f'STEP {step}: START {action}')

        # TODO

        # save in alf path
        df_full.to_parquet(file_alf)

        _logger.info(f'STEP {step}: END {action}')

    return file_alf


def lightning_action(
    pose_file: str,
    pose_timestamp_file: str,
    wheel_file: str,
    wheel_timestamp_file: str,
    ckpts_path: Path | None = None,
    force: bool = False,
    remove_files: bool = True,
) -> Path:
    """Analyse poses from a leftCamera or rightCamera video with Lightning Action.

    The process consists of 4 steps:
    1. run Lightning Action on 'paw_r'
    2. run Lightning Action on 'paw_l'
    3. output ALF dataset for the raw Lightning Action output

    :param pose_file: pose file
    :param pose_timestamp_file: timestamps associated with pose file
    :param wheel_file: wheel file
    :param wheel_timestamp_file: timestamps associated with wheel file
    :param ckpts_path: path to folder with Lightning Pose weights
    :param force: whether to overwrite existing intermediate files
    :param remove_files: True (default) to remove temp files, False to leave (for debugging)
    :return out_file: path to Lightning Pose table in parquet file format
    """

    if ckpts_path is None:
        ckpts_path = download_la_models()

    # initiate
    pose_file = Path(pose_file)  # e.g. '_iblrig_leftCamera.lightningPose.pqt'
    file_label = pose_file.stem.split('.')[0].split('_')[-1]  # e.g. 'leftCamera'
    if 'rightCamera' in file_label:
        camera_params = RIGHT_VIDEO
    elif 'leftCamera' in file_label:
        camera_params = LEFT_VIDEO
    else:
        raise NotImplementedError

    paw_labels = ['paw_r', 'paw_l']

    # create a directory for temporary files
    pose_path = pose_file.parent.parent
    tdir = pose_path.joinpath(f'la_tmp_iblrig_{file_label}')
    tdir.mkdir(exist_ok=True)

    net_weights = sorted(Path(ckpts_path).glob('paw-*'))

    # run networks on each paw
    for paw_label in paw_labels:
        # loop over multiple networks (for ensembling)
        for m, model_path in enumerate(net_weights):
            _run_network(
                tdir=tdir,
                pose_file=Path(pose_file),
                pose_timestamp_file=Path(pose_timestamp_file),
                wheel_file=Path(wheel_file),
                wheel_timestamp_file=Path(wheel_timestamp_file),
                file_label=file_label,
                paw_label=paw_label,
                model_path=model_path,
                camera_params=camera_params,
                ensemble_number=m,
                force=force,
            )

    # run ensembling
    for paw_label in paw_labels:
        if len(list(net_weights)) > 1:
            _run_ensembling(
                tdir=tdir,
                pose_file=Path(pose_file),
                file_label=file_label,
                paw_label=paw_label,
                force=force,
            )

    # collect all outputs into a single file
    out_file = _extract_pose_alf(
        tdir=tdir,
        file_label=file_label,
        paw_labels=paw_labels,
        force=force,
    )

    # clean up temp files
    if remove_files:
        shutil.rmtree(tdir)

    return out_file


if __name__ == '__main__':

    from iblvideo.tests.download_test_data import _download_la_test_data

    cam = 'left'
    # cam = 'right'

    test_dir = _download_la_test_data()
    ckpts_path_local = download_la_models()

    alf_file = lightning_action(
        pose_file=Path(test_dir).joinpath(f'input/_ibl_{cam}Camera.lightningPose.pqt'),
        pose_timestamp_file=Path(test_dir).joinpath(f'input/_ibl_{cam}Camera.times.npy'),
        wheel_file=Path(test_dir).joinpath(f'input/_ibl_wheel.position.npy'),
        wheel_timestamp_file=Path(test_dir).joinpath(f'input/_ibl_wheel.timestamps.npy'),
        ckpts_path=ckpts_path_local,
        force=False,
    )
