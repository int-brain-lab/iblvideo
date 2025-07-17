import gc
import multiprocessing
import shutil

import numpy as np
import pandas as pd
import torch

from iblvideo.segmentation_la import lightning_action
from iblvideo.tests.download_test_data import _download_la_test_data
from iblvideo.weights import download_la_models


def _test_lightning_action(cam):

    test_data = _download_la_test_data()
    ckpts_path = download_la_models()

    pose_file = test_data.joinpath('input', f'_iblrig_{cam}Camera.lightningPose.pqt')
    pose_timestamp_file = Path(test_dir).joinpath(f'input/_ibl_{cam}Camera.times.npy'),
    wheel_file = Path(test_dir).joinpath(f'input/_ibl_wheel.position.npy'),
    wheel_timestamp_file = Path(test_dir).joinpath(f'input/_ibl_wheel.timestamps.npy'),

    tmp_dir = test_data.joinpath('input', f'lp_tmp_iblrig_{cam}Camera')

    out_file = lightning_action(
        pose_file=str(pose_file),
        pose_timestamp_file=str(pose_timestamp_file),
        wheel_file=str(wheel_file),
        wheel_timestamp_file=str(wheel_timestamp_file),
        ckpts_path=ckpts_path,
        force=True,
    )
    assert out_file
    assert (tmp_dir.is_dir() is False)

    out_pqt = pd.read_parquet(out_file)
    ctrl_pqt = pd.read_parquet(
        test_data.joinpath('output', f'_ibl_{cam}Camera.pawstates.pqt')
    )

    # make sure all keypoints/columns exist in output
    cols_to_compare = ('_x', '_y', '_likelihood')
    ctrl_columns = ctrl_pqt.columns[ctrl_pqt.columns.str.endswith(cols_to_compare)]
    out_columns = out_pqt.columns[out_pqt.columns.str.endswith(cols_to_compare)]
    assert np.all(out_columns == ctrl_columns), f'{out_columns}\n\n{ctrl_columns}'

    # compare entries
    try:
        assert np.allclose(
            np.array(out_pqt.loc[:, out_columns]), np.array(ctrl_pqt.loc[:, ctrl_columns]),
            rtol=1e-1, equal_nan=True,
        )
    except AssertionError:
        diff = np.abs(np.array(out_pqt) - np.array(ctrl_pqt))
        out_pqt.to_parquet(test_data.joinpath(f'_ibl_{cam}Camera.pawstates.failed.pqt'))
        print(np.nanmax(diff, axis=0), np.nanmean(diff, axis=0))
        # trigger test fail after saving out diff dataframe
        assert np.allclose(np.array(out_pqt), np.array(ctrl_pqt), rtol=1e-10, equal_nan=True)

    alf_path = test_data.joinpath('alf')
    shutil.rmtree(str(alf_path))


def test_lightning_action_left():
    _test_lightning_action('left')


def test_lightning_action_right():
    _test_lightning_action('right')
