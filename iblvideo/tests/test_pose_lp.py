import gc
import shutil

import numpy as np
import pandas as pd
import torch

from iblvideo.pose_lit import lightning_pose
from iblvideo.tests.download_test_data import _download_lp_test_data
from iblvideo.weights import download_lit_model


def test_lightning_pose_left():
    try:
        _test_lightning_pose(cam='left')
    finally:
        # clean up, even if error
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def test_lightning_pose_right():
    try:
        _test_lightning_pose(cam='right')
    finally:
        # clean up, even if error
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def test_lightning_pose_body():
    try:
        _test_lightning_pose(cam='body')
    finally:
        # clean up, even if error
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _test_lightning_pose(cam):

    test_data = _download_lp_test_data()
    ckpts_path = download_lit_model()

    mp4_file = test_data.joinpath('input', f'_iblrig_{cam}Camera.raw.mp4')
    tmp_dir = test_data.joinpath('input', f'lp_tmp_iblrig_{cam}Camera.raw')

    out_file = lightning_pose(mp4_file=str(mp4_file), ckpts_path=ckpts_path, force=True)
    assert out_file
    assert (tmp_dir.is_dir() is False)

    out_pqt = pd.read_parquet(out_file)
    ctrl_pqt = pd.read_parquet(
        test_data.joinpath('output', f'_ibl_{cam}Camera.lightningPose.pqt')
    )

    # make sure all keypoints/columns exist in output
    cols_to_compare = ('_x', '_y', '_likelihood')
    ctrl_columns = ctrl_pqt.columns[ctrl_pqt.columns.str.endswith(cols_to_compare)]
    out_columns = out_pqt.columns[out_pqt.columns.str.endswith(cols_to_compare)]
    assert np.all(out_columns == ctrl_columns), f'{out_columns}\n\n{ctrl_columns}'

    # only compare entries with likelihood over 0.9
    targets = np.unique(['_'.join(col.split('_')[:-1]) for col in ctrl_columns])
    for t in targets:
        idx_ctrl = ctrl_pqt.loc[ctrl_pqt[f'{t}_likelihood'] < 0.9].index
        idx_out = out_pqt.loc[out_pqt[f'{t}_likelihood'] < 0.9].index
        for idx in [idx_ctrl, idx_out]:
            ctrl_pqt.loc[idx, [f'{t}_x', f'{t}_y', f'{t}_likelihood']] = np.nan
            out_pqt.loc[idx, [f'{t}_x', f'{t}_y', f'{t}_likelihood']] = np.nan

    try:
        assert np.allclose(
            np.array(out_pqt.loc[:, out_columns]), np.array(ctrl_pqt.loc[:, ctrl_columns]),
            rtol=1e-1, equal_nan=True,
        )
    except AssertionError:
        diff = np.abs(np.array(out_pqt) - np.array(ctrl_pqt))
        out_pqt.to_parquet(test_data.joinpath(f'_ibl_{cam}Camera.lightningPose.failed.pqt'))
        print(np.nanmax(diff, axis=0), np.nanmean(diff, axis=0))
        # trigger test fail after saving out diff dataframe
        assert np.allclose(np.array(out_pqt), np.array(ctrl_pqt), rtol=1e-10, equal_nan=True)

    alf_path = test_data.joinpath('alf')
    shutil.rmtree(str(alf_path))
