import shutil
import numpy as np
import pandas as pd
from pathlib import Path

from one.api import ONE
from iblvideo.pose_lit import lightning_pose
from iblvideo.weights import download_weights
from iblvideo.tests import _download_dlc_test_data
from iblvideo import __version__


def test_lightning_pose(version=__version__):

    one = ONE()

    test_data = _download_dlc_test_data(one=one)
    # ckpts_path = download_weights(version=version, one=one)
    ckpts_path = Path('/media/mattw/ibl/tracking/current-lp-networks')

    for cam in ['body', 'left', 'right']:

        mp4_file = test_data.joinpath('input', f'_iblrig_{cam}Camera.raw.mp4')
        tmp_dir = test_data.joinpath('input', f'lp_tmp_iblrig_{cam}Camera.raw')

        out_file = lightning_pose(mp4_file=mp4_file, ckpts_path=ckpts_path)
        assert out_file
        assert (tmp_dir.is_dir() is False)

        out_pqt = pd.read_parquet(out_file)
        ctrl_pqt = pd.read_parquet(
            test_data.joinpath('output', f'_ibl_{cam}Camera.dlc.pqt'))

        assert np.all(out_pqt.columns == ctrl_pqt.columns)

        # only compare entries with likelihood over 0.9
        targets = np.unique(['_'.join(col.split('_')[:-1]) for col in ctrl_pqt.columns])
        for t in targets:
            idx_ctrl = ctrl_pqt.loc[ctrl_pqt[f'{t}_likelihood'] < 0.9].index
            idx_out = out_pqt.loc[out_pqt[f'{t}_likelihood'] < 0.9].index
            for idx in [idx_ctrl, idx_out]:
                ctrl_pqt.loc[idx, [f'{t}_x', f'{t}_y', f'{t}_likelihood']] = np.nan
                out_pqt.loc[idx, [f'{t}_x', f'{t}_y', f'{t}_likelihood']] = np.nan

        try:
            assert np.allclose(np.array(out_pqt), np.array(ctrl_pqt), rtol=1, equal_nan=True)
        except AssertionError:
            diff = np.abs(np.array(out_pqt) - np.array(ctrl_pqt))
            out_pqt.to_parquet(test_data.joinpath(f'_ibl_{cam}Camera.lp.failed.pqt'))

            print(np.nanmax(diff, axis=0), np.nanmean(diff, axis=0))
            assert np.allclose(np.array(out_pqt), np.array(ctrl_pqt), rtol=1, equal_nan=True)

    alf_path = test_data.joinpath('alf')
    shutil.rmtree(alf_path)
