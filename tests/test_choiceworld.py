import os
import numpy as np
import pandas as pd
from pathlib import Path
from ..choiceworld import dlc
from ..weights import download_weights_flatiron
from . import __version__

data_dir = Path('/mnt/s0/Data/test_data/')


def test_dlc_body(version=__version__):

    file_mp4 = data_dir.joinpath('input', '_iblrig_bodyCamera.raw.mp4')
    tmp_dir = data_dir.joinpath('input', 'dlc_tmp_iblrig_bodyCamera.raw')

    path_dlc = download_weights_flatiron(version=version)

    out_file = dlc(file_mp4, path_dlc)
    assert out_file
    assert (tmp_dir.is_dir() is False)

    out_pqt = pd.read_parquet(out_file)
    ctrl_pqt = pd.read_parquet(data_dir.joinpath('output', f'v{version}',
                                                 '_iblrig_bodyCamera.dlc.pqt'))

    assert np.allclose(np.array(out_pqt), np.array(ctrl_pqt), rtol=10e-2)
    assert np.all(out_pqt.columns, ctrl_pqt.columns)
    os.remove(out_pqt)
