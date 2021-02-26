import os
import numpy as np
import pandas as pd
from ..choiceworld import dlc
from ..weights import download_weights
from .download_test_data import download_test_data
from .. import __version__


def test_dlc(version=__version__):

    test_data = download_test_data()
    path_dlc = download_weights(version=version)

    for cam in ['body', 'left', 'right']:
        file_mp4 = test_data.joinpath('input', f'_iblrig_{cam}Camera.raw.mp4')
        tmp_dir = test_data.joinpath('input', f'dlc_tmp_iblrig_{cam}Camera.raw')

        out_file = dlc(file_mp4, path_dlc)
        assert out_file
        assert (tmp_dir.is_dir() is False)

        out_pqt = pd.read_parquet(out_file)
        ctrl_pqt = pd.read_parquet(test_data.joinpath('output', f'v{version}',
                                                      f'_iblrig_{cam}Camera.dlc.pqt'))

        assert np.allclose(np.array(out_pqt), np.array(ctrl_pqt), rtol=10e-2)
        assert np.all(out_pqt.columns, ctrl_pqt.columns)
        os.remove(out_pqt)
