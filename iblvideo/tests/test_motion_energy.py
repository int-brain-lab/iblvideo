import os
import numpy as np
from iblvideo.motion_energy import motion_energy
from iblvideo.tests import _download_me_test_data


def test_motion_energy():

    test_data = _download_me_test_data()
    for cam in ['body', 'left', 'right']:
        print(f"Running test for {cam}")
        ctrl_me = np.load(test_data.joinpath(f'output/{cam}Camera.ROIMotionEnergy.npy'))
        ctrl_roi = np.load(test_data.joinpath(f'output/{cam}ROIMotionEnergy.position.npy'))
        dlc_pqt = test_data.joinpath(f'alf/_ibl_{cam}Camera.dlc.pqt')

        # Test with all frames
        me_file, roi_file = motion_energy(test_data, dlc_pqt, frames=None)
        test_me = np.load(me_file)
        test_roi = np.load(roi_file)
        assert all(test_me == ctrl_me)
        assert all(test_roi == ctrl_roi)

        # Test with frame chunking
        me_file, roi_file = motion_energy(test_data, dlc_pqt, frames=70)
        test_me = np.load(me_file)
        test_roi = np.load(roi_file)
        assert all(test_me == ctrl_me)
        assert all(test_roi == ctrl_roi)

        os.remove(me_file)
        os.remove(roi_file)
