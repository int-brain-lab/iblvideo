import os
import pytest
import numpy as np
import pandas as pd
from iblvideo.motion_energy import motion_energy
from iblvideo.tests.download_test_data import _download_me_test_data


def test_motion_energy():

    test_data = _download_me_test_data()
    for cam in ['body', 'left', 'right']:
        print(f"Running test for {cam}")
        ctrl_me = np.load(test_data.joinpath(f'output/{cam}Camera.ROIMotionEnergy.npy'))
        ctrl_roi = np.load(test_data.joinpath(f'output/{cam}ROIMotionEnergy.position.npy'))
        pose_pqt = test_data.joinpath(f'alf/_ibl_{cam}Camera.lightningPose.pqt')
        file_mp4 = test_data.joinpath('raw_video_data', f'_iblrig_{cam}Camera.raw.mp4')

        # Test with all frames
        me_file, roi_file = motion_energy(file_mp4, pose_pqt, frames=None)
        test_me = np.load(me_file)
        test_roi = np.load(roi_file)
        assert all(test_me == ctrl_me)
        assert all(test_roi == ctrl_roi)

        os.remove(me_file)
        os.remove(roi_file)


def test_with_chunks():

    test_data = _download_me_test_data()
    for cam in ['body', 'left', 'right']:
        print(f"Running test for {cam}")
        ctrl_me = np.load(test_data.joinpath(f'output/{cam}Camera.ROIMotionEnergy.npy'))
        ctrl_roi = np.load(test_data.joinpath(f'output/{cam}ROIMotionEnergy.position.npy'))
        pose_pqt = test_data.joinpath(f'alf/_ibl_{cam}Camera.lightningPose.pqt')
        file_mp4 = test_data.joinpath('raw_video_data', f'_iblrig_{cam}Camera.raw.mp4')

        # Test with frame chunking
        me_file, roi_file = motion_energy(file_mp4, pose_pqt, frames=70)
        test_me = np.load(me_file)
        test_roi = np.load(roi_file)
        assert all(test_me == ctrl_me)
        assert all(test_roi == ctrl_roi)

        os.remove(me_file)
        os.remove(roi_file)


def test_with_nans():
    test_data = _download_me_test_data()
    for cam in ['body', 'left', 'right']:
        print(f"Running test for {cam}")
        pose_pqt = test_data.joinpath(f'alf/_ibl_{cam}Camera.lightningPose.pqt')
        nan_pqt = test_data.joinpath(f'alf/_ibl_{cam}Camera.nan.pqt')
        file_mp4 = test_data.joinpath('raw_video_data', f'_iblrig_{cam}Camera.raw.mp4')

        # Test that all NaN in used columns give correct error
        df_nan = pd.read_parquet(pose_pqt)
        if cam == 'body':
            df_nan['tail_start_y'] = np.nan
        else:
            df_nan['pupil_top_r_x'] = np.nan
            df_nan['pupil_bottom_r_x'] = np.nan
            df_nan['pupil_left_r_x'] = np.nan
            df_nan['pupil_right_r_x'] = np.nan
        df_nan.to_parquet(nan_pqt)
        with pytest.raises(ValueError):
            motion_energy(file_mp4, nan_pqt)
        os.remove(nan_pqt)
