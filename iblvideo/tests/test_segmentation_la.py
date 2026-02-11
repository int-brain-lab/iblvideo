import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from iblvideo import download_la_models
from iblvideo.segmentation_la import lightning_action
from iblvideo.segmentation_la_utils import combine_input_streams, resample_dataframe
from iblvideo.tests.download_test_data import _download_la_test_data


def _test_lightning_action(cam):

    test_data = _download_la_test_data()
    ckpts_path = download_la_models()

    pose_file = test_data.joinpath('input', f'_ibl_{cam}Camera.lightningPose.pqt')
    pose_timestamp_file = test_data.joinpath('input', f'_ibl_{cam}Camera.times.npy')
    wheel_file = test_data.joinpath('input', '_ibl_wheel.position.npy')
    wheel_timestamp_file = test_data.joinpath('input', '_ibl_wheel.timestamps.npy')

    tmp_dir = test_data.joinpath('input', f'la_tmp_iblrig_{cam}Camera')

    out_file = lightning_action(
        pose_file=pose_file,
        pose_timestamp_file=pose_timestamp_file,
        wheel_file=wheel_file,
        wheel_timestamp_file=wheel_timestamp_file,
        ckpts_path=ckpts_path,
        sequence_length=70,  # default (500) is longer than test data
        force=True,
    )
    assert out_file
    assert (tmp_dir.is_dir() is False)

    out_pqt = pd.read_parquet(out_file)
    ctrl_pqt = pd.read_parquet(test_data.joinpath('output', f'_ibl_{cam}Camera.pawstates.pqt'))

    # make sure all keypoints/columns exist in output
    cols_to_compare = ('paw_r', 'paw_l')
    ctrl_columns = ctrl_pqt.columns[ctrl_pqt.columns.str.startswith(cols_to_compare)]
    out_columns = out_pqt.columns[out_pqt.columns.str.startswith(cols_to_compare)]
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


class TestResampleDataframe:

    def test_basic_interpolation(self):
        """Test basic linear interpolation functionality"""
        # Create simple test data
        x1 = np.array([0, 1, 2, 3, 4])
        y1 = pd.DataFrame({
            'col1': [0, 10, 20, 30, 40],
            'col2': [0, 5, 10, 15, 20]
        })
        x2 = np.array([0, 0.5, 1.5, 2.5, 3.5, 4])

        y2_result = resample_dataframe(x1, y1, x2)

        # Check return values
        assert isinstance(y2_result, pd.DataFrame)
        assert len(y2_result) == len(x2)
        assert list(y2_result.columns) == list(y1.columns)

        # Check interpolated values
        expected_col1 = [0, 5, 15, 25, 35, 40]  # linear interpolation
        expected_col2 = [0, 2.5, 7.5, 12.5, 17.5, 20]

        np.testing.assert_allclose(y2_result['col1'], expected_col1)
        np.testing.assert_allclose(y2_result['col2'], expected_col2)

    def test_extrapolation(self):
        """Test that extrapolation works correctly"""
        x1 = np.array([1, 2, 3])
        y1 = pd.DataFrame({'col1': [10, 20, 30]})
        x2 = np.array([0, 1.5, 4])  # includes points outside x1 range

        y2_result = resample_dataframe(x1, y1, x2)

        # Should extrapolate linearly
        # At x=0: extrapolated value should be 0 (line: y = 10x)
        # At x=1.5: interpolated value should be 15
        # At x=4: extrapolated value should be 40
        expected = [0, 15, 40]
        np.testing.assert_allclose(y2_result['col1'], expected)

    def test_single_valid_point(self):
        """Test handling when only one valid data point exists"""
        x1 = np.array([0, 1, 2, 3])
        y1 = pd.DataFrame({
            'col1': [np.nan, 10, np.nan, np.nan],  # only one valid point
            'col2': [1, 2, 3, 4]  # all valid points
        })
        x2 = np.array([0, 1, 2])

        y2_result = resample_dataframe(x1, y1, x2)

        # col1 should be all NaN (insufficient data)
        assert y2_result['col1'].isna().all()

        # col2 should be interpolated normally
        assert not y2_result['col2'].isna().any()

    def test_all_nan_column(self):
        """Test handling when entire column is NaN"""
        x1 = np.array([0, 1, 2, 3])
        y1 = pd.DataFrame({
            'col1': [np.nan, np.nan, np.nan, np.nan],
            'col2': [1, 2, 3, 4]
        })
        x2 = np.array([0, 1, 2])

        y2_result = resample_dataframe(x1, y1, x2)

        # col1 should be all NaN
        assert y2_result['col1'].isna().all()

        # col2 should be interpolated normally
        assert not y2_result['col2'].isna().any()

    def test_partial_nan_interpolation(self):
        """Test interpolation with some NaN values"""
        x1 = np.array([0, 1, 2, 3, 4])
        y1 = pd.DataFrame({
            'col1': [0, np.nan, 20, np.nan, 40]  # NaN at indices 1,3
        })
        x2 = np.array([0, 1, 2, 3, 4])

        y2_result = resample_dataframe(x1, y1, x2)

        # Should interpolate using only valid points (0,0), (2,20), (4,40)
        # Linear interpolation: y = 10x
        expected = [0, 10, 20, 30, 40]
        np.testing.assert_allclose(y2_result['col1'], expected)


class TestCombineInputStreams:

    @pytest.fixture
    def setup_test_data(self):
        """Create temporary test data files"""
        temp_dir = Path(tempfile.mkdtemp())

        def create_test_files(fps, duration=5.0, paw_label='paw_l'):
            """Create test files for given fps"""
            n_frames = int(fps * duration)

            # Create pose timestamps
            pose_timestamps = np.linspace(0, duration, n_frames)
            pose_timestamp_file = temp_dir / f'pose_timestamps_{fps}hz.npy'
            np.save(pose_timestamp_file, pose_timestamps)

            # Create pose data
            poses_data = {
                f'{paw_label}_x': np.random.uniform(100, 500, n_frames),
                f'{paw_label}_y': np.random.uniform(100, 400, n_frames),
                'other_x': np.random.uniform(0, 600, n_frames),  # other columns to test filtering
                'other_y': np.random.uniform(0, 500, n_frames),
            }
            poses_df = pd.DataFrame(poses_data)
            pose_file = temp_dir / f'poses_{fps}hz.pqt'
            poses_df.to_parquet(pose_file)

            # Create wheel data (simpler, just some test data)
            wheel_duration = duration + 1  # slightly longer
            wheel_n_samples = int(wheel_duration * 100)  # 100 Hz wheel data
            wheel_timestamps = np.linspace(0, wheel_duration, wheel_n_samples)
            wheel_positions = np.cumsum(np.random.randn(wheel_n_samples) * 0.1)  # random walk

            wheel_timestamp_file = temp_dir / f'wheel_timestamps_{fps}hz.npy'
            wheel_file = temp_dir / f'wheel_positions_{fps}hz.npy'
            np.save(wheel_timestamp_file, wheel_timestamps)
            np.save(wheel_file, wheel_positions)

            output_file = temp_dir / f'output_{fps}hz.csv'

            return {
                'pose_file': pose_file,
                'pose_timestamp_file': pose_timestamp_file,
                'wheel_file': wheel_file,
                'wheel_timestamp_file': wheel_timestamp_file,
                'output_file': output_file,
                'original_poses': poses_df,
                'original_timestamps': pose_timestamps,
                'n_frames': n_frames,
                'fps': fps,
                'paw_label': paw_label,
            }

        return create_test_files

    def test_fps_60_no_resampling(self, setup_test_data):
        """Test that 60 Hz data is not resampled"""

        # Create test data at 60 Hz
        test_data = setup_test_data(fps=60, duration=5.0)

        # Run function
        result_poses, result_timestamps, result_resampled = combine_input_streams(
            pose_file=test_data['pose_file'],
            pose_timestamp_file=test_data['pose_timestamp_file'],
            wheel_file=test_data['wheel_file'],
            wheel_timestamp_file=test_data['wheel_timestamp_file'],
            paw_label=test_data['paw_label'],
            flip=False,
            original_dims=[480, 640],
            file_out=test_data['output_file'],
        )

        # Assertions
        # Timestamps should be unchanged
        assert not result_resampled
        np.testing.assert_array_almost_equal(result_timestamps, test_data['original_timestamps'])

        # Number of frames should be unchanged
        assert len(result_poses) == test_data['n_frames']
        assert len(result_timestamps) == test_data['n_frames']

        # Pose data should contain the correct columns
        expected_columns = [f"{test_data['paw_label']}_x", f"{test_data['paw_label']}_y", 'wheel_vel']
        assert list(result_poses.columns) == expected_columns

        # Output file should be created
        assert test_data['output_file'].exists()

    def test_fps_150_downsampling(self, setup_test_data):
        """Test that 150 Hz data is properly downsampled to 60 Hz"""

        # Create test data at 150 Hz
        duration = 5.0
        test_data = setup_test_data(fps=150, duration=duration)

        # Run function
        result_poses, result_timestamps, result_resampled = combine_input_streams(
            pose_file=test_data['pose_file'],
            pose_timestamp_file=test_data['pose_timestamp_file'],
            wheel_file=test_data['wheel_file'],
            wheel_timestamp_file=test_data['wheel_timestamp_file'],
            paw_label=test_data['paw_label'],
            flip=False,
            original_dims=[480, 640],
            file_out=test_data['output_file'],
        )

        # Assertions
        # Should be downsampled to approximately 60 Hz
        assert result_resampled
        expected_frames_60hz = int(duration * 60) + 1
        assert len(result_timestamps) == expected_frames_60hz
        assert len(result_poses) == expected_frames_60hz

        # Timestamps should be uniformly spaced at 60 Hz
        dt = np.diff(result_timestamps)
        expected_dt = 1.0 / 60.0
        np.testing.assert_allclose(dt, expected_dt, rtol=1e-10)

        # Check that timestamps span the original duration
        assert abs(result_timestamps[0] - test_data['original_timestamps'][0]) < 1e-10
        assert abs(result_timestamps[-1] - test_data['original_timestamps'][-1]) < 1e-10

        # Pose data should be interpolated (values should be reasonable)
        paw_x_col = f"{test_data['paw_label']}_x"
        paw_y_col = f"{test_data['paw_label']}_y"

        # Check that interpolated values are within reasonable bounds
        original_x_range = (test_data['original_poses'][paw_x_col].min(),
                            test_data['original_poses'][paw_x_col].max())
        original_y_range = (test_data['original_poses'][paw_y_col].min(),
                            test_data['original_poses'][paw_y_col].max())

        assert result_poses[paw_x_col].min() >= original_x_range[0] - 10  # small tolerance for extrapolation
        assert result_poses[paw_x_col].max() <= original_x_range[1] + 10
        assert result_poses[paw_y_col].min() >= original_y_range[0] - 10
        assert result_poses[paw_y_col].max() <= original_y_range[1] + 10

    def test_fps_30_upsampling(self, setup_test_data):
        """Test that 30 Hz data is properly upsampled to 60 Hz"""

        # Create test data at 30 Hz
        duration = 5.0
        test_data = setup_test_data(fps=30, duration=duration)

        # Run function
        result_poses, result_timestamps, result_resampled = combine_input_streams(
            pose_file=test_data['pose_file'],
            pose_timestamp_file=test_data['pose_timestamp_file'],
            wheel_file=test_data['wheel_file'],
            wheel_timestamp_file=test_data['wheel_timestamp_file'],
            paw_label=test_data['paw_label'],
            flip=False,
            original_dims=[480, 640],
            file_out=test_data['output_file'],
        )

        # Assertions
        # Should be upsampled to approximately 60 Hz
        assert result_resampled
        expected_frames_60hz = int(duration * 60) + 1
        assert len(result_timestamps) == expected_frames_60hz
        assert len(result_poses) == expected_frames_60hz

        # Timestamps should be uniformly spaced at 60 Hz
        dt = np.diff(result_timestamps)
        expected_dt = 1.0 / 60.0
        np.testing.assert_allclose(dt, expected_dt, rtol=1e-10)

        # Check that timestamps span the original duration
        assert abs(result_timestamps[0] - test_data['original_timestamps'][0]) < 1e-10
        assert abs(result_timestamps[-1] - test_data['original_timestamps'][-1]) < 1e-10

        # Pose data should be interpolated (values should be reasonable)
        paw_x_col = f"{test_data['paw_label']}_x"
        paw_y_col = f"{test_data['paw_label']}_y"

        # Check that interpolated values are within reasonable bounds
        original_x_range = (test_data['original_poses'][paw_x_col].min(),
                            test_data['original_poses'][paw_x_col].max())
        original_y_range = (test_data['original_poses'][paw_y_col].min(),
                            test_data['original_poses'][paw_y_col].max())

        assert result_poses[paw_x_col].min() >= original_x_range[0] - 10  # small tolerance for extrapolation
        assert result_poses[paw_x_col].max() <= original_x_range[1] + 10
        assert result_poses[paw_y_col].min() >= original_y_range[0] - 10
        assert result_poses[paw_y_col].max() <= original_y_range[1] + 10

    def test_flip_coordinates(self, setup_test_data):
        """Test that coordinate flipping works correctly"""

        # Create test data
        test_data = setup_test_data(fps=60, duration=2.0)
        original_x = test_data['original_poses'][f"{test_data['paw_label']}_x"].copy()

        # Test with flip=True
        result_poses, _, _ = combine_input_streams(
            pose_file=test_data['pose_file'],
            pose_timestamp_file=test_data['pose_timestamp_file'],
            wheel_file=test_data['wheel_file'],
            wheel_timestamp_file=test_data['wheel_timestamp_file'],
            paw_label=test_data['paw_label'],
            flip=True,
            original_dims=[480, 640],  # height, width
            file_out=test_data['output_file'],
        )

        # Check that x coordinates were flipped: new_x = width - old_x
        expected_x = 640 - original_x
        paw_x_col = f"{test_data['paw_label']}_x"
        np.testing.assert_array_almost_equal(result_poses[paw_x_col], expected_x)

    def test_file_creation(self, setup_test_data):
        """Test that output files are created correctly"""

        test_data = setup_test_data(fps=60, duration=2.0)

        # Ensure output file doesn't exist initially
        if test_data['output_file'].exists():
            test_data['output_file'].unlink()

        # Run function
        combine_input_streams(
            pose_file=test_data['pose_file'],
            pose_timestamp_file=test_data['pose_timestamp_file'],
            wheel_file=test_data['wheel_file'],
            wheel_timestamp_file=test_data['wheel_timestamp_file'],
            paw_label=test_data['paw_label'],
            flip=False,
            original_dims=[480, 640],
            file_out=test_data['output_file'],
        )

        # Check that output file was created and has correct structure
        assert test_data['output_file'].exists()

        # Load and verify CSV structure
        output_df = pd.read_csv(test_data['output_file'], index_col=0)
        expected_columns = [f"{test_data['paw_label']}_x", f"{test_data['paw_label']}_y", 'wheel_vel']
        assert list(output_df.columns) == expected_columns
