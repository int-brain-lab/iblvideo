from pathlib import Path

import numpy as np
import pandas as pd
from lightning_action.api import Model
from scipy.interpolate import interp1d
from scipy.signal import butter, sosfiltfilt


def interpolate_position(re_ts, re_pos, freq=1000, kind='linear', fill_gaps=None):
    """
    Return linearly interpolated wheel position.

    Copied from brainbox.behavior.wheel to avoid additional dependencies.

    Parameters
    ----------
    re_ts : array_like
        Array of timestamps
    re_pos: array_like
        Array of unwrapped wheel positions
    freq : float
        frequency in Hz of the interpolation
    kind : {'linear', 'cubic'}
        Type of interpolation. Defaults to linear interpolation.
    fill_gaps : float
        Minimum gap length to fill. For gaps over this time (seconds),
        forward fill values before interpolation
    Returns
    -------
    yinterp : array
        Interpolated position
    t : array
        Timestamps of interpolated positions
    """
    t = np.arange(re_ts[0], re_ts[-1], 1 / freq)  # Evenly resample at frequency
    if t[-1] > re_ts[-1]:
        t = t[:-1]  # Occasionally due to precision errors the last sample may be outside of range.
    yinterp = interp1d(re_ts, re_pos, kind=kind)(t)

    if fill_gaps:
        #  Find large gaps and forward fill @fixme This is inefficient
        gaps, = np.where(np.diff(re_ts) >= fill_gaps)

        for i in gaps:
            yinterp[(t >= re_ts[i]) & (t < re_ts[i + 1])] = re_pos[i]

    return yinterp, t


def velocity_filtered(pos, fs, corner_frequency=20, order=8):
    """
    Compute wheel velocity from uniformly sampled wheel data.

    Copied from brainbox.behavior.wheel to avoid additional dependencies.

    pos: array_like
        Vector of uniformly sampled wheel positions.
    fs : float
        Frequency in Hz of the sampling frequency.
    corner_frequency : float
       Corner frequency of low-pass filter.
    order : int
        Order of Butterworth filter.

    Returns
    -------
    vel : np.ndarray
        Array of velocity values.
    acc : np.ndarray
        Array of acceleration values.
    """
    sos = butter(**{'N': order, 'Wn': corner_frequency / fs * 2, 'btype': 'lowpass'}, output='sos')
    vel = np.insert(np.diff(sosfiltfilt(sos, pos)), 0, 0) * fs
    acc = np.insert(np.diff(vel), 0, 0) * fs
    return vel, acc


def combine_input_streams(
    pose_file: Path,
    pose_timestamp_file: Path,
    wheel_file: Path,
    wheel_timestamp_file: Path,
    paw_label: str,
    flip: bool,
    original_dims: list,
    file_out: Path,
) -> pd.DataFrame:
    """Combine pose and wheel data for paw segmentation model input.

    :param pose_file: pose file
    :param pose_timestamp_file: timestamps associated with pose file
    :param wheel_file: wheel file
    :param wheel_timestamp_file: timestamps associated with wheel file
    :param paw_label: which paw to run network on
    :param flip: true to flip coordinates around vertical axis before feeding into network
    :param original_dims: [height, width] of original video (for flipping)
    :param file_out: where to save combined data
    return: dataframe with combined data

    """
    # load poses and extract paw of interest
    poses = pd.read_parquet(pose_file)
    poses = poses.loc[:, (f'{paw_label}_x', f'{paw_label}_y')]

    # load wheel data
    wheel_ticks = np.load(wheel_file)
    wheel_ticks_times = np.load(wheel_timestamp_file)

    # sample and compute velocity
    freq = 1000  # sample at 1000 Hz
    wheel_pos, wheel_t = interpolate_position(wheel_ticks_times, wheel_ticks, freq=freq)
    wheel_vel, _ = velocity_filtered(wheel_pos, freq)

    # interpolate wheel data to match pose timestamps
    interpolator = interp1d(wheel_t, wheel_vel, fill_value='extrapolate')
    pose_t = np.load(pose_timestamp_file)
    wheel_vel_aligned = interpolator(pose_t)

    # combine data streams
    assert poses.shape[0] == wheel_vel_aligned.shape[0]
    poses['wheel_vel'] = wheel_vel_aligned
    file_out.parent.mkdir(exist_ok=True, parents=True)
    poses.to_csv(file_out)

    return poses


def analyze_video(
    tdir: Path,
    pose_file: Path,
    pose_timestamp_file: Path,
    wheel_file: Path,
    wheel_timestamp_file: Path,
    paw_label: str,
    ensemble_number: int,
    model_path: Path,
    flip: bool,
    original_dims: list,
    sequence_length: int,
    file_out: str,
) -> pd.DataFrame:
    """Run Lightning Action network on a set of input files.

    :param tdir: temporary directory for intermediate files
    :param pose_file: pose file
    :param pose_timestamp_file: timestamps associated with pose file
    :param wheel_file: wheel file
    :param wheel_timestamp_file: timestamps associated with wheel file
    :param paw_label: which paw to run network on
    :param ensemble_number: unique integer to track predictions from different ensemble members
    :param model_path: path to model directory
    :param flip: true to flip coordinates around vertical axis before feeding into network
    :param original_dims: [height, width] of original video (for flipping)
    :param sequence_length: length of temporal sequences for processing
    :param file_out: where to save model outputs
    return: dataframe with results
    """

    # load pose and wheel data, combine, and save to a new tmp file
    file_int = tdir.joinpath('features', f'{paw_label}.{ensemble_number}.csv')
    combine_input_streams(
        pose_file=pose_file,
        pose_timestamp_file=pose_timestamp_file,
        wheel_file=wheel_file,
        wheel_timestamp_file=wheel_timestamp_file,
        paw_label=paw_label,
        flip=flip,
        original_dims=original_dims,
        file_out=file_int,
    )

    # load lightning action model
    model = Model.from_dir(model_path)
    model.config['training']['sequence_length'] = 70

    # generate predictions
    model.predict(
        data_path=tdir,
        input_dir='features',
        output_dir=tdir,
        expt_ids=[f'{paw_label}.{ensemble_number}'],
    )


def run_ensembling(
    paw_label: str,
    csv_files: list[Path],
    file_out: Path,
) -> pd.DataFrame:
    """Run ensembling on Lightning Action outputs.

    :param paw_label: which paw to run network on
    :param csv_files: output files to run ensembling on
    :param file_out: where to save model outputs
    return: dataframe with results
    """
    # TODO
