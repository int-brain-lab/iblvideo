"""Helper functions for running Lightning Action on IBL pose and wheel data."""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightning_action.api import Model
from scipy.interpolate import interp1d
from scipy.signal import butter, sosfiltfilt

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
_logger = logging.getLogger('ibllib')


def interpolate_position(
    re_ts: np.ndarray,
    re_pos: np.ndarray,
    freq: float = 1000,
    kind: str = 'linear',
    fill_gaps: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return linearly interpolated wheel position.

    Copied from brainbox.behavior.wheel to avoid additional dependencies.

    Args:
        re_ts: array of timestamps
        re_pos: array of unwrapped wheel positions
        freq: frequency in Hz of the interpolation
        kind: type of interpolation; 'linear' or 'cubic'
        fill_gaps: minimum gap length to fill; for gaps over this time (seconds),
            forward-fills values before interpolating

    Returns:
        tuple of (interpolated position array, timestamps of interpolated positions)
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


def velocity_filtered(
    pos: np.ndarray,
    fs: float,
    corner_frequency: float = 20,
    order: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute wheel velocity from uniformly sampled wheel data.

    Copied from brainbox.behavior.wheel to avoid additional dependencies.

    Args:
        pos: vector of uniformly sampled wheel positions
        fs: sampling frequency in Hz
        corner_frequency: corner frequency of the low-pass filter
        order: order of the Butterworth filter

    Returns:
        tuple of (velocity array, acceleration array)
    """
    sos = butter(**{'N': order, 'Wn': corner_frequency / fs * 2, 'btype': 'lowpass'}, output='sos')
    vel = np.insert(np.diff(sosfiltfilt(sos, pos)), 0, 0) * fs
    acc = np.insert(np.diff(vel), 0, 0) * fs
    return vel, acc


def resample_dataframe(x1: np.ndarray, y1: pd.DataFrame, x2: np.ndarray) -> pd.DataFrame:
    """Resample columns of dataframe.

    Args:
        x1: initial timestamps
        y1: initial data in dataframe
        x2: final timestamps to resample to

    Returns:
        dataframe with data resampled to x2 timestamps
    """

    # interpolate pose data to new timestamps
    data_resampled = pd.DataFrame(index=range(len(x2)))

    for col in y1.columns:
        # handle NaN values by interpolating only valid data points
        valid_mask = ~y1[col].isna()
        if valid_mask.sum() > 1:  # need at least 2 points to interpolate
            interpolator = interp1d(
                x1[valid_mask],
                y1.loc[valid_mask, col],
                kind='linear',
                bounds_error=False,
                fill_value='extrapolate',
            )
            data_resampled[col] = interpolator(x2)
        else:
            # if insufficient valid data, fill with NaN
            data_resampled[col] = np.nan

    return data_resampled


def combine_input_streams(
    pose_file: Path,
    pose_timestamp_file: Path,
    wheel_file: Path,
    wheel_timestamp_file: Path,
    paw_label: str,
    flip: bool,
    original_dims: list,
    file_out: Path,
) -> tuple[pd.DataFrame, np.ndarray, bool]:
    """Combine pose and wheel data for paw segmentation model input.

    Args:
        pose_file: pose file
        pose_timestamp_file: timestamps associated with pose file
        wheel_file: wheel file
        wheel_timestamp_file: timestamps associated with wheel file
        paw_label: which paw to run network on
        flip: if True flip coordinates around vertical axis before feeding into network
        original_dims: [height, width] of original video (used for flipping)
        file_out: where to save combined data

    Returns:
        tuple of (dataframe with combined pose and wheel data, timestamp array,
        bool indicating whether pose data was resampled)
    """
    # load poses and extract paw of interest
    poses = pd.read_parquet(pose_file)
    poses = poses.loc[:, (f'{paw_label}_x', f'{paw_label}_y')]

    # load pose timestamps
    pose_t = np.load(pose_timestamp_file)

    # calculate framerate
    dt = np.diff(pose_t)
    median_dt = np.median(dt)
    fps = 1.0 / median_dt
    _logger.info(f'Detected pose framerate: {fps:.1f} Hz')

    # check if we need to downsample
    target_fps = 60.0
    fps_tolerance = 10.0

    resampled = False
    if abs(fps - target_fps) <= fps_tolerance:
        _logger.info(f'Framerate {fps:.1f} Hz within tolerance, no resampling needed')
    else:
        _logger.info(
            f'Framerate {fps:.1f} Hz > {target_fps + fps_tolerance} Hz, '
            f'downsampling to {target_fps} Hz'
        )

        # create new uniform timestamp array at target fps
        n_frames_beg = pose_t.shape[0]
        duration = pose_t[-1] - pose_t[0]
        n_frames_target = int(duration * target_fps) + 1
        pose_t_resampled = np.linspace(pose_t[0], pose_t[-1], n_frames_target)

        poses = resample_dataframe(pose_t, poses, pose_t_resampled)
        pose_t = pose_t_resampled
        n_frames_end = pose_t.shape[0]

        resampled = True

        _logger.info(f'Resampled from {n_frames_beg} to {n_frames_end} frames')

    # load wheel data
    wheel_ticks = np.load(wheel_file)
    wheel_ticks_times = np.load(wheel_timestamp_file)

    # sample and compute velocity
    freq = 1000  # sample at 1000 Hz
    wheel_pos, wheel_t = interpolate_position(wheel_ticks_times, wheel_ticks, freq=freq)
    wheel_vel, _ = velocity_filtered(wheel_pos, freq)

    # interpolate wheel data to match pose timestamps
    interpolator = interp1d(
        wheel_t,
        wheel_vel,
        fill_value=(wheel_vel[0], wheel_vel[-1]),
        bounds_error=False,
    )
    wheel_vel_aligned = interpolator(pose_t)

    # flip x coordinates and wheel velocity if necessary (right camera)
    if flip:
        poses[f'{paw_label}_x'] = original_dims[1] - poses[f'{paw_label}_x']
        wheel_vel_aligned *= -1.0  # keep correlation between paw and wheel directions

    # combine data streams
    assert poses.shape[0] == wheel_vel_aligned.shape[0]
    poses['wheel_vel'] = wheel_vel_aligned
    file_out.parent.mkdir(exist_ok=True, parents=True)
    poses.to_csv(file_out)

    return poses, pose_t, resampled


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

    Args:
        tdir: temporary directory for intermediate files
        pose_file: pose file
        pose_timestamp_file: timestamps associated with pose file
        wheel_file: wheel file
        wheel_timestamp_file: timestamps associated with wheel file
        paw_label: which paw to run network on
        ensemble_number: unique integer to track predictions from different ensemble members
        model_path: path to model directory
        flip: if True flip coordinates around vertical axis before feeding into network
        original_dims: [height, width] of original video (used for flipping)
        sequence_length: length of temporal sequences for processing
        file_out: where to save model outputs
    """

    # load pose and wheel data, combine, and save to a new tmp file
    file_int = tdir.joinpath('features', f'{paw_label}.{ensemble_number}.csv')
    _, timestamps, resampled = combine_input_streams(
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
    model.config['training']['sequence_length'] = sequence_length

    # generate predictions
    model.predict(
        data_path=tdir,
        input_dir='features',
        output_dir=str(Path(file_out).parent),  # cannot be None, update in LA
        output_file=file_out,
        expt_ids=[f'{paw_label}.{ensemble_number}'],
    )

    # resample timestamps if necessary
    if resampled:

        # load original timestamps
        pose_t = np.load(pose_timestamp_file)

        # redefine resampled timestamps
        pose_t_resampled = timestamps

        # load saved predictions
        probs_resampled = pd.read_csv(file_out, index_col=0, header=[0])

        # interpolate pose data to new timestamps
        probs = resample_dataframe(pose_t_resampled, probs_resampled, pose_t)
        assert probs.shape[0] == pose_t.shape[0], (
            f'probs (n_t={probs.shape[0]}) and timestamps (n_t={pose_t.shape[0]}) mismatch'
        )

        # re-save probabilities
        probs.to_csv(file_out)


def run_ensembling(
    csv_files: list[Path],
    file_out: Path,
) -> None:
    """Ensemble predictions from multiple networks by averaging probabilities.

    Args:
        csv_files: list of paths to individual network prediction CSV files
        file_out: path where ensembled results will be saved
    """
    if not csv_files:
        raise ValueError("No CSV files provided for ensembling")

    # load all prediction files
    dfs = []
    network_nums = []

    for csv_file in csv_files:
        df = pd.read_csv(csv_file, index_col=0, header=[0])  # assuming first column is index
        dfs.append(df)

        # extract network number from filename (e.g., 'paw_l.0.csv' -> 0)
        filename = csv_file.name
        # expected format: {paw_label}.{network_num}.csv
        parts = filename.split('.')
        if len(parts) >= 3:
            network_num = parts[-2]  # second to last part before .csv
        else:
            # Fallback: use position in list
            network_num = str(len(network_nums))
        network_nums.append(network_num)

    # get prediction column names (excluding index column)
    pred_columns = dfs[0].columns.tolist()

    # create output dataframe with same index as input
    result_df = pd.DataFrame(index=dfs[0].index)

    # calculate ensemble means and variances for each prediction column
    for col in pred_columns:
        # stack all predictions for this column
        stacked_preds = pd.concat([df[col] for df in dfs], axis=1)
        # calculate mean across networks
        result_df[col] = stacked_preds.mean(axis=1)
        # calculate variance across networks
        result_df[f'{col}_ens_var'] = stacked_preds.var(axis=1)

    # add individual network predictions with network suffix
    # for i, (df, net_num) in enumerate(zip(dfs, network_nums)):
    #     for col in pred_columns:
    #         result_df[f'{col}_{net_num}'] = df[col]

    # reorder columns: ensemble means first, then individual network predictions
    ensemble_cols = pred_columns + [f'{col}_ens_var' for col in pred_columns]
    # individual_cols = [f'{col}_{net_num}' for net_num in network_nums for col in pred_columns]

    # final column order
    # final_columns = ensemble_cols + individual_cols
    # result_df = result_df[final_columns]
    result_df = result_df[ensemble_cols]

    # save to CSV
    result_df.to_csv(file_out)
