"""Functions to run DLC on IBL data with existing networks."""
import deeplabcut  # needs to be imported first  # isort: skip
import logging
import os
import shutil
import subprocess
import time
from collections import OrderedDict
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from ibllib.io.video import get_video_meta

from iblvideo.params_dlc import BODY_FEATURES, BODY_VIDEO, LEFT_VIDEO, RIGHT_VIDEO, SIDE_FEATURES

_logger = logging.getLogger('ibllib')


def _set_dlc_paths(path_dlc: Path) -> None:
    """Replace hard-coded paths in the config.yaml file.

    Args:
        path_dlc: path to directory containing DLC weights
    """
    for yaml_file in path_dlc.rglob('config.yaml'):
        # read the yaml config file
        with open(yaml_file) as fid:
            yaml_data = yaml.safe_load(fid)
        # if the path is correct skip to next
        if Path(yaml_data['project_path']) == yaml_file.parent:
            continue
        # else read the whole file
        with open(yaml_file) as fid:
            yaml_raw = fid.read()
        # patch the offending line and rewrite properly
        with open(yaml_file, 'w+') as fid:
            fid.writelines(
                yaml_raw.replace(
                    yaml_data['project_path'], str(yaml_file.parent)))


def _dlc_init(
    file_mp4: str | Path,
    path_dlc: str | Path,
) -> tuple[Path, dict, dict, Path, dict, str]:
    """Prepare inputs and create temporary filenames.

    Args:
        file_mp4: path to input video file
        path_dlc: path to directory containing DLC weights

    Returns:
        tuple of (video path, dlc config paths dict, networks dict, temp directory,
        temp file paths dict, camera label string)
    """
    # Prepare inputs
    file_mp4 = Path(file_mp4)  # _iblrig_leftCamera.raw.mp4
    file_label = file_mp4.stem.split('.')[0].split('_')[-1]
    if 'bodyCamera' in file_label:
        networks = BODY_FEATURES
    else:
        networks = SIDE_FEATURES

    path_dlc = Path(path_dlc)
    _set_dlc_paths(path_dlc)
    dlc_params = {k: Path(glob(str(path_dlc.joinpath(networks[k]['weights'], 'config.yaml')))[0])
                  for k in networks}

    # Create a dictionary with the paths of temporary files
    raw_video_path = file_mp4.parent
    tdir = raw_video_path.joinpath(f'dlc_tmp{file_mp4.name[:-4]}')
    tdir.mkdir(exist_ok=True)
    tfile = {k: tdir.joinpath(file_mp4.name.replace('.raw.', f'.{k}.')) for k in networks}
    tfile['mp4_sub'] = tdir / file_mp4.name.replace('.raw.', '.subsampled.')

    return file_mp4, dlc_params, networks, tdir, tfile, file_label


def _get_crop_window(file_df_crop: Path, network: dict) -> list:
    """Get average position of a pivot point for autocropping.

    Args:
        file_df_crop: path to dataframe from hdf5 file from video data
        network: dictionary describing the network; see constants SIDE and BODY

    Returns:
        list of floats [width, height, x, y] defining window used for ffmpeg crop command
    """
    df_crop = pd.read_hdf(file_df_crop)
    XYs = []
    for part in network['features']:
        x_values = df_crop[(df_crop.keys()[0][0], part, 'x')].values
        y_values = df_crop[(df_crop.keys()[0][0], part, 'y')].values
        likelyhoods = df_crop[(df_crop.keys()[0][0], part, 'likelihood')].values

        mx = np.ma.masked_where(likelyhoods < 0.9, x_values)
        x = np.ma.compressed(mx)
        my = np.ma.masked_where(likelyhoods < 0.9, y_values)
        y = np.ma.compressed(my)

        XYs.append([int(np.nanmean(x)), int(np.nanmean(y))])

    xy = np.mean(XYs, axis=0)
    return network['crop'](*xy)


def _s00_transform_rightCam(
    file_mp4: Path,
    tdir: Path,
    nframes: int,
    force: bool = False,
) -> tuple[str, bool]:
    """Flip and rotate the right cam and increase spatial resolution.

    Transforms the rightCamera video so it looks like the leftCamera video.

    Args:
        file_mp4: path to input video file
        tdir: temporary directory for intermediate files
        nframes: number of frames in the video
        force: whether to overwrite existing intermediate files

    Returns:
        tuple of (path to transformed video file, updated force flag)
    """
    file_out1 = str(Path(tdir).joinpath(str(file_mp4).replace('.raw.', '.flipped.')))
    # If flipped right cam does not exist, compute
    if os.path.exists(file_out1) and force is not True:
        _logger.info('STEP 00a Flipped rightCamera video exists, not computing.')
    else:
        _logger.info('STEP 00a START Flipping and turning rightCamera video')
        command_flip = (f'ffmpeg -nostats -y -loglevel 0 -i {file_mp4} -frames:v {nframes} -vf '
                        f'"transpose=1,transpose=1" -vf hflip {file_out1}')
        pop = _run_command(command_flip)
        if pop['process'].returncode != 0:
            _logger.error(f' DLC 0a/5: Flipping ffmpeg failed: {file_mp4} ' + pop['stderr'])
        _logger.info('STEP 00a END Flipping and turning rightCamera video')
        # Set force to true to recompute all subsequent steps
        force = True

    # If oversampled cam does not exist, compute
    file_out2 = file_out1.replace('.flipped.', '.raw.transformed.')
    if os.path.exists(file_out2) and force is not True:
        _logger.info('STEP 00b Oversampled rightCamera video exists, not computing.')
    else:
        _logger.info('STEP 00b START Oversampling rightCamera video')
        command_upsample = (f'ffmpeg -nostats -y -loglevel 0 -i {file_out1} -frames:v {nframes} '
                            f'-vf scale=1280:1024 {file_out2}')
        pop = _run_command(command_upsample)
        if pop['process'].returncode != 0:
            _logger.error(f' DLC 0b/5: Increase reso ffmpeg failed: {file_mp4}' + pop['stderr'])
        _logger.info('STEP 00b END Oversampling rightCamera video')
        # Set force to true to recompute all subsequent steps
        force = True

    return file_out2, force


def _s01_subsample(file_in: Path, file_out: Path, force: bool = False) -> tuple[Path, bool]:
    """Step 1: subsample video for detection using 500 uniformly sampled frames.

    Args:
        file_in: path to input video file
        file_out: path to write subsampled video
        force: whether to overwrite existing intermediate files

    Returns:
        tuple of (path to subsampled video file, updated force flag)
    """
    file_in = Path(file_in)
    file_out = Path(file_out)

    if file_out.exists() and force is not True:
        _logger.info(f"STEP 01 Sparse frame video {file_out.name} exists, not computing")
    else:
        _logger.info(f"STEP 01 START Generating sparse video {file_out.name} for posture"
                     f" detection")
        cap = cv2.VideoCapture(str(file_in))
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # get from 20 to 500 samples linearly spaced throughout the session
        nsamples = min(max(20, int(frameCount / cap.get(cv2.CAP_PROP_FPS))), 500)
        samples = np.int32(np.round(np.linspace(0, frameCount - 1, nsamples)))

        size = (int(cap.get(3)), int(cap.get(4)))
        out = cv2.VideoWriter(str(file_out), cv2.VideoWriter_fourcc(*'mp4v'), 5, size)
        for i in samples:
            cap.set(1, i)
            _, frame = cap.read()
            out.write(frame)
        out.release()
        _logger.info(f"STEP 01 END Generating sparse video {file_out.name} for posture detection")
        # Set force to true to recompute all subsequent steps
        force = True

    return file_out, force


def _s02_detect_rois(
    tdir: Path,
    sparse_video: Path,
    dlc_params: dict,
    create_labels: bool = False,
    force: bool = False,
) -> tuple[Path | None, bool]:
    """Step 2: run DLC to detect ROIs.

    Args:
        tdir: temporary directory for intermediate files
        sparse_video: path to temporally subsampled video
        dlc_params: mapping of network name to DLC config file path
        create_labels: whether to create labeled videos for debugging
        force: whether to overwrite existing intermediate files

    Returns:
        tuple of (path to dataframe used to crop video or None, updated force flag)
    """
    file_out = next(tdir.glob('*subsampled*.h5'), None)
    if file_out is None or force is True:
        _logger.info(f"STEP 02 START Posture detection for {sparse_video.name}")
        out = deeplabcut.analyze_videos(dlc_params['roi_detect'], [str(sparse_video)])
        if create_labels:
            deeplabcut.create_labeled_video(dlc_params['roi_detect'], [str(sparse_video)])
        file_out = next(tdir.glob(f'*{out}*.h5'), None)
        _logger.info(f"STEP 02 END Posture detection for {sparse_video.name}")
        # Set force to true to recompute all subsequent steps
        force = True
    else:
        _logger.info(f"STEP 02 Posture detection for {sparse_video.name} exists, not computing.")
    return file_out, force


def _s03_crop_videos(
    file_df_crop: Path,
    file_in: Path,
    file_out: Path,
    network: dict,
    nframes: int,
    force: bool = False,
) -> tuple[Path, bool]:
    """Step 3: crop videos using ffmpeg.

    Args:
        file_df_crop: path to dataframe with ROI detection results
        file_in: path to input video file
        file_out: path to write cropped video
        network: dictionary describing the network; see constants SIDE and BODY
        nframes: number of frames in the video
        force: whether to overwrite existing intermediate files

    Returns:
        tuple of (path to cropped video file, updated force flag)
    """
    # Don't run if outputs exist and force is False
    file_out = Path(file_out)
    whxy_file = file_out.parent.joinpath(file_out.stem + '.whxy.npy')
    if file_out.exists() and whxy_file.exists() and force is not True:
        _logger.info(f'STEP 03 Cropped video {file_out.name} exists, not computing.')
    else:
        _logger.info(f'STEP 03 START generating cropped video {file_out.name}')
        crop_command = (
            'ffmpeg -nostats -y -loglevel 0  -i {file_in} -frames:v {nframes}'
            ' -vf "crop={w[0]}:{w[1]}:{w[2]}:{w[3]}" -c:v libx264 -crf 11 -c:a copy {file_out}'
        )
        whxy = _get_crop_window(file_df_crop, network)
        pop = _run_command(
            crop_command.format(file_in=file_in, file_out=file_out, nframes=nframes, w=whxy)
        )
        if pop['process'].returncode != 0:
            _logger.error(f'DLC 3/6: Cropping ffmpeg failed for ROI \
                          {network["label"]}, file: {file_in}')
        np.save(str(whxy_file), whxy)
        _logger.info(f'STEP 03 END generating cropped video {file_out.name}')
        # Set force to true to recompute all subsequent steps
        force = True
    return file_out, force


def _s04_brightness_eye(file_in: Path, nframes: int, force: bool = False) -> tuple[Path, bool]:
    """Step 4a: adjust brightness for eye video for better network performance.

    Args:
        file_in: path to input eye video file
        nframes: number of frames in the video
        force: whether to overwrite existing intermediate files

    Returns:
        tuple of (path to brightness-adjusted video file, updated force flag)
    """
    # This function renames the input to 'eye.nobright' and then saves the adjusted
    # output under the same name as the original input. Therefore:
    file_in = Path(file_in)
    file_out = file_in.parent.joinpath(file_in.name.replace('eye', 'eye_adjusted'))

    if file_out.exists() and force is not True:
        _logger.info(f'STEP 04 Adjusted eye brightness {file_out.name} exists, not computing')
    else:
        # Else run command
        _logger.info(f'STEP 04 START Generating adjusting eye brightness video {file_out.name}')
        cmd = (f'ffmpeg -nostats -y -loglevel 0 -i {file_in} -frames:v {nframes} -vf '
               f'colorlevels=rimax=0.25:gimax=0.25:bimax=0.25 -c:a copy {file_out}')
        pop = _run_command(cmd)
        if pop['process'].returncode != 0:
            _logger.error(f"DLC 4/6: Adjust eye brightness failed: {file_in}")
        _logger.info(f'STEP 04 END Generating adjusting eye brightness video {file_out.name}')
        # Set force to true to recompute all subsequent steps
        force = True
    return file_out, force


def _s04_resample_paws(
    file_in: Path,
    tdir: Path,
    nframes: int,
    force: bool = False,
) -> tuple[Path, bool]:
    """Step 4b: spatially downsample paws video to speed up processing.

    Args:
        file_in: path to input video file
        tdir: temporary directory for intermediate files
        nframes: number of frames in the video
        force: whether to overwrite existing intermediate files

    Returns:
        tuple of (path to downsampled video file, updated force flag)
    """
    file_in = Path(file_in)
    file_out = Path(tdir) / file_in.name.replace('raw', 'paws_downsampled')

    if file_out.exists() and force is not True:
        _logger.info(f'STEP 04 resampled paws {file_out.name} exists, not computing')
    else:
        _logger.info(f'STEP 04 START generating resampled paws video {file_out.name}')
        cmd = (
            f'ffmpeg -nostats -y -loglevel 0 -i {file_in} -frames:v {nframes}'
            f' -vf scale=128:102 -c:v libx264 -crf 23 -c:a copy {file_out}'  # was 112:100
        )
        pop = _run_command(cmd)
        if pop['process'].returncode != 0:
            _logger.error(f"DLC 4/6: Subsampling paws failed: {file_in}")
        _logger.info(f'STEP 04 END generating resampled paws video {file_out.name}')
        # Set force to true to recompute all subsequent steps
        force = True
    return file_out, force


def _s05_run_dlc_specialized_networks(
    dlc_params: Path,
    tfile: Path,
    network: str,
    create_labels: bool = False,
    force: bool = True,
) -> None:
    """Step 5: run a specialized DLC network on a pre-processed video.

    Args:
        dlc_params: path to DLC config file for this network
        tfile: path to pre-processed video file for this network
        network: network label (e.g. 'eye', 'tongue')
        create_labels: whether to create labeled videos for debugging
        force: whether to overwrite existing intermediate files
    """
    # Check if final result exists
    result = next(tfile.parent.glob(f'*{network}*filtered.h5'), None)
    if result and force is not True:
        _logger.info(f'STEP 05 dlc feature for {tfile.name} already extracted, not computing.')
    else:
        _logger.info(f'STEP 05 START extract dlc feature for {tfile.name}')
        deeplabcut.analyze_videos(str(dlc_params), [str(tfile)])
        if create_labels:
            deeplabcut.create_labeled_video(str(dlc_params), [str(tfile)])
        deeplabcut.filterpredictions(str(dlc_params), [str(tfile)])
        _logger.info(f'STEP 05 END extract dlc feature for {tfile.name}')
        # Set force to true to recompute all subsequent steps
        force = True
    return


def _s06_extract_dlc_alf(
    tdir: Path,
    file_label: str,
    networks: dict,
    file_mp4: Path,
    *args,
) -> Path:
    """Step 6: collect all DLC outputs into a single ALF parquet file.

    Output matrix has shape [nframes, nfeatures] with column names from DLC results.

    Args:
        tdir: temporary directory containing per-network DLC result files
        file_label: camera label string used to name the output file
        networks: mapping of network name to network parameter dict
        file_mp4: path to original video file

    Returns:
        path to the output ALF parquet file
    """
    _logger.info(f'STEP 06 START wrap-up and extract ALF files {file_label}')
    if 'bodyCamera' in file_label:
        video_params = BODY_VIDEO
    elif 'leftCamera' in file_label:
        video_params = LEFT_VIDEO
    elif 'rightCamera' in file_label:
        video_params = RIGHT_VIDEO

    # Create alf path to store final files
    alf_path = tdir.parent.parent.joinpath('alf')
    alf_path.mkdir(exist_ok=True, parents=True)

    for roi in networks:
        if networks[roi]['features'] is None:
            continue
        # we need to make sure we use filtered traces
        df = pd.read_hdf(next(tdir.glob(f'*{roi}*filtered.h5')))
        if roi == 'paws':
            whxy = [0, 0, 0, 0]
        else:
            whxy = np.load(next(tdir.glob(f'*{roi}*.whxy.npy')))

        # Simplify the indices of the multi index df
        columns = [f'{c[1]}_{c[2]}' for c in df.columns.to_flat_index()]
        df.columns = columns

        # translate and scale the specialized window in the full initial frame
        post_crop_scale = networks[roi]['postcrop_downsampling']
        pre_crop_scale = 1.0 / video_params['sampling']
        for ind in columns:
            if ind[-1] == 'x':
                df[ind] = df[ind].apply(lambda x: (x * post_crop_scale + whxy[2]) * pre_crop_scale)
                if video_params['flip']:
                    df[ind] = df[ind].apply(lambda x: video_params['original_size'][0] - x)
            elif ind[-1] == 'y':
                df[ind] = df[ind].apply(lambda x: (x * post_crop_scale + whxy[3]) * pre_crop_scale)

        # concatenate this in one dataframe for all networks
        if 'df_full' not in locals():
            df_full = df.copy()
        else:
            df_full = pd.concat([df_full.reset_index(drop=True),
                                 df.reset_index(drop=True)], axis=1)

    # save in alf path
    file_alf = alf_path.joinpath(f'_ibl_{file_label}.dlc.pqt')
    df_full.to_parquet(file_alf)

    _logger.info(f'STEP 06 END wrap-up and extract ALF files {file_label}')
    return file_alf


def dlc(
    file_mp4: str | Path,
    path_dlc: Path | None = None,
    force: bool = False,
    dlc_timer: OrderedDict | None = None,
) -> tuple[Path, OrderedDict]:
    """Analyse a leftCamera, rightCamera or bodyCamera video with DeepLabCut.

    The process consists of 7 steps:
    0. check if rightCamera video, then make it look like leftCamera video
    1. subsample video frames using ffmpeg
    2. run DLC to detect ROIs: 'eye', 'nose_tip', 'tongue', 'paws'
    3. crop videos for each ROI using ffmpeg, subsample paws videos
    4. downsample the paw videos
    5. run DLC specialized networks on each ROI
    6. output ALF dataset for the raw DLC output

    Args:
        file_mp4: video file to run
        path_dlc: path to folder with DLC weights
        force: whether to overwrite existing intermediate files
        dlc_timer: ordered dict to accumulate per-step timing; created if None

    Returns:
        tuple of (path to DLC table in parquet file format, timing dict)
    """
    # Set up timing
    dlc_timer = dlc_timer or OrderedDict()
    time_total_on = time.time()
    # Initiate
    file_mp4, dlc_params, networks, tdir, tfile, file_label = _dlc_init(file_mp4, path_dlc)
    file_meta = get_video_meta(file_mp4)
    nframes = file_meta['length']

    # Run the processing steps in order
    if 'rightCamera' not in file_mp4.name:
        file2segment = file_mp4
    else:
        time_on = time.time()
        # CPU Python
        file2segment, force = _s00_transform_rightCam(file_mp4, tdir, nframes, force=force)
        time_off = time.time()
        dlc_timer['Transform right camera'] = time_off - time_on

    time_on = time.time()
    file_sparse, force = _s01_subsample(file2segment, tfile['mp4_sub'], force=force)  # CPU ffmpeg
    time_off = time.time()
    dlc_timer['Subsample video'] = time_off - time_on

    time_on = time.time()
    file_df_crop, force = _s02_detect_rois(tdir, file_sparse, dlc_params, force=force)  # GPU dlc
    time_off = time.time()
    dlc_timer['Detect ROIs'] = time_off - time_on

    input_force = force
    for k in networks:
        time_on = time.time()
        if networks[k]['features'] is None:
            continue
        # Run preprocessing depending on the feature
        if k == 'paws':
            preproc_vid, force = _s04_resample_paws(file2segment, tdir, nframes, force=force)
        elif k == 'eye':
            cropped_vid, force = _s03_crop_videos(file_df_crop, file2segment, tfile[k],
                                                  networks[k], nframes, force=force)
            preproc_vid, force = _s04_brightness_eye(cropped_vid, nframes, force=force)
        else:
            preproc_vid, force = _s03_crop_videos(file_df_crop, file2segment, tfile[k],
                                                  networks[k], nframes, force=force)
        time_off = time.time()
        dlc_timer[f'Prepare video for {k} network'] = time_off - time_on

        time_on = time.time()
        # Allows manually setting force to true but default to rerunning this for safety
        _s05_run_dlc_specialized_networks(dlc_params[k], preproc_vid, k)
        time_off = time.time()
        dlc_timer[f'Run {k} network'] = time_off - time_on
        # Reset force to the original input value as the reset is network-specific
        force = input_force

    out_file = _s06_extract_dlc_alf(tdir, file_label, networks, file_mp4)

    # at the end mop up the mess
    # For right camera video only
    file2segment = Path(file2segment)
    if '.raw.transformed' in file2segment.name:
        file2segment.unlink()
        flipped = Path(str(file2segment).replace('raw.transformed', 'flipped'))
        flipped.unlink()

    shutil.rmtree(tdir)
    # Back to home folder else there  are conflicts in a loop
    os.chdir(Path.home())
    print(file_label)
    time_total_off = time.time()
    dlc_timer['DLC total'] = time_total_off - time_total_on

    return out_file, dlc_timer


def _run_command(command: str) -> dict:
    """Run a shell command using subprocess.

    Args:
        command: shell command string to run

    Returns:
        dict with keys 'process', 'stdout', and 'stderr'
    """
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    info, error = process.communicate()
    return {
        'process': process,
        'stdout': info.decode(),
        'stderr': error.decode(),
    }
