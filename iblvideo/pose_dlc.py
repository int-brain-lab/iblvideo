"""Functions to run DLC on IBL data with existing networks."""
import deeplabcut  # needs to be imported first
import os
import shutil
import logging
import yaml
import time
from glob import glob
from pathlib import Path
from collections import OrderedDict

import numpy as np
import pandas as pd
import cv2

from iblvideo.params_dlc import BODY_FEATURES, SIDE_FEATURES, LEFT_VIDEO, RIGHT_VIDEO, BODY_VIDEO
from iblvideo.utils import _run_command
from ibllib.io.video import get_video_meta

_logger = logging.getLogger('ibllib')


def _set_dlc_paths(path_dlc):
    """Replace hard-coded paths in the config.yaml file."""
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


def _dlc_init(file_mp4, path_dlc):
    """Prepare inputs and create temporary filenames."""
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


def _get_crop_window(file_df_crop, network):
    """
    Get average position of a pivot point for autocropping.
    :param file_df_crop: Path to data frame from hdf5 file from video data
    :param network: dictionary describing the networks.
                    See constants SIDE and BODY
    :return: list of floats [width, height, x, y] defining window used for
             ffmpeg crop command
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


def _s00_transform_rightCam(file_mp4, tdir, nframes, force=False):
    """
    Flip and rotate the right cam and increase spatial resolution.
    Such that the rightCamera video looks like the leftCamera video.
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


def _s01_subsample(file_in, file_out, force=False):
    """
    Step 1 subsample video for detection.
    Put 500 uniformly sampled frames into new video.
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


def _s02_detect_rois(tdir, sparse_video, dlc_params, create_labels=False, force=False):
    """
    Step 2 run DLC to detect ROIS.
    returns: Path to dataframe used to crop video
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


def _s03_crop_videos(file_df_crop, file_in, file_out, network, nframes, force=False):
    """
    Step 3 crop videos using ffmpeg.
    returns: dictionary of cropping coordinates relative to upper left corner
    """
    # Don't run if outputs exist and force is False
    file_out = Path(file_out)
    whxy_file = file_out.parent.joinpath(file_out.stem + '.whxy.npy')
    if file_out.exists() and whxy_file.exists() and force is not True:
        _logger.info(f'STEP 03 Cropped video {file_out.name} exists, not computing.')
    else:
        _logger.info(f'STEP 03 START generating cropped video {file_out.name}')
        crop_command = ('ffmpeg -nostats -y -loglevel 0  -i {file_in} -frames:v {nframes} -vf "crop={w[0]}:{w[1]}:'
                        '{w[2]}:{w[3]}" -c:v libx264 -crf 11 -c:a copy {file_out}')
        whxy = _get_crop_window(file_df_crop, network)
        pop = _run_command(crop_command.format(file_in=file_in, file_out=file_out, nframes=nframes, w=whxy))
        if pop['process'].returncode != 0:
            _logger.error(f'DLC 3/6: Cropping ffmpeg failed for ROI \
                          {network["label"]}, file: {file_in}')
        np.save(str(whxy_file), whxy)
        _logger.info(f'STEP 03 END generating cropped video {file_out.name}')
        # Set force to true to recompute all subsequent steps
        force = True
    return file_out, force


def _s04_brightness_eye(file_in, nframes, force=False):
    """
    Adjust brightness for eye for better network performance.
    wget -O- http://ffmpeg.org/releases/ffmpeg-snapshot.tar.bz2 | tar xj
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


def _s04_resample_paws(file_in, tdir, nframes, force=False):
    """For paws, spatially downsample to speed up processing x100."""
    file_in = Path(file_in)
    file_out = Path(tdir) / file_in.name.replace('raw', 'paws_downsampled')

    if file_out.exists() and force is not True:
        _logger.info(f'STEP 04 resampled paws {file_out.name} exists, not computing')
    else:
        _logger.info(f'STEP 04 START generating resampled paws video {file_out.name}')
        cmd = (f'ffmpeg -nostats -y -loglevel 0 -i {file_in} -frames:v {nframes} -vf scale=128:102 -c:v libx264 -crf 23'
               f' -c:a copy {file_out}')  # was 112:100
        pop = _run_command(cmd)
        if pop['process'].returncode != 0:
            _logger.error(f"DLC 4/6: Subsampling paws failed: {file_in}")
        _logger.info(f'STEP 04 END generating resampled paws video {file_out.name}')
        # Set force to true to recompute all subsequent steps
        force = True
    return file_out, force


def _s05_run_dlc_specialized_networks(dlc_params, tfile, network, create_labels=False,
                                      force=True):

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


def _s06_extract_dlc_alf(tdir, file_label, networks, file_mp4, *args):
    """
    Output an ALF matrix.
    Column names contain the full DLC results [nframes, nfeatures]
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


def dlc(file_mp4, path_dlc=None, force=False, dlc_timer=None):
    """
    Analyse a leftCamera, rightCamera or bodyCamera video with DeepLabCut.

    The process consists in 7 steps:
    0- Check if rightCamera video, then make it look like leftCamera video
    1- subsample video frames using ffmpeg
    2- run DLC to detect ROIS: 'eye', 'nose_tip', 'tongue', 'paws'
    3- crop videos for each ROIs using ffmpeg, subsample paws videos
    4- downsample the paw videos
    5- run DLC specialized networks on each ROIs
    6- output ALF dataset for the raw DLC output

    :param file_mp4: Video file to run
    :param path_dlc: Path to folder with DLC weights
    :param force: bool, whether to overwrite existing intermediate files
    :return out_file: Path to DLC table in parquet file format
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
        file2segment, force = _s00_transform_rightCam(file_mp4, tdir, nframes, force=force)  # CPU Python
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


# def dlc_parallel(file_mp4, path_dlc=None, force=False):
#     """
#     Run dlc in parallel.
#
#     :param file_mp4: Video file to run
#     :param path_dlc: Path to folder with DLC weights
#     :return out_file: Path to DLC table in parquet file format
#     """
#
#     import dask
#     cluster, client = create_cpu_gpu_cluster()
#
#     start_T = time.time()
#     file_mp4, dlc_params, networks, tdir, tfile, file_label = _dlc_init(file_mp4, path_dlc)
#
#     # Run the processing steps in order
#     future_s00 = client.submit(_s00_transform_rightCam, file_mp4, tdir, workers='CPU')
#     file2segment = future_s00.result()
#
#     future_s01 = client.submit(_s01_subsample, file2segment, tfile['mp4_sub'], workers='CPU')
#     file_sparse = future_s01.result()
#
#     future_s02 = client.submit(_s02_detect_rois, tdir, file_sparse, dlc_params, workers='GPU')
#     df_crop = future_s02.result()
#
#     networks_run = {}
#     for k in networks:
#         if networks[k]['features'] is None:
#             continue
#         if k == 'paws':
#             future_s04 = client.submit(_s04_resample_paws, file2segment, tdir, workers='CPU')
#             cropped_vid = future_s04.result()
#         elif k == 'eye':
#             future_s03 = client.submit(_s03_crop_videos, df_crop, file2segment, tfile[k],
#                                        networks[k], workers='CPU')
#             cropped_vid_a = future_s03.result()
#             future_s04 = client.submit(_s04_brightness_eye, cropped_vid_a, workers='CPU')
#             cropped_vid = future_s04.result()
#         else:
#             future_s03 = client.submit(_s03_crop_videos, df_crop, file2segment, tfile[k],
#                                        networks[k], workers='CPU')
#             cropped_vid = future_s03.result()
#
#         future_s05 = client.submit(_s05_run_dlc_specialized_networks, dlc_params[k], cropped_vid,
#                                    networks[k], workers='GPU')
#         network_run = future_s05.result()
#         networks_run[k] = network_run
#
#     pipeline = dask.delayed(_s06_extract_dlc_alf)(tdir, file_label, networks_run, file_mp4)
#     future = client.compute(pipeline)
#     out_file = future.result()
#
#     cluster.close()
#     client.close()
#
#     shutil.rmtree(tdir)
#
#     # Back to home folder else there  are conflicts in a loop
#     os.chdir(Path.home())
#     end_T = time.time()
#     print(file_label)
#     print('In total this took: ', end_T - start_T)
#
#     return out_file
