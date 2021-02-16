"""Functions to run DLC on IBL data with existing networks."""
import deeplabcut
import os
import shutil
import logging
import subprocess
import json
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from dlc_params import BODY_FEATURES, SIDE_FEATURES, LEFT_VIDEO, \
                        RIGHT_VIDEO, BODY_VIDEO

_logger = logging.getLogger('ibllib')


def _run_command(command):
    """
    Run a shell command using subprocess.

    :param command: command to run
    :return: dictionary with keys: process, stdout, stderr
    """
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    info, error = process.communicate()
    return {
        'process': process,
        'stdout': info.decode(),
        'stderr': error.decode()}


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


def _get_crop_window(df_crop, network):
    """
    Get average position of a pivot point for autocropping.

    h5 is local file name only;

    :param df_crop: data frame from hdf5 file from video data
    :param network: dictionary describing the networks.
                    See constants SIDE and BODY
    :return: list of floats [width, height, x, y] defining window used for
             ffmpeg crop command
    """
    # choose parts to find pivot point which is used to crop around a ROI

    XYs = []
    for part in network['features']:
        x_values = df_crop[(df_crop.keys()[0][0], part, 'x')].values
        y_values = df_crop[(df_crop.keys()[0][0], part, 'y')].values
        likelyhoods = df_crop[(df_crop.keys()[0][0],
                               part, 'likelihood')].values

        mx = np.ma.masked_where(likelyhoods < 0.9, x_values)
        x = np.ma.compressed(mx)
        my = np.ma.masked_where(likelyhoods < 0.9, y_values)
        y = np.ma.compressed(my)

        XYs.append([int(np.nanmean(x)), int(np.nanmean(y))])

    xy = np.mean(XYs, axis=0)

    return network['crop'](*xy)


def _s00_transform_rightCam(file_mp4):
    """
    Flip and rotate the right cam and increase spatial resolution.

    Such that the rightCamera video looks like the leftCamera video.
    """
    # TODO use the video parameters above not to have this hard-coded
    # (sampling + original size)
    if 'rightCamera' not in file_mp4.name:
        return file_mp4

    else:
        _logger.info('STEP 00 Flipping and turning rightCamera video')
        file_out1 = str(file_mp4).replace('.raw.', '.flipped.')
        command_flip = (f'ffmpeg -nostats -y -loglevel 0 -i {file_mp4} -vf '
                        f'"transpose=1,transpose=1" -vf hflip {file_out1}')
        pop = _run_command(command_flip)
        if pop['process'].returncode != 0:
            _logger.error(f' DLC 0a/5: Flipping ffmpeg failed: {file_mp4}'
                          + pop['stderr'])
        _logger.info('Oversampling rightCamera video')
        file_out2 = file_out1.replace('.flipped.', '.raw.transformed.')

        command_upsample = (f'ffmpeg -nostats -y -loglevel 0 -i {file_out1} '
                            f'-vf scale=1280:1024 {file_out2}')
        pop = _run_command(command_upsample)
        if pop['process'].returncode != 0:
            _logger.error(f' DLC 0b/5: Increase reso ffmpeg failed: {file_mp4}'
                          + pop['stderr'])
        Path(file_out1).unlink()
        _logger.info('STEP 00 STOP Flipping and turning rightCamera video')

        return file_out2


def _s01_subsample(file_in, file_out, force=False):
    """
    Step 1 subsample video for detection.

    Put 500 uniformly sampled frames into new video.
    """
    _logger.info(f"STEP 01 Generating sparse frame video {file_out} \
                 for posture detection")
    file_in = Path(file_in)
    file_out = Path(file_out)

    if file_out.exists() and not force:
        return file_out

    cap = cv2.VideoCapture(str(file_in))
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # get from 20 to 500 samples linearly spaced throughout the session
    nsamples = min(max(20, frameCount / cap.get(cv2.CAP_PROP_FPS)), 500)
    samples = np.int32(np.round(np.linspace(0, frameCount - 1, nsamples)))

    size = (int(cap.get(3)), int(cap.get(4)))
    out = cv2.VideoWriter(str(file_out), cv2.VideoWriter_fourcc(*'mp4v'),
                          5, size)
    for i in samples:
        cap.set(1, i)
        _, frame = cap.read()
        out.write(frame)

    out.release()
    _logger.info(f"STEP 01 STOP Generating sparse frame video {file_out} \
                 for posture detection")
    return file_out


def _s02_detect_rois(tpath, sparse_video, dlc_params, create_labels=False):
    """
    Step 2 run DLC to detect ROIS.

    returns: df_crop, dataframe used to crop video
    """
    _logger.info(f"STEP 02 START Posture detection {sparse_video}")
    out = deeplabcut.analyze_videos(dlc_params['roi_detect'],
                                    [str(sparse_video)])
    if create_labels:
        deeplabcut.create_labeled_video(dlc_params['roi_detect'],
                                        [str(sparse_video)])
    h5_sub = next(tpath.glob(f'*{out}*.h5'), None)
    _logger.info(f"STEP 02 END Posture detection {sparse_video}")
    return pd.read_hdf(h5_sub)


def _s03_crop_videos(df_crop, file_in, file_out, network):
    """
    Step 3 crop videos using ffmpeg.

    returns: dictionary of cropping coordinates relative to upper left corner
    """
    _logger.info(f'STEP 03 START cropping {network["label"]}  video')
    crop_command = (
        'ffmpeg -nostats -y -loglevel 0  -i {file_in} -vf "crop={w[0]}:{w[1]}:'
        '{w[2]}:{w[3]}" -c:v libx264 -crf 11 -c:a copy {file_out}')
    whxy = _get_crop_window(df_crop, network)
    pop = _run_command(crop_command.format(file_in=file_in, file_out=file_out,
                                           w=whxy))
    if pop['process'].returncode != 0:
        _logger.error(f'DLC 3/6: Cropping ffmpeg failed for ROI \
                      {network["label"]}, file: {file_in}')
    np.save(file_out.parent.joinpath(file_out.stem + '.whxy.npy'), whxy)
    _logger.info(f'STEP 03 STOP cropping {network["label"]}  video')
    return file_out


def _s04_brightness_eye(file_mp4, force=False):
    """
    Adjust brightness for eye for better network performance.

    wget -O- http://ffmpeg.org/releases/ffmpeg-snapshot.tar.bz2 | tar xj
    """
    file_out = file_mp4
    file_in = file_mp4.parent.joinpath(file_mp4.name.replace('eye',
                                                             'eye.nobright'))
    if file_in.exists() and not force:
        return file_out
    _logger.info('STEP 04 START Adjusting eye brightness')
    file_out.rename(file_in)
    cmd = (f'ffmpeg -nostats -y -loglevel 0 -i {file_in} -vf '
           f'colorlevels=rimax=0.25:gimax=0.25:bimax=0.25 \
           -c:a copy {file_out}')
    pop = _run_command(cmd)
    if pop['process'].returncode != 0:
        _logger.error(f"DLC 4/6: (str(dlc_params), [str(tfile)]) failed: \
                      {file_in}")
    _logger.info('STEP 04 STOP Adjusting eye brightness')
    return file_out


def _s04_resample_paws(file_mp4, tdir, force=False):
    """For paws, spatially downsample to speed up processing x100."""
    file_mp4 = Path(file_mp4)
    file_in = file_mp4
    file_out = Path(tdir) / file_mp4.name.replace('raw', 'paws_downsampled')

    cmd = (f'ffmpeg -nostats -y -loglevel 0 -i {file_in} -vf scale=128:102 \
           -c:v libx264 -crf 23'
           f' -c:a copy {file_out}')  # was 112:100
    pop = _run_command(cmd)
    if pop['process'].returncode != 0:
        _logger.error(f"DLC 4/6: Subsampling paws failed: {file_in}")
    _logger.info('STEP 04 STOP resample paws')
    return file_out


def _s05_run_dlc_specialized_networks(dlc_params, tfile, create_labels=False):
    _logger.info(f'STEP 05 START extract dlc feature {tfile}')
    deeplabcut.analyze_videos(str(dlc_params), [str(tfile)])
    if create_labels:
        deeplabcut.create_labeled_video(str(dlc_params), [str(tfile)])
    deeplabcut.filterpredictions(str(dlc_params), [str(tfile)])
    _logger.info(f'STEP 05 STOP extract dlc feature {tfile}')
    return


def _s06_extract_dlc_alf(tdir, file_label, networks, file_mp4, *args):
    """
    Output an ALF matrix.

    Column names contain the full DLC results [nframes, nfeatures]
    """
    _logger.info('STEP 06 START wrap-up and extract ALF files')
    if 'bodyCamera' in file_label:
        video_params = BODY_VIDEO
    elif 'leftCamera' in file_label:
        video_params = LEFT_VIDEO
    elif 'rightCamera' in file_label:
        video_params = RIGHT_VIDEO

    raw_video_path = tdir.parent
    columns = []
    for roi in networks:
        if networks[roi]['features'] is None:
            continue
        # we need to make sure we use filtered traces
        df = pd.read_hdf(next(tdir.glob(f'*{roi}*filtered.h5')))
        if roi == 'paws':
            whxy = [0, 0, 0, 0]
        else:
            whxy = np.load(next(tdir.glob(f'*{roi}*.whxy.npy')))
        # get the indices of this multi index hierarchical thing
        # translate and scale the specialized window in the full initial frame
        indices = df.columns.to_flat_index()
        post_crop_scale = networks[roi]['postcrop_downsampling']
        pre_crop_scale = 1.0 / video_params['sampling']
        for ind in indices:
            if ind[-1] == 'x':
                df[ind] = df[ind].apply(lambda x:
                                        (x * post_crop_scale + whxy[2])
                                        * pre_crop_scale)
                if video_params['flip']:
                    df[ind] = df[ind].apply(lambda x:
                                            video_params['original_size'][0]
                                            - x)
            elif ind[-1] == 'y':
                df[ind] = df[ind].apply(lambda x:
                                        (x * post_crop_scale + whxy[3])
                                        * pre_crop_scale)
        # concatenate this in a flat matrix
        columns.extend([f'{c[1]}_{c[2]}' for c in df.columns.to_flat_index()])
        if 'A' not in locals():
            A = np.zeros([df.shape[0], 0], np.float)
        A = np.c_[A, np.array(df).astype(np.float)]
    assert (len(columns) == A.shape[1])

    # write the ALF files without depending on ibllib
    file_alf_dlc = raw_video_path.joinpath(f'_ibl_{file_label}.dlc.npy')
    file_meta_data = raw_video_path.joinpath(f'_ibl_\
                                             {file_label}.dlc.metadata.json')

    np.save(file_alf_dlc, A)
    with open(file_meta_data, 'w+') as fid:
        fid.write(json.dumps({'columns': columns}, indent=1))

    _logger.info('STEP 06 STOP wrap-up and extract ALF files')
    return file_alf_dlc, file_meta_data


def dlc(file_mp4, path_dlc=None, force=False, parallel=False):
    """
    Analyse a leftCamera, rightCamera or bodyCamera video with DeepLabCut.

    The process consists in 7 steps:
    0- Check if rightCamera video, then make it look like leftCamera video
    1- subsample video frames using ffmpeg
    2- run DLC to detect ROIS: 'eye', 'nose_tip', 'tongue', 'paws'
    3- crop videos for each ROIs using ffmpeg, subsample paws videos
    4- downsample the paw videos
    5- run DLC specialized networks on each ROIs
    6- output ALF dataset for the raw DLC output in
       ./session/alf/_ibl_leftCamera.dlc.json

    # This is a directory tree of the temporary files created
    # ./raw_video_data/dlc_tmp/  # tpath: temporary path
    #   _iblrig_leftCamera.raw.mp4'  # file_mp4
    #   _iblrig_leftCamera.subsampled.mp4  # file_temp['mp4_sub']
    #   _iblrig_leftCamera.subsampledDeepCut_resnet50_trainingRigFeb11shuffle1_550000.h5
    # tfile['h5_sub']
    #   _iblrig_leftCamera.eye.mp4 # tfile['eye']
    #   _iblrig_leftCamera.nose_tip.mp4 # tfile['nose_tip']
    #   _iblrig_leftCamera.tongue.mp4 # tfile['tongue']
    #   _iblrig_leftCamera.pose.mp4 # tfile['pose']

    :param file_mp4: file to run
    :return: None
    """
    # Process inputs
    file_mp4 = Path(file_mp4)  # _iblrig_leftCamera.raw.mp4
    file_label = file_mp4.stem.split('.')[0].split('_')[-1]
    if 'bodyCamera' in file_label:
        networks = BODY_FEATURES
    else:
        networks = SIDE_FEATURES

    path_dlc = Path(path_dlc)
    _set_dlc_paths(path_dlc)
    dlc_params = {k: path_dlc.joinpath(networks[k]['weights'],
                  'config.yaml') for k in networks}

    # Create a dictionary with the paths of temporary files
    raw_video_path = file_mp4.parent
    tdir = raw_video_path.joinpath(f'dlc_tmp_{file_mp4.name[:-4]}')
    tdir.mkdir(exist_ok=True)
    tfile = {k: tdir.joinpath(file_mp4.name.replace('.raw.', f'.{k}.'))
             for k in networks}
    tfile['mp4_sub'] = tdir / file_mp4.name.replace('.raw.', '.subsampled.')

    # Run the processing steps in order
    file2segment = _s00_transform_rightCam(file_mp4)  # CPU pure Python
    file_sparse = _s01_subsample(file2segment, tfile['mp4_sub'])  # CPU ffmpeg
    df_crop = _s02_detect_rois(tdir, file_sparse, dlc_params)   # GPU dlc

    for k in networks:
        if networks[k]['features'] is None:
            continue
        if k == 'paws':
            cropped_vid = file2segment
        else:
            cropped_vid = _s03_crop_videos(df_crop, file2segment, tfile[k],
                                           networks[k])   # CPU ffmpeg
        if k == 'paws':
            cropped_vid = _s04_resample_paws(cropped_vid, tdir)
        if k == 'eye':
            cropped_vid = _s04_brightness_eye(cropped_vid)
        status = _s05_run_dlc_specialized_networks(dlc_params[k], cropped_vid)
        # GPU dlc

    alf_files = _s06_extract_dlc_alf(tdir, file_label,
                                     networks, file_mp4, status)

    file2segment = Path(file2segment)
    # at the end mop up the mess
    shutil.rmtree(tdir)

    if '.raw.transformed' in file2segment.name:
        file2segment.unlink()

    # Back to home folder else there  are conflicts in a loop
    os.chdir(Path.home())
    return alf_files
