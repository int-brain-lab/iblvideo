"""Helper functions to run Lightning Pose on a single IBL video with trained networks."""

import gc
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import lightning.pytorch as pl
import numpy as np
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import pandas as pd
import torch
import yaml
from eks.ibl_pupil_smoother import ensemble_kalman_smoother_ibl_pupil
from eks.singlecam_smoother import ensemble_kalman_smoother_singlecam
from eks.utils import convert_lp_dlc
from lightning_pose.data import _IMAGENET_MEAN, _IMAGENET_STD
from lightning_pose.data.dali import LitDaliWrapper
from lightning_pose.data.utils import count_frames
from lightning_pose.utils.predictions import PredictionHandler, load_model_from_checkpoint
from moviepy.editor import VideoFileClip
from nvidia.dali import pipeline_def
from nvidia.dali.plugin.pytorch import LastBatchPolicy
from omegaconf import DictConfig


def get_crop_window(roi_df_file: Path, network_params: dict, scale: int) -> list:
    """Get average position of a anchor point for autocropping.

    :param roi_df_file: path to dataframe output by ROI network
    :param network_params: parameters for network, see SIDE_FEATURES and BODY_FEATURES in params_lp.py
    :param scale: downsampling factor; >1 means reduce crop window parameters
    :return: list of floats [width, height, x, y] defining window used for cropping
    """
    df_crop = pd.read_csv(roi_df_file, header=[0, 1, 2], index_col=0)
    XYs = []
    for part in network_params['features']:
        # get x/y values wrt actual video size, not LEFT video size returned by ROI network
        x_values = df_crop[(df_crop.keys()[0][0], part, 'x')].values
        y_values = df_crop[(df_crop.keys()[0][0], part, 'y')].values
        likelihoods = df_crop[(df_crop.keys()[0][0], part, 'likelihood')].values

        mx = np.ma.masked_where(likelihoods < 0.9, x_values)
        x = np.ma.compressed(mx)
        my = np.ma.masked_where(likelihoods < 0.9, y_values)
        y = np.ma.compressed(my)

        XYs.append([int(np.nanmean(x)), int(np.nanmean(y))])

    xy = np.mean(XYs, axis=0)
    return network_params['crop'](*xy, scale)


@pipeline_def
def video_pipe_crop_resize_flip(
    # arguments for video reader:
    filenames: Union[List[str], str],
    sequence_length: int = 32,
    pad_sequences: bool = True,
    pad_last_batch: bool = False,
    step: int = 1,
    name: str = 'reader',  # arbitrary
    # arguments for frame manipulations:
    crop_params: Optional[Dict] = None,
    normalization_mean: List[float] = _IMAGENET_MEAN,
    normalization_std: List[float] = _IMAGENET_STD,
    resize_dims: Optional[List[int]] = None,
    brightness: Optional[float] = None,
    flip: bool = False,
    # arguments consumed by decorator:
    # batch_size,
    # num_threads,
    # device_id,
) -> tuple:
    """Video reader pipeline that loads videos, normalizes, crops, and optionally flips.

    :param filenames: list of absolute paths of video files to feed through pipeline
    :param sequence_length: number of frames to load per sequence
    :param pad_sequences: allows creation of incomplete sequences if there is an insufficient
        number of frames at the very end of the video
    :param pad_last_batch: pad final batch with empty sequences
    :param step: number of frames to advance on each read; will be different for context
        vs non-context models
    :param name: pipeline name, used to string together DALI DataNode elements
    :param crop_params: keys are
        - 'crop_h': height in pixels
        - 'crop_w': width in pixels
        - 'crop_pos_x': x position of top left corner; normalized in (0, 1)
        - 'crop_pos_y': y position of top left corner; normalized in (0, 1)
    :param normalization_mean: mean values in (0, 1) to subtract from each channel
    :param normalization_std: standard deviation values to divide by for each channel
    :param resize_dims: [height, width] to resize raw frames
    :param brightness: multiplicative factor to increase brightness of frames
    :param flip: True to flip frame around vertical axis
    :param batch_size: number of sequences per batch
    :param num_threads: number of cpu threads used by the pipeline
    :param device_id: id of the gpu used by the pipeline
    :return:
        pipeline object to be fed to DALIGenericIterator
        placeholder int to represent unused "transforms" field in dataloader

    """

    device = 'gpu'  # pipeline cannot run on cpu

    # read batches of video from file
    video = fn.readers.video(
        filenames=filenames,
        sequence_length=sequence_length,
        pad_sequences=pad_sequences,
        pad_last_batch=pad_last_batch,
        step=step,
        name=name,
        device=device,
        random_shuffle=False,
        initial_fill=sequence_length,
        normalized=False,
        dtype=types.DALIDataType.FLOAT,
        file_list_include_preceding_frame=True,  # to get rid of dali warnings
    )
    orig_size = video.shape(device=device)

    # original videos range [0, 255]; transform it to [0, 1] for our models
    video = (video / 255.0)

    # adjust color levels
    # brightness=4 is equivalent to ffmpeg's "colorlevels=rimax=0.25:gimax=0.25:bimax=0.25"
    if brightness:
        video = fn.coord_transform(
            video,
            M=np.array([[brightness, 0, 0], [0, brightness, 0], [0, 0, brightness]])
        )
        # clip
        mask_high = video > 1.0
        mask_low = video <= 1.0
        video = mask_low * video + mask_high * types.Constant(1.0)

    if flip:
        video = fn.flip(video, horizontal=1)

    # change channel layout, crop, and normalize according to imagenet statistics
    if crop_params:
        video = fn.crop_mirror_normalize(
            video,
            output_layout='FCHW',
            mean=normalization_mean,
            std=normalization_std,
            crop_h=crop_params['crop_h'],  # pixels
            crop_w=crop_params['crop_w'],  # pixels
            crop_pos_x=crop_params['crop_pos_x'],  # normalized in (0, 1)
            crop_pos_y=crop_params['crop_pos_y'],  # normalized in (0, 1)
        )
        # update size of frames
        orig_size = video.shape(device=device)
    else:
        video = fn.crop_mirror_normalize(
            video,
            output_layout='FCHW',
            mean=normalization_mean,
            std=normalization_std,
        )

    # resize to match accepted network sizes
    if resize_dims:
        video = fn.resize(video, size=resize_dims)

    return video, -1, orig_size


def compute_num_iters(
    video_path: str,
    sequence_length: int,
    step: int,
    model_type: str
) -> int:
    """Compute number of iterations necessary to iterate through a video.

    :param video_path: absolute path to video file
    :param sequence_length: number of frames to load per sequence
    :param step: number of frames to advance on each read
    :param model_type: 'baseline' or 'context', affects how sequence_length/step are interpreted
    :return: number of iterations
    """
    frame_count = count_frames(video_path)
    if model_type == 'baseline':
        num_iters = int(np.ceil(frame_count / sequence_length))
    elif model_type == 'context':
        if step == 1:
            num_iters = int(np.ceil(frame_count))
        elif step == (sequence_length - 4):
            if step <= 0:
                raise ValueError('step cannot be 0, please modify sequence_length to be > 4')
            # remove the first sequence
            data_except_first_batch = frame_count - sequence_length
            # calculate how many "steps" are needed to get at least to the end
            # count back the first sequence
            num_iters = int(np.ceil(data_except_first_batch / step)) + 1
        else:
            raise NotImplementedError
    else:
        raise ValueError(f'model_type must be "baseline" or "context", not {model_type}')
    return num_iters


def build_dataloader(
    network: str,
    mp4_file: str,
    model_type: str,
    sequence_length: int,
    flip: bool,
    resize_dims: list,
    original_dims: list,
    crop_window: Optional[list] = None,
) -> LitDaliWrapper:
    """Build pytorch data loader that wraps DALI pipeline.

    :param network: network name, key for `camera_params` features dict
    :param mp4_file: path to video file
    :param model_type: 'baseline' | 'context'
    :param sequence_length: number of frames to load per sequence
    :param flip: True to flip horizontally
    :param resize_dims: [height, width], resize dims for network resizing
    :param original_dims: [height, width], original dims of video
    :param crop_window: list of floats [width, height, x, y] defining window used for cropping
    :return: pytorch data loader
    """

    if model_type == 'baseline':
        # video reader args
        step = sequence_length
        # dataset iterator args
        do_context = False
    else:
        # video reader args
        step = sequence_length - 4
        # dataset iterator args
        do_context = True

    num_iters = compute_num_iters(mp4_file, sequence_length, step, model_type)
    iter_args = {
        'num_iters': num_iters,
        'eval_mode': 'predict',
        'output_map': ['frames', 'transforms', 'frame_size'],
        'last_batch_policy': LastBatchPolicy.FILL,
        'last_batch_padded': False,
        'auto_reset': False,
        'reader_name': 'reader',
        'do_context': do_context,
    }

    # set network-specific args
    if network in ['roi_detect', 'paws', 'tail_start']:

        crop_params = None
        brightness = None

    elif network in ['eye', 'tongue', 'nose_tip']:

        crop_w, crop_h, crop_x, crop_y = crop_window
        frame_height, frame_width = original_dims

        # check if crop extends beyond left/top edge
        if crop_x < 0:
            crop_x = 0
        if crop_y < 0:
            crop_y = 0

        # check if crop extends beyond right/bottom edge
        if crop_x + crop_w > frame_width:
            crop_x = frame_width - crop_w
        if crop_y + crop_h > frame_height:
            crop_y = frame_height - crop_h

        # dali normalizes crop params in a funny way
        crop_x_norm = crop_x / (frame_width - crop_w)
        crop_y_norm = crop_y / (frame_height - crop_h)
        crop_params = {
            'crop_h': crop_h,
            'crop_w': crop_w,
            'crop_pos_x': crop_x_norm,
            'crop_pos_y': crop_y_norm,
        }

        brightness = 4. if network == 'eye' else None

    else:
        raise ValueError(f'{network} is not a valid network label')

    # build pipeline
    pipe = video_pipe_crop_resize_flip(
        filenames=mp4_file,
        sequence_length=sequence_length,
        step=step,
        resize_dims=resize_dims,
        crop_params=crop_params,
        brightness=brightness,
        flip=flip,
        batch_size=1,  # do not change
        num_threads=1,  # do not change
        device_id=0,
    )

    # build torch iterator around pipeline
    predict_loader = LitDaliWrapper(pipe, **iter_args)

    return predict_loader


def analyze_video(
    network: str,
    mp4_file: str,
    model_path: str,
    flip: bool,
    original_dims: list,
    crop_window: Optional[list] = None,
    ensemble_number: int = 0,
    sequence_length: int = 32,
    save_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Analyze video with a single network.

    :param network: network name, key for `camera_params` features dict
    :param mp4_file: path to video file
    :param model_path: path to model directory that contains weights
    :param flip: True to flip horizontally
    :param original_dims: [height, width], original dims of video
    :param sequence_length: number of frames to load per sequence
    :param crop_window: list of floats [width, height, x, y] defining window used for cropping
    :param ensemble_number: unique integer to track predictions from different ensemble members
    :param save_dir: path to directory where results are saved in csv format
    :return: pandas DataFrame containing results
    """

    # NOTE: this call is critical; it forces a flusing of the GPU memory, which indirectly mitgates
    # memory fragmentation that can arise when processing a video with multiple networks.
    # If, for example, this line is commented out, the `iblvideo/tests/test_pose_lp.py` test will
    # fail because it runs many networks sequentially, each fragmenting the GPU memory (despite the
    # variable deletions and garbage collection performed at the end of this function).
    torch.cuda.synchronize()

    # load config file
    cfg_file = Path(model_path).joinpath('.hydra/config.yaml')
    if not cfg_file.exists():
        cfg_file = Path(model_path).joinpath('config.yaml')
        if not cfg_file.exists():
            raise IOError(f'Did not find {network} config.yaml file in model directory')
    cfg = DictConfig(yaml.safe_load(open(str(cfg_file), 'r')))

    # initialize data loader
    predict_loader = build_dataloader(
        network=network,
        mp4_file=mp4_file,
        model_type='context' if cfg.model.model_type == 'heatmap_mhcrnn' else 'baseline',
        sequence_length=sequence_length,
        flip=flip,
        crop_window=crop_window,
        resize_dims=[cfg.data.image_resize_dims.height, cfg.data.image_resize_dims.width],
        original_dims=original_dims,
    )

    # load model
    model = load_model_from_checkpoint(
        cfg=cfg,
        ckpt_file=str(next(Path(model_path).glob('*/*/*/*/*.ckpt'))),
        eval=True,
        data_module=None,
        skip_data_module=True,
    )

    # initialize trainer to run inference
    trainer = pl.Trainer(accelerator='gpu', logger=False)

    # run inference
    preds = trainer.predict(
        model=model,
        dataloaders=predict_loader,
        return_predictions=True,
    )

    # clean up predictions (resize, reformat, etc.)
    pred_handler = PredictionHandler(cfg=cfg, data_module=None, video_file=mp4_file)
    preds_df = pred_handler(preds=preds)
    csv_file = mp4_file.replace('.mp4', f'.{network}{ensemble_number}.csv')
    if save_dir:
        csv_file = os.path.join(save_dir, os.path.basename(csv_file))
    preds_df.to_csv(csv_file)

    # clear up GPU memory
    del preds
    del predict_loader
    del model
    del trainer
    gc.collect()  # GPU memory not cleared without this
    torch.cuda.empty_cache()

    return preds_df


def run_eks(
    network: str,
    eks_params: dict,
    mp4_file: str,
    csv_files: list,
    remove_files: bool = True,
) -> pd.DataFrame:
    """Run ensemble Kalman smoother using multiple network predictions.

    :param network: network name, key for `camera_params` features dict
    :param eks_params: parameters for eks, will be network-specific
    :param mp4_file: path to video file
    :param csv_files: paths to individual network outputs
    :param remove_files: True to remove prediction files from individual ensemble members
    :return: pandas DataFrame containing eks results
    """

    if len(csv_files) == 0:
        raise FileNotFoundError(f'Empty csv_files list provided to run_eks function')

    # get framerate of video in order to modify smoothing params
    clip = VideoFileClip(mp4_file)
    fps = clip.fps
    clip.close()

    # load files and put them in correct format
    markers_list = []
    for csv_file in csv_files:
        markers_curr = pd.read_csv(csv_file, header=[0, 1, 2], index_col=0)
        keypoint_names = [c[1] for c in markers_curr.columns[::3]]
        model_name = markers_curr.columns[0][0]
        markers_curr_fmt = convert_lp_dlc(markers_curr, keypoint_names, model_name=model_name)
        markers_list.append(markers_curr_fmt)

    if network == 'eye':

        df_smoothed, _, _ = ensemble_kalman_smoother_ibl_pupil(
            markers_list=markers_list,
            smooth_params=[eks_params['diameter'], eks_params['com']],
            avg_mode='median',
            var_mode='conf_weighted_var',
        )

    elif network == 'paws':

        if 25.0 <= fps <= 35.0:
            smooth_param = 100.0  # based on inspection of training videos
        elif 55.0 <= fps <= 65.0:
            smooth_param = eks_params['s']
        else:
            smooth_param = eks_params['s']

        df_smoothed, _ = ensemble_kalman_smoother_singlecam(
            markers_list=markers_list,
            keypoint_names=keypoint_names,
            smooth_param=smooth_param,
            avg_mode='median',
            var_mode='conf_weighted_var',
        )
        # apply global variance inflation factor
        # this value was computed from an independent labeled test set such that the posterior
        # variance output by EKS (after multiplication by the global variance inflation factor) is
        # roughly proportional to the squared pixel error
        # (and therefore the EKS posterior standard deviation is proportional to pixel error)
        mask = df_smoothed.columns.get_level_values('coords').str.endswith('_posterior_var')
        df_smoothed.loc[:, mask] *= 100.0

    else:
        raise NotImplementedError

    # save smoothed predictions
    csv_file_name = os.path.basename(mp4_file.replace('.mp4', f'.{network}.csv'))
    csv_file_dir = os.path.dirname(csv_files[0])
    csv_file_smooth = os.path.join(csv_file_dir, csv_file_name)
    df_smoothed.to_csv(csv_file_smooth)

    # delete individual predictions so that smoothed version is used downstream
    if remove_files:
        for csv_file in csv_files:
            os.remove(csv_file)

    return df_smoothed
