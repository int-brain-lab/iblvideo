import lightning.pytorch as pl
from lightning_pose.data import _IMAGENET_MEAN, _IMAGENET_STD
from lightning_pose.data.dali import LitDaliWrapper
from lightning_pose.data.utils import count_frames
from lightning_pose.utils.predictions import (
    create_labeled_video,
    load_model_from_checkpoint,
    PredictionHandler,
)
from lightning_pose.utils.scripts import get_imgaug_transform, get_dataset, get_data_module
import numpy as np
from omegaconf import DictConfig
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import LastBatchPolicy
import nvidia.dali.types as types
import os
import pandas as pd
from pathlib import Path
import torch
from typing import List, Optional, Dict, Union
import yaml

from iblvideo.params import BODY_FEATURES, SIDE_FEATURES, LEFT_VIDEO, RIGHT_VIDEO, BODY_VIDEO


def collect_model_paths(view: str):
    if view == 'body':
        raise NotImplementedError
    else:
        return {
            'roi_detect': '/media/mattw/behavior/results/pose-estimation/ibl-roi-detect/2023-06-21/19-47-17',
            'nose_tip': '/media/mattw/behavior/results/pose-estimation/ibl-nose-tip/2023-06-21/11-24-04',
            'eye': '/media/mattw/behavior/results/pose-estimation/ibl-pupil/ensembling-expts/models/functional-nightingale-5302-exp0/outputs/2023-01-26/18-47-00',
            'paws': '/media/mattw/behavior/results/pose-estimation/ibl-paw/ensembling-expts/models/discerning-hertz-4605-exp0/outputs/2023-02-06/19-40-18',
            'tongue': '/media/mattw/behavior/results/pose-estimation/ibl-tongue/2023-06-21/12-38-42',
        }


def get_crop_window(file_df_crop: Path, network: dict):
    """Get average position of a pivot point for autocropping.

    :param file_df_crop: Path to data frame from csv file from video data
    :param network: dictionary describing the networks. See constants SIDE and BODY
    :return: list of floats [width, height, x, y] defining window used for ffmpeg crop command
    """
    df_crop = pd.read_csv(file_df_crop, header=[0, 1, 2], index_col=0)
    XYs = []
    for part in network['features']:
        x_values = df_crop[(df_crop.keys()[0][0], part, 'x')].values
        y_values = df_crop[(df_crop.keys()[0][0], part, 'y')].values
        likelihoods = df_crop[(df_crop.keys()[0][0], part, 'likelihood')].values

        mx = np.ma.masked_where(likelihoods < 0.9, x_values)
        x = np.ma.compressed(mx)
        my = np.ma.masked_where(likelihoods < 0.9, y_values)
        y = np.ma.compressed(my)

        XYs.append([int(np.nanmean(x)), int(np.nanmean(y))])

    xy = np.mean(XYs, axis=0)
    return network['crop'](*xy)


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
):
    """Video reader pipeline that loads videos, normalizes, crops, and optionally flips.

    Args:
        filenames: list of absolute paths of video files to feed through pipeline
        sequence_length: number of frames to load per sequence
        pad_sequences: allows creation of incomplete sequences if there is an
            insufficient number of frames at the very end of the video
        pad_last_batch: pad final batch with empty sequences
        step: number of frames to advance on each read; will be different for context
            vs non-context models
        name: pipeline name, used to string together DataNode elements
        crop_params: keys are
            - 'crop_h': height in pixels
            - 'crop_w': width in pixels
            - 'crop_pos_x': x position of top left corner; normalized in (0, 1)
            - 'crop_pos_y': y position of top left corner; normalized in (0, 1)
        normalization_mean: mean values in (0, 1) to subtract from each channel
        normalization_std: standard deviation values to subtract from each
        resize_dims: [height, width] to resize raw frames
        brightness: increase brightness of frames
        flip: True to flip frame around vertical axis
        batch_size: number of sequences per batch
        num_threads: number of cpu threads used by the pipeline
        device_id: id of the gpu used by the pipeline

    Returns:
        pipeline object to be fed to DALIGenericIterator
        placeholder int to represent unused "transforms" field in dataloader

    """

    # read batches of video from file
    video = fn.readers.video(
        filenames=filenames,
        sequence_length=sequence_length,
        pad_sequences=pad_sequences,
        pad_last_batch=pad_last_batch,
        step=step,
        name=name,
        device='gpu',
        random_shuffle=False,
        initial_fill=sequence_length,
        normalized=False,
        dtype=types.DALIDataType.FLOAT,
        file_list_include_preceding_frame=True,  # to get rid of dali warnings
    )
    # original videos range [0, 255]; transform it to [0, 1]
    # torchvision automatically performs this transformation upon tensor creation
    video = (video / 255.0)
    # adjust color levels
    # brightness=4 is equivalent to ffmpeg's
    #     "colorlevels=rimax=0.25:gimax=0.25:bimax=0.25"
    if brightness:
        video = fn.coord_transform(
            video,
            M=np.array([[brightness, 0, 0], [0, brightness, 0], [0, 0, brightness]])
        )
    # change channel layout, crop, and normalize according to imagenet statistics
    if crop_params:
        video = fn.crop_mirror_normalize(
            video,
            crop_h=crop_params['crop_h'],  # pixels
            crop_w=crop_params['crop_w'],  # pixels
            crop_pos_x=crop_params['crop_pos_x'],  # normalized in (0, 1)
            crop_pos_y=crop_params['crop_pos_y'],  # normalized in (0, 1)
            output_layout='FCHW',
            mean=normalization_mean,
            std=normalization_std,
        )
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
    # flip video in horizontal direction to match left/right views
    if flip:
        video = fn.flip(video, horizontal=1)

    return video, -1


def compute_num_iters(video_path: str, sequence_length: int, step: int, model_type: str):
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
        raise ValueError(f'model_type must be "baseline" or "context", not {moel_type}')
    return num_iters


def build_dataloader(
    network: str,
    mp4_file: str,
    view: str,
    model_type: str,
    sequence_length: int,
    crop_window: Optional[list] = None,
) -> LitDaliWrapper:

    flip = True if view == 'right' else False

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
        'output_map': ['frames', 'transforms'],
        'last_batch_policy': LastBatchPolicy.FILL,
        'last_batch_padded': False,
        'auto_reset': False,
        'reader_name': 'reader',
        'do_context': do_context,
    }

    # set network-specific args
    if network in ['roi_detect', 'paws']:

        resize_dims = LEFT_VIDEO['features'][network]['resize_dims']
        crop_params = None
        brightness = None

    elif network in ['eye', 'tongue', 'nose_tip']:

        if view == 'left':
            frame_width = LEFT_VIDEO['original_size'][0]
            frame_height = LEFT_VIDEO['original_size'][1]
        else:
            frame_width = RIGHT_VIDEO['original_size'][0]
            frame_height = RIGHT_VIDEO['original_size'][1]

        resize_dims = SIDE_FEATURES[network]['resize_dims']
        brightness = 4. if network == 'eye' else None

        # dali normalizes crop params in a funny way
        crop_w, crop_h, crop_x, crop_y = crop_window
        crop_x_norm = crop_x / (frame_width - crop_w)
        crop_y_norm = crop_y / (frame_height - crop_h)
        crop_params = {
            'crop_h': crop_h,
            'crop_w': crop_w,
            'crop_pos_x': crop_x_norm,
            'crop_pos_y': crop_y_norm,
        }

    else:
        raise ValueError(f'{network} is not a valid network')

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
    view: str,
    create_labels: bool = False,
    crop_window: Optional[list] = None,
    sequence_length: int = 32,
    save_dir: Optional[str] = None,
) -> pd.DataFrame:

    def get_data_dir(n):
        if n == 'nose_tip':
            return '/media/mattw/behavior/pose-estimation-data-final/ibl-nose-tip'
        elif n == 'eye':
            return '/media/mattw/behavior/pose-estimation-data-final/ibl-pupil'
        elif n == 'paws':
            return '/media/mattw/behavior/pose-estimation-data-final/ibl-paw'
        elif n == 'tongue':
            return '/media/mattw/behavior/pose-estimation-data-final/ibl-tongue'
        elif n == 'roi_detect':
            return '/media/mattw/behavior/pose-estimation-data-final/ibl-roi-detect'
        else:
            raise NotImplementedError(f'add path for {n} network')

    # load config file
    cfg_file = Path(model_path) / '.hydra' / 'config.yaml'
    cfg = DictConfig(yaml.safe_load(open(str(cfg_file), 'r')))
    cfg.training.imgaug = 'default'  # IMPORTANT! don't augment frames
    cfg.data.data_dir = get_data_dir(network)
    cfg.data.video_dir = os.path.join(cfg.data.data_dir, 'videos')

    # build data module  # TODO: can we get rid of this?
    imgaug_transform = get_imgaug_transform(cfg=cfg)
    dataset = get_dataset(cfg=cfg, data_dir=cfg.data.data_dir, imgaug_transform=imgaug_transform)
    data_module = get_data_module(cfg=cfg, dataset=dataset, video_dir=cfg.data.video_dir)

    # initialize data loader
    predict_loader = build_dataloader(
        network=network,
        mp4_file=mp4_file,
        view=view,
        model_type='context' if cfg.model.do_context else 'baseline',
        sequence_length=sequence_length,
        crop_window=crop_window,
    )

    # load model
    model = load_model_from_checkpoint(
        cfg=cfg,
        ckpt_file=next(Path(model_path).glob('*/*/*/*/*.ckpt')),
        eval=True,
        data_module=data_module,
    )

    # initialize trainer to run inference
    trainer = pl.Trainer(accelerator='gpu')

    # run inference
    preds = trainer.predict(
        model=model,
        dataloaders=predict_loader,
        return_predictions=True,
    )

    # call this instance on a single vid's preds
    pred_handler = PredictionHandler(cfg=cfg, data_module=data_module, video_file=mp4_file)
    preds_df = pred_handler(preds=preds)
    csv_file = mp4_file.replace('.mp4', f'.{network}.csv')
    if save_dir:
        csv_file = os.path.join(save_dir, os.path.basename(csv_file))
    preds_df.to_csv(csv_file)

    # create labeled video
    if create_labels:
        from moviepy.editor import VideoFileClip
        mp4_file_labeled = Path(str(mp4_file).replace('.mp4', f'.{network}.labeled.mp4'))
        if save_dir:
            mp4_file_labeled = os.path.join(save_dir, os.path.basename(mp4_file_labeled))
        video_clip = VideoFileClip(mp4_file)
        # transform df to numpy array
        keypoints_arr = np.reshape(preds_df.to_numpy(), [preds_df.shape[0], -1, 3])
        xs_arr = keypoints_arr[:, :, 0]
        ys_arr = keypoints_arr[:, :, 1]
        mask_array = keypoints_arr[:, :, 2] > cfg.eval.get('confidence_thresh_for_vid', 0.9)
        # video generation
        create_labeled_video(
            clip=video_clip,
            xs_arr=xs_arr,
            ys_arr=ys_arr,
            mask_array=mask_array,
            filename=str(mp4_file_labeled),
        )

    # clear up memory
    del dataset
    del data_module
    del predict_loader
    del model
    del trainer
    torch.cuda.empty_cache()

    return preds_df
