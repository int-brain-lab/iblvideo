"""Helper functions to run Lightning Pose on a single IBL video with trained networks."""

from eks.utils import convert_lp_dlc
from eks.pupil_smoother import ensemble_kalman_smoother_pupil
from eks.singleview_smoother import ensemble_kalman_smoother_single_view
import gc
import lightning.pytorch as pl
from lightning_pose.data import _IMAGENET_MEAN, _IMAGENET_STD
from lightning_pose.data.dali import LitDaliWrapper
from lightning_pose.data.utils import count_frames
from lightning_pose.utils.predictions import load_model_from_checkpoint, PredictionHandler
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


def get_crop_window(roi_df_file: Path, network_params: dict) -> list:
    """Get average position of a anchor point for autocropping.

    :param roi_df_file: path to dataframe output by ROI network
    :param network_params: parameters for network, see SIDE_FEATURES and BODY_FEATURES in params.py
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
    return network_params['crop'](*xy)


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
    orig_size = fn.shapes(video)

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
        orig_size = fn.shapes(video)
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
    network_label: str,
    mp4_file: str,
    camera_params: dict,
    model_type: str,
    sequence_length: int,
    crop_window: Optional[list] = None,
) -> LitDaliWrapper:
    """Build pytorch data loader that wraps DALI pipeline.

    :param network_label: network name, key for `camera_params` features dict
    :param mp4_file: path to video file
    :param camera_params: parameters for camera, see LEFT_VIDEO etc in params.py
    :param model_type: 'baseline' | 'context'
    :param sequence_length: number of frames to load per sequence
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

    features = camera_params['features'][network_label]
    flip = camera_params['flip']

    # set network-specific args
    if network_label in ['roi_detect', 'paws', 'tail_start']:

        crop_params = None
        brightness = None
        resize_dims = features['resize_dims']

    elif network_label in ['eye', 'tongue', 'nose_tip']:

        # resize crop window depending on view
        crop_w, crop_h, crop_x, crop_y = crop_window
        crop_w /= camera_params['sampling']
        crop_h /= camera_params['sampling']
        crop_x /= camera_params['sampling']
        crop_y /= camera_params['sampling']
        frame_width = camera_params['original_size'][0]
        frame_height = camera_params['original_size'][1]

        # dali normalizes crop params in a funny way
        crop_x_norm = crop_x / (frame_width - crop_w)
        crop_y_norm = crop_y / (frame_height - crop_h)
        crop_params = {
            'crop_h': crop_h,
            'crop_w': crop_w,
            'crop_pos_x': crop_x_norm,
            'crop_pos_y': crop_y_norm,
        }

        brightness = 4. if network_label == 'eye' else None
        resize_dims = features['resize_dims']

    else:
        raise ValueError(f'{network_label} is not a valid network label')

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
    camera_params: dict,
    crop_window: Optional[list] = None,
    ensemble_number: int = 0,
    sequence_length: int = 32,
    save_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Analyze video with a single network.

    :param network: network name, key for `camera_params` features dict
    :param mp4_file: path to video file
    :param model_path: path to model directory that contains weights
    :param camera_params: parameters for camera, see LEFT_VIDEO etc in params.py
    :param sequence_length: number of frames to load per sequence
    :param crop_window: list of floats [width, height, x, y] defining window used for cropping
    :param ensemble_number: unique integer to track predictions from different ensemble members
    :param save_dir: path to directory where results are saved in csv format
    :return: pandas DataFrame containing results
    """

    # load config file
    cfg_file = Path(model_path).joinpath('.hydra/config.yaml')
    cfg = DictConfig(yaml.safe_load(open(str(cfg_file), 'r')))

    # initialize data loader
    predict_loader = build_dataloader(
        network_label=network,
        mp4_file=mp4_file,
        camera_params=camera_params,
        model_type='context' if cfg.model.model_type == 'heatmap_mhcrnn' else 'baseline',
        sequence_length=sequence_length,
        crop_window=crop_window,
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
    trainer = pl.Trainer(accelerator='gpu')

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

    # clear up memory
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

    # load files and put them in correct format
    markers_list = []
    for csv_file in csv_files:
        markers_curr = pd.read_csv(csv_file, header=[0, 1, 2], index_col=0)
        keypoint_names = [c[1] for c in markers_curr.columns[::3]]
        model_name = markers_curr.columns[0][0]
        markers_curr_fmt = convert_lp_dlc(markers_curr, keypoint_names, model_name=model_name)
        markers_list.append(markers_curr_fmt)

    if network == 'eye':

        # parameters hand-picked for smoothing purposes (diameter_s, com_s, com_s)
        state_transition_matrix = np.asarray([
            [eks_params['diameter'], 0, 0],
            [0, eks_params['com'], 0],
            [0, 0, eks_params['com']]
        ])

        # run eks
        df_dicts = ensemble_kalman_smoother_pupil(
            markers_list=markers_list,
            keypoint_names=keypoint_names,
            tracker_name='ensemble-kalman_tracker',
            state_transition_matrix=state_transition_matrix,
            likelihood_default=1.0,
        )
        df_tmp = df_dicts['markers_df']
        good_cols = [c[2].find('var') == -1 for c in df_tmp.columns.to_flat_index()]
        df_smoothed = df_tmp.loc[:, good_cols]

    elif network == 'paws':

        # make empty dataframe to write eks results into
        df_smoothed = markers_curr.copy()
        df_smoothed.columns = df_smoothed.columns.set_levels(['ensemble-kalman_tracker'], level=0)
        for col in df_smoothed.columns:
            if col[-1] == 'likelihood':
                # set this to 1.0 so downstream filtering functions don't get tripped up
                df_smoothed[col].values[:] = 1.0
            else:
                df_smoothed[col].values[:] = np.nan

        # loop over keypoints; apply eks to each individually
        for kp in keypoint_names:
            # run eks
            keypoint_df_dict, s_final, nll_values = ensemble_kalman_smoother_single_view(
                markers_list=markers_list,
                keypoint_ensemble=kp,
                smooth_param=eks_params['s'],
            )
            keypoint_df = keypoint_df_dict[kp + '_df']
            # put results into new dataframe
            for coord in ['x', 'y', 'zscore']:
                src_cols = ('ensemble-kalman_tracker', f'{kp}', coord)
                dst_cols = ('ensemble-kalman_tracker', f'{kp}', coord)
                df_smoothed.loc[:, dst_cols] = keypoint_df.loc[:, src_cols]

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
