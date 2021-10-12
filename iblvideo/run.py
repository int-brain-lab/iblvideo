import logging
import shutil
import os
import traceback
import time
import cv2
import warnings
import shutil

from glob import glob
from datetime import datetime
from collections import OrderedDict

import numpy as np

from one.api import ONE
from one.alf.exceptions import ALFObjectNotFound
from ibllib.oneibl.patcher import FTPPatcher
from iblvideo.choiceworld import dlc
from iblvideo.motion_energy import motion_energy
from iblvideo.weights import download_weights
from iblvideo.utils import _check_nvidia_status, NvidiaDriverNotReady
from iblvideo import __version__
from ibllib.pipes import tasks
from ibllib.qc.dlc import DlcQC
from ibllib.io.video import assert_valid_label

_logger = logging.getLogger('ibllib')


# re-using the Task class allows to not re-write all the logging, error management
# and automatic settings of task statuses in the database
def _format_timer(timer):
    logstr = ''
    for item in timer.items():
        logstr += f'\nTiming {item[0]}Camera [sec]\n'
        for subitem in item[1].items():
            logstr += f'{subitem[0]}: {int(np.round(subitem[1]))}\n'
    return logstr


class TaskDLC(tasks.Task):
    gpu = 1
    cpu = 4
    io_charge = 90
    level = 0

    def _result_exists(self, session_id, fname):
        """ Checks if dlc result is available locally or in database. """
        result = None
        mode = None
        if os.path.exists(self.session_path.joinpath('alf', fname)):
            result = self.session_path.joinpath('alf', fname)
            _logger.info(f'Using local version of {fname}')
            mode = 'local'
        else:
            try:
                result = self.one.load_dataset(session_id, dataset=fname, download_only=True)
                _logger.info(f'Downloaded {fname} from database')
                mode = 'download'
            except ALFObjectNotFound:
                pass
        return result, mode

    @staticmethod
    def _video_intact(file_mp4):
        cap = cv2.VideoCapture(str(file_mp4))
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        intact = True if frame_count > 0 else False
        cap.release()
        return intact

    def _run(self, cams=('left', 'body', 'right'), version=__version__, frames=None,
             overwrite=False):
        session_id = self.one.path2eid(self.session_path)
        timer = OrderedDict()
        dlc_results, me_results, me_rois = [], [], []

        # Loop through cams
        for cam in cams:
            timer[f'{cam}'] = OrderedDict()
            # Check if dlc and me results are available locally or in database, if latter download
            if overwrite is True:
                dlc_result, me_result, me_roi = None, None, None
                dlc_mode, me_mode = None, None
            else:
                dlc_result, dlc_mode = self._result_exists(session_id, f'_ibl_{cam}Camera.dlc.pqt')
                # If dlc needs to be rerun, me should be rerun as well, regardless if it exists
                if dlc_result is None:
                    me_result, me_roi, me_mode = None, None, None
                else:
                    me_result, me_mode = self._result_exists(session_id,
                                                             f'{cam}Camera.ROIMotionEnergy.npy')
                    me_roi, _ = self._result_exists(session_id,
                                                    f'{cam}ROIMotionEnergy.position.npy')

            # If either dlc or me needs to be rerun, check if raw video exists, else download
            if dlc_result is None or me_result is None or me_roi is None:
                time_on = time.time()
                _logger.info(f'Downloading {cam}Camera.')
                video_intact, clobber_vid, attempt = False, False, 0
                while video_intact is False and attempt < 3:
                    dset = self.one.alyx.rest('datasets', 'list', session=session_id,
                                              name=f'_iblrig_{cam}Camera.raw.mp4', no_cache=True)
                    file_mp4 = self.one._download_dataset(dset[0], clobber=clobber_vid)
                    # Check if video is downloaded completely, otherwise retry twice
                    video_intact = self._video_intact(file_mp4)
                    if video_intact is False:
                        print('Timeout between download trials (2 min).')
                        time.sleep(120)  # wait two minutes e.g. to allow connection to come back
                    attempt += 1
                    clobber_vid = True
                if video_intact is False:
                    self.status = -1
                    _logger.error(f'{cam}Camera video failed to download.')
                    continue
                time_off = time.time()
                timer[f'{cam}'][f'Download video'] = time_off - time_on

            # If dlc_result doesn't exist or should be overwritten, run DLC
            dlc_computed = False
            if dlc_result is None:
                # Download weights if not exist locally
                path_dlc = download_weights(version=version)
                _logger.info(f'Running DLC on {cam}Camera.')
                try:
                    dlc_result, timer[f'{cam}'] = dlc(file_mp4, path_dlc=path_dlc, force=overwrite,
                                                      dlc_timer=timer[f'{cam}'])
                    _logger.info(dlc_result)
                    dlc_computed = True
                except BaseException:
                    _logger.error(f'DLC {cam}Camera failed.\n' + traceback.format_exc())
                    self.status = -1
                    # We dont' run motion energy, or add any files to be patched if dlc failed to run
                    continue

            # If me outputs don't exist or should be overwritten, run me
            me_computed = False
            if me_result is None or me_roi is None:
                _logger.info(f'Computing motion energy for {cam}Camera')
                try:
                    time_on = time.time()
                    me_result, me_roi = motion_energy(file_mp4, dlc_result, frames=frames)
                    time_off = time.time()
                    timer[f'{cam}']['Compute motion energy'] = time_off - time_on
                    _logger.info(me_result)
                    _logger.info(me_roi)
                    me_computed = True
                except BaseException:
                    _logger.error(f'Motion energy {cam}Camera failed.\n' + traceback.format_exc())
                    self.status = -1

            if dlc_computed is True or dlc_mode == 'local':
                dlc_results.append(dlc_result)
            if me_computed is True or me_mode == 'local':
                me_results.append(me_result)
                me_rois.append(me_roi)
        _logger.info(_format_timer(timer))
        return dlc_results, me_results, me_rois


def run_session(session_id, machine=None, cams=('left', 'body', 'right'), one=None,
                version=__version__, remove_videos=True, frames=10000, overwrite=True, **kwargs):
    """
    Run DLC on a single session in the database.

    :param session_id: Alyx eID of session to run
    :param machine: Tag for the machine this job ran on (string)
    :param cams: Tuple of labels of videos to run dlc and motion energy on.
                 Valid labels are 'left', 'body' and 'right'.
    :param one: ONE instance to use for query (optional)
    :param version: Version of iblvideo and DLC weights to use (default is current version)
    :param remove_videos: Whether to remove the local raw_video_data after DLC (default is True)
    :param frames: Number of video frames loaded into memory at once while computing ME. If None,
                   all frames of a video are loaded at once. (default is 50000, see below)
    :param overwrite: whether to overwrite existing outputs of previous runs (default is True)
    :param kwargs: Additional keyword arguments to be passed to TaskDLC

    :return status: final status of the task

    The frames parameter determines how many cropped frames per camera are loaded into memory at
    once and should be set depending on availble RAM. Some approximate numbers for orientation,
    assuming 90 min video and frames set to:
    1       : 152 KB (body),   54 KB (left),   15 KB (right)
    50000   : 7.6 GB (body),  2.7 GB (left), 0.75 GB (right)
    None    :  25 GB (body), 17.5 GB (left), 12.5 GB (right)
    """
    # Check if the GPU is addressable
    _check_nvidia_status()
    # Catch all errors that are not caught inside run function and put them in the log
    try:
        # Create ONE instance if none is given
        one = one or ONE()
        session_path = one.eid2path(session_id)
        tdict = one.alyx.rest('tasks', 'list',
                              django=f"name__icontains,DLC,session__id,{session_id}",
                              no_cache=True)[0]
    except IndexError:
        print(f"No DLC task found for session {session_id}")
        return -1

    try:
        # Check if labels are valid
        cams = tuple(assert_valid_label(cam) for cam in cams)  # raises ValueError if label invalid
        # Check if all requested videos exist
        vids = [dset['name'] for dset in one.alyx.rest('datasets', 'list',
                                                       django=(f'session__id,{session_id},'
                                                               'data_format__name,mp4,'
                                                               f'name__icontains,camera'),
                                                       no_cache=True
                                                       )]
        no_vid = [cam for cam in cams if f'_iblrig_{cam}Camera.raw.mp4' not in vids]
        if len(no_vid) > 0:
            # If less datasets, update task and raise error
            log_str = '\n'.join([f"No raw video file found for {no_cam}Camera." for
                                 no_cam in no_vid])
            patch_data = {'log': log_str, 'status': 'Errored'}
            one.alyx.rest('tasks', 'partial_update', id=tdict['id'], data=patch_data,
                          no_cache=True)
            return -1
        else:
            # set a flag in local session folder to later resume if interrupted
            session_path.mkdir(parents=True, exist_ok=True)
            session_path.joinpath('dlc_started').touch(exist_ok=True)
            # create the task instance and run it, update task
            task = TaskDLC(session_path, one=one, taskid=tdict['id'], machine=machine, **kwargs)
            status = task.run(cams=cams, version=version, frames=frames, overwrite=overwrite)
            patch_data = {'time_elapsed_secs': task.time_elapsed_secs, 'log': task.log,
                          'status': 'Errored' if status == -1 else 'Complete'}
            one.alyx.rest('tasks', 'partial_update', id=tdict['id'], data=patch_data,
                          no_cache=True)
            # register the data using the FTP patcher
            if task.outputs:
                # it is safer to instantiate the FTP right before transfer to prevent time-out
                ftp_patcher = FTPPatcher(one=one)
                # Make a flat list of outputs
                outputs = [element for output_list in task.outputs for element in output_list]
                if len(outputs) > 0:
                    for output in outputs:
                        ftp_patcher.create_dataset(path=output)
                    # Update the version only now and only if new outputs are uploaded
                    one.alyx.rest('tasks', 'partial_update', id=tdict['id'], data={'version': version})
                else:
                    _logger.warning("No new outputs computed.")
            if remove_videos is True:
                shutil.rmtree(session_path.joinpath('raw_video_data'), ignore_errors=True)

            # Run DLC QC
            # Download camera times and then force qc to use local data as dlc might not have
            # been updated on FlatIron at this stage
            try:
                datasets = one.type2datasets(session_id, 'camera.times')
                one.load_datasets(session_id, datasets=datasets, download_only=True)
                alf_path = one.eid2path(session_id).joinpath('alf')
                for cam in cams:
                    # Only run if dlc actually exists
                    if alf_path.joinpath(f'_ibl_{cam}Camera.dlc.pqt').exists():
                        qc = DlcQC(session_id, cam, one=one, download_data=False)
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=RuntimeWarning)
                            qc.run(update=True)
            except AssertionError:
                # If the camera.times don't exist we cannot run QC, but the DLC task shouldn't fail
                # Make sure to not overwrite the task log if that has already been updated
                tdict = one.alyx.rest('tasks', 'list',
                                      django=f"name__icontains,DLC,session__id,{session_id}",
                                      no_cache=True)[0]
                patch_data = {'log': tdict['log'] + '\n\n' + traceback.format_exc()}
                one.alyx.rest('tasks', 'partial_update', id=tdict['id'], data=patch_data,
                              no_cache=True)

    except BaseException:
        # Make sure to not overwrite the task log if that has already been updated
        tdict = one.alyx.rest('tasks', 'list',
                              django=f"name__icontains,DLC,session__id,{session_id}",
                              no_cache=True)[0]
        patch_data = {'log': tdict['log'] + '\n\n' + traceback.format_exc(), 'status': 'Errored'}
        one.alyx.rest('tasks', 'partial_update', id=tdict['id'], data=patch_data, no_cache=True)
        return -1
    # Remove in progress flag
    session_path.joinpath('dlc_started').unlink()
    # Set status to Incomplete if the length of cameras was < 3 but everything passed
    if status == 0 and len(cams) < 3:
        one.alyx.rest('tasks', 'partial_update', id=tdict['id'], data={'status': 'Incomplete'}, no_cache=True)
    return status


def run_queue(machine=None, target_versions=(__version__), statuses=('Empty', 'Waiting', 'Complete'),
              restart_local=True, overwrite=True, n_sessions=1000, delta_query=600,
              **kwargs):
    """
    Run all tasks that have a version below min_version and a status in statuses. By default
    overwrites pre-exisitng results.

    :param machine: str, tag for the machine this job ran on
    :param target_versions: str, if the task is at this version it should not be rerun
    :param statuses: tuple, task statuses which should be (re)run
    :param restart_local: bool, whether to restart interrupted local jobs (default is True)
    :param overwrite: bool, whether overwrite existing outputs of previous runs (default is True)
    :param n_sessions: int, number of sessions to run from queue
    :param delta_query: int, time between querying the database in sec
    :param kwargs: Keyword arguments to be passed to run_session
    """

    one = ONE()
    # Loop until n_sessions is reached or something breaks
    machine = machine or one.alyx.user
    status_dict = {}

    # Check if any interrupted local sessions are present
    if restart_local is True:
        local_tmp = glob(one.alyx._par.CACHE_DIR + '/*lab*/Subjects/*/*/*/dlc_started')
        if len(local_tmp) > 0:
            local_sessions = list(set([one.path2eid(local) for local in local_tmp]))
            n_sessions -= len(local_sessions)
            for eid in local_sessions:
                print(f'Restarting local session {eid}')
                status_dict[eid] = run_session(eid, machine=machine, one=one, **kwargs)

    count = 0
    last_query = datetime.now()
    while count < n_sessions:
        # Query EphysDLC tasks, find those with version lower than min_version
        # redo this only every delta_query seconds
        delta = (datetime.now() - last_query).total_seconds()
        if (delta > delta_query) or (count == 0):
            last_query = datetime.now()
            all_tasks = one.alyx.rest('tasks', 'list', name='EphysDLC', no_cache=True)
            task_queue = [t for t in all_tasks if t['status'] in statuses and
                          (t['version'] is None or t['version'] not in target_versions)]
            task_queue = sorted(task_queue, key=lambda k: k['priority'], reverse=True)
            if len(task_queue) == 0:
                print("No sessions to run")
                return
            else:
                sessions = [t['session'] for t in task_queue]
        # Return if no more sessions to run
        if len(sessions) == 0:
            print("No sessions to run")
            return
        # Run next session in the list, capture status in dict and move on to next session
        eid = sessions[0]
        status_dict[eid] = run_session(sessions.pop(0), machine=machine, overwrite=overwrite,
                                       one=one, **kwargs)
        count += 1

    return status_dict
