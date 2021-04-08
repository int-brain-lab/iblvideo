import logging
import shutil
import traceback
from datetime import datetime
import numpy as np

from oneibl.one import ONE
from oneibl.patcher import FTPPatcher
from iblvideo.choiceworld import dlc
from iblvideo.motion_energy import motion_energy
from iblvideo.weights import download_weights
from iblvideo import __version__
from ibllib.pipes import tasks
from ibllib.qc.dlc import DlcQC
from ibllib.io.video import label_from_path, assert_valid_label

_logger = logging.getLogger('ibllib')


# re-using the Task class allows to not re-write all the logging, error management
# and automatic settings of task statuses in the database
class TaskDLC(tasks.Task):
    gpu = 1
    cpu = 4
    io_charge = 90
    level = 0

    def _run(self, cams=('left', 'body', 'right'), version=__version__, frames=None):

        session_id = self.one.eid_from_path(self.session_path)
        # Loop through cams on this level
        dlc_results, me_results = [], []
        for cam in cams:
            # Check for existing dlc data
            dlc_data = self.one.alyx.rest('datasets', 'list', session=session_id,
                                          django=f'name__icontains,{cam}Camera.dlc')
            if len(dlc_data) == 0:
                # Download the camera data if not available locally
                file_mp4 = self.one.load(session_id, dataset_types=['_iblrig_Camera.raw'],
                                         download_only=True)
                # Download weights into ONE Cache directory under 'resources/DLC' if not exist
                path_dlc = download_weights(version=version)
                # Run DLC and log
                _logger.info(f'Running DLC on {cam}Camera.')
                try:
                    dlc_result = dlc(file_mp4, path_dlc)  # TODO: Possibly pass logger?
                    _logger.info(dlc_result)
                    dlc_results.append(dlc_result)
                except BaseException:
                    _logger.error(f'DLC {cam}Camera failed.\n' + traceback.format_exc())
            else:
                # Download dlc data if not locally available
                dlc_result = self.one.load(session_id, dataset_types='camera.dlc',
                                           download_only=True)
            # Run DLC QC
            qc = DlcQC(session_id, cam, one=self.one, log=_logger)
            qc.run(update=True)
            _logger.info(f'Computing motion energy for {cam}Camera')
            try:
                me_result, _ = motion_energy(self.session_path, dlc_result, frames=frames,
                                             one=self.one)
                _logger.info(me_result)
                me_results.append(me_result)
            except BaseException:
                _logger.error(f'Motion energy {cam}Camera failed.\n' + traceback.format_exc())

        return dlc_results, me_results


def run_session(session_id, machine=None, cams=('left', 'body', 'right'), one=None,
                version=__version__, remove_videos=True, frames=50000, **kwargs):
    """
    Run DLC on a single session in the database.

    :param session_id: Alyx eID of session to run
    :param machine: Tag for the machine this job ran on (string)
    :param cams: Tuple of labels of videos to run dlc and motion energy on.
                 Valid labels are 'left', 'body' and 'right'.
    :param one: ONE instance to use for query (optional)
    :param version: Version of iblvideo / DLC weights to use (default is current version)
    :param remove_videos: Whether to remove the local raw_video_data after DLC (default is True)
    :param frames: Number of video frames loaded into memory at once while computing ME. If None,
                   all frames of a video are loaded at once. (default is 50000, see below)
    :param kwargs: Additional keyword arguments to be passed to TaskDLC
    :return status: final status of the task

    The frames parameter determines how many cropped frames per camera are loaded into memory at
    once and should be set depending on availble RAM. Some approximate numbers for orientation,
    assuming 90 min video and frames set to:
    1       : 152 KB (body),   54 KB (left),   15 KB (right)
    50000   : 7.6 GB (body),  2.7 GB (left), 0.75 GB (right)
    None    :  25 GB (body), 17.5 GB (left), 12.5 GB (right)
    """
    # Catch all errors that are not caught inside run function and put them in the log
    try:
        # Create ONE instance if none are given
        one = one or ONE()
        session_path = one.path_from_eid(session_id)
        tdict = one.alyx.rest('tasks', 'list',
                              django=f"name__icontains,DLC,session__id,{session_id}")[0]
    except IndexError:
        print(f"No DLC task found for session {session_id}")

    try:
        # Check if labels are valid
        cams = tuple(assert_valid_label(cam) for cam in cams)  # raises ValueError if label invalid
        # Check if all requested videos exist
        vids = [dset['name'] for dset in one.alyx.rest('datasets', 'list',
                                                       django=(f'session__id,{session_id},'
                                                               'data_format__name,mp4,'
                                                               f'name__icontains,camera')
                                                       )]
        no_vid = [cam for cam in cams if f'_iblrig_{cam}Camera.raw.mp4' not in vids]
        if len(no_vid) > 0:
            # If less datasets, update task and raise error
            log_str = '\n'.join([f"No raw video file found for {no_cam}Camera." for
                                 no_cam in no_vid])
            patch_data = {'log': log_str, 'version': version, 'status': 'Errored'}
            one.alyx.rest('tasks', 'partial_update', id=tdict['id'], data=patch_data)
            status = -1
        else:
            # create the task instance and run it, update task
            task = TaskDLC(session_path, one=one, taskid=tdict['id'], machine=machine, **kwargs)
            status = task.run(cams=cams, version=version, frames=frames)
            patch_data = {'time_elapsed_secs': task.time_elapsed_secs, 'log': task.log,
                          'version': version, 'status': 'Errored' if status == -1 else 'Complete'}
            one.alyx.rest('tasks', 'partial_update', id=tdict['id'], data=patch_data)
            # register the data using the FTP patcher
            if task.outputs:
                # it is safer to instantiate the FTP right before transfer to prevent time-out
                ftp_patcher = FTPPatcher(one=one)
                ftp_patcher.create_dataset(path=task.outputs[0] + task.outputs[1])
            if remove_videos is True:
                shutil.rmtree(session_path.joinpath('raw_video_data'), ignore_errors=True)

    except BaseException:
        patch_data = {'log': tdict['log'] + '\n\n' + traceback.format_exc(),
                      'version': version, 'status': 'Errored'}
        one.alyx.rest('tasks', 'partial_update', id=tdict['id'], data=patch_data)
        status = -1

    return status


def run_queue(machine=None, n_sessions=np.inf, delta_query=600, **kwargs):
    """
    Run the entire queue, or n_sessions, of DLC tasks on Alyx.

    :param machine: Tag for the machine this job ran on (string)
    :param n_sessions: Number of sessions to run from queue (default is run whole queue)
    :param version: Version of iblvideo / DLC weights to use (default is current version)
    :param delta_query: Time between querying the database for Empty tasks, in sec
    :param kwargs: Keyword arguments to be passed to run_session.
    """

    one = ONE()
    # Loop until n_sessions is reached or something breaks
    status_dict = {}
    count = 0
    last_query = datetime.now()
    while count < n_sessions:
        # Query EphysDLC tasks that have not been run, redo this only every delta_query seconds
        delta = (datetime.now() - last_query).total_seconds()
        if (delta > delta_query) or (count == 0):
            last_query = datetime.now()
            tasks = one.alyx.rest('tasks', 'list', status='Empty', name='EphysDLC')
            sessions = [t['session'] for t in tasks]
        # Return if no more sessions to run
        if len(sessions) == 0:
            print("No sessions to run")
            return
        # Run next session in the list, capture status in dict and move on to next session
        eid = sessions[0]
        status_dict[eid] = run_session(sessions.pop(0), machine=machine, one=one, **kwargs)
        count += 1

    return status_dict
