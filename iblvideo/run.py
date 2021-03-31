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
from ibllib.io.video import label_from_path

_logger = logging.getLogger('ibllib')


# re-using the Task class allows to not re-write all the logging, error management
# and automatic settings of task statuses in the database
class TaskDLC(tasks.Task):
    gpu = 1
    cpu = 4
    io_charge = 90
    level = 0

    def _run(self, n_cams=3, version=__version__, **kwargs):

        session_id = self.one.eid_from_path(self.session_path)
        # Check for existing dlc data
        dlc_data = self.one.alyx.rest('datasets', 'list', django=(f'session__id,{session_id},'
                                                                  'name__icontains,dlc'))
        # TODO: rerun only for the cams that don't yet exist
        run_dlc = None
        if len(dlc_data) < n_cams:
            run_dlc = True
            # Download the camera data
            self.one.load(session_id, dataset_types=['_iblrig_Camera.raw'], download_only=True)
            # Download weights into ONE Cache directory under 'resources/DLC' if not exist
            path_dlc = download_weights(version=version)
            # Run dlc on all videos in raw_video_data
            files_mp4 = list(self.session_path.joinpath('raw_video_data').glob('*.mp4'))
            _logger.info(f'Running DLC on {len(files_mp4)} video files.')
            for cam in range(len(files_mp4)):
                dlc_result = dlc(files_mp4[cam], path_dlc)
                _logger.info(dlc_result)
        else:
            # Download dlc data if not locally available
            self.one.load(session_id, dataset_types='camera.dlc')

        # Run DLC QC, if it passes run motion energy
        # sort results so that left is first in case one only wants to run left
        dlc_results = sorted(self.session_path.joinpath('alf').glob('*dlc.pqt'),
                             key=lambda f: 'left' in str(f), reverse=True)
        for i in range(n_cams):
            dlc_pqt = dlc_results[i]
            label = label_from_path(dlc_pqt)
            qc = DlcQC(session_id, label, one=self.one, log=_logger)
            outcome, metrics = qc.run(update=True)
            if all(x == 'PASS' for x in metrics.values()):
                _logger.info(f'Computing motion energy for {label}Camera')
                frames = kwargs.pop('frames', None)
                me_result, _ = motion_energy(self.session_path, dlc_pqt,
                                             frames=frames, one=self.one)
                _logger.info(me_result)
            else:
                _logger.info(f'{label}Camera did not pass DLC QC, skipping motion energy')

        me_results = list(self.session_path.joinpath('alf').glob('*ROIMotionEnergy*.npy'))

        if run_dlc:
            return dlc_results, me_results
        else:
            return [], me_results


def run_session(session_id, machine=None, n_cams=3, one=None, version=__version__,
                remove_videos=True, frames=10000):
    """
    Run DLC on a single session in the database.

    :param session_id: Alyx eID of session to run
    :param machine: Tag for the machine this job ran on (string)
    :param n_cams: Minimum number of camera datasets required
    :param one: ONE instance to use for query (optional)
    :param version: Version of iblvideo / DLC weights to use (default is current version)
    :param remove_data: Whether to remove the local raw_video_data after DLC (default is True)
    :return status: final status of the task
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
        # Check if required number of cameras is available
        dsets = one.alyx.rest('datasets', 'list', django=(f'session__id,{session_id},'
                                                          'data_format__name,mp4,'
                                                          'name__icontains,camera'))
        if len(dsets) < n_cams:
            # If less datasets, update task and raise error
            patch_data = {'log': f"Found only {len(dsets)} video files, user required {n_cams}.",
                          'version': version, 'status': 'Errored'}
            one.alyx.rest('tasks', 'partial_update', id=tdict['id'], data=patch_data)
            status = -1
        else:
            # create the task instance and run it, update task
            task = TaskDLC(session_path, one=one, taskid=tdict['id'], machine=machine)
            status = task.run(n_cams=n_cams, version=version, frames=frames)
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
