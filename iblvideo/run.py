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

_logger = logging.getLogger('ibllib')


# re-using the Task class allows to not re-write all the logging, error management
# and automatic settings of task statuses in the database
class TaskDLC(tasks.Task):
    gpu = 1
    cpu = 4
    io_charge = 90
    level = 0

    def _run(self, machine=None, n_cams=3, version=__version__):
        if machine is not None:
            _logger.info(f'Running on {machine}')

        session_id = self.one.eid_from_path(self.session_path)
        # Check for existing dlc data
        dlc_data = self.one.alyx.rest('datasets', 'list', django=(f'session__id,{session_id},'
                                                                  'name__icontains,dlc'))
        if len(dlc_data) != n_cams or self.overwrite is True:
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
        dlc_pqts = list(self.session_path.joinpath('alf').glob('*dlc.pqt'))
        # Run DLC QC
        ###### TODO #########
        # If QC passes, compute motion energy
        for cam in dlc_pqts:
            ### figure out video label
            label = label_from_path(dlc_pqts)
            me_result = motion_energy(session_id, label, one=self.one)
            _logger.info(dlc_result)
        return dlc_pqts


def run_session(session_id, machine=None, n_cams=3, one=None, version=__version__,
                remove_videos=True, rerun_dlc=False):
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
        if one is None:
            one = ONE()
        session_path = one.path_from_eid(session_id)
        tdict = one.alyx.rest('tasks', 'list',
                              django=f"name__icontains,DLC,session__id,{session_id}")[0]

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
            task = TaskDLC(session_path, one=one, taskid=tdict['id'])
            status = task.run(machine=machine, n_cams=n_cams, version=version)
            patch_data = {'time_elapsed_secs': task.time_elapsed_secs, 'log': task.log,
                          'version': version, 'status': 'Errored' if status == -1 else 'Complete'}
            one.alyx.rest('tasks', 'partial_update', id=tdict['id'], data=patch_data)
            # register the data using the FTP patcher
            if task.outputs:
                # it is safer to instantiate the FTP right before transfer to prevent time-out
                ftp_patcher = FTPPatcher(one=one)
                ftp_patcher.create_dataset(path=task.outputs)

            if remove_videos is True:
                shutil.rmtree(session_path.joinpath('raw_video_data'), ignore_errors=True)

    except BaseException:
        patch_data = {'log': traceback.format_exc(),
                      'version': version, 'status': 'Errored'}
        one.alyx.rest('tasks', 'partial_update', id=tdict['id'], data=patch_data)
        status = -1

    return status


def run_queue(machine=None, n_sessions=np.inf, version=__version__, delta_query=600):
    """
    Run the entire queue, or n_sessions, of DLC tasks on Alyx.

    :param machine: Tag for the machine this job ran on (string)
    :param n_sessions: Number of sessions to run from queue (default is run whole queue)
    :param version: Version of iblvideo / DLC weights to use (default is current version)
    :param delta_query: Time between querying the database for Empty tasks, in sec
    """

    # Create ONE instance
    one = ONE()

    status_dict = {}

    # Loop until n_sessions is reached or something breaks
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
        status_dict[eid] = run_session(sessions.pop(0), machine=machine, one=one, version=version)
        count += 1

    return status_dict


# def run_session_qc_motion(session_id, machine=None, n_cams=3, one=None, version=__version__):
#     """
#     Run DLC, QC and, if QC passes, motion energy on a single session in the database.
#
#     :param session_id: Alyx eID of session to run
#     :param machine: Tag for the machine this job ran on (string)
#     :param n_cams: Minimum number of camera datasets required
#     :param one: ONE instance to use for query (optional)
#     :param version: Version of iblvideo / DLC weights to use (default is current version)
#     :return :
#     """
#
#     # run session but do not remove local data
#     status = run_session(session_id, machine, n_cams, one, version, remove_data=False)
#
#     if status == -1:
#         out = 'DLC failed'
#     else:
#         # run qc on all videos
#         # TODO: maybe better to do this per video, so that it doesn't depend on three vids existing
#         qc = run_all_qc(session_id, update=True)
#
#         # run motion energy for all videos that pass qc
#         out = {}
#         for video_type in ['body', 'left', 'right']:
#             if all(x == 'PASS' for x in qc[video_type].metrics.values()):
#                 # run motion energy
#             else:
#
#     return out
