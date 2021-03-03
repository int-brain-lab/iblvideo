import logging
import shutil
from datetime import datetime

from oneibl.one import ONE
from oneibl.patcher import FTPPatcher
from iblvideo.choiceworld import dlc
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

    def _run(self, version=__version__):
        # Download weights into ONE Cache directory under 'resources/DLC' if not exist
        path_dlc = download_weights(version=version)
        files_mp4 = list(self.session_path.joinpath('raw_video_data').glob('*.mp4'))
        _logger.info(f'Running DLC on {len(files_mp4)} video files.')
        # Run dlc on all videos
        out_files = []
        for cam in range(len(files_mp4)):
            dlc_result = dlc(files_mp4[cam], path_dlc)
            out_files.append(dlc_result)
            _logger.info(dlc_result)
        pqts = list(self.session_path.joinpath('alf').glob('*.pqt'))
        return pqts


def run_session(session_id, n_cams=3, one=None, version=__version__):
    """
    Run DLC on a single session in the database.

    :param session_id: Alyx eID of session to run
    :param n_cams: Minimum number of camera datasets required
    :param one: ONE instance to use for query (optional)
    :param version: Version of iblvideo / DLC weights to use (default is current version)
    :return task: ibllib task instance
    """
    # Create ONE instance if none are given
    if one is None:
        one = ONE()

    # Find task for session and set to running
    session_path = one.path_from_eid(session_id)
    tdict = one.alyx.rest('tasks', 'list',
                          django=f"name__icontains,DLC,session__id,{session_id}")[0]
    one.alyx.rest('tasks', 'partial_update', id=tdict['id'], data={'status': 'Started'})

    # Before starting to download, check if required number of cameras are available
    dsets = one.alyx.rest('datasets', 'list', django=(f'session__id,{session_id},'
                                                      'data_format__name,mp4,'
                                                      'name__icontains,camera'))

    if len(dsets) < n_cams:
        # If less datasets, update task and raise error
        patch_data = {'log': f"Found only {len(dsets)} video files, user required {n_cams}.",
                      'version': version, 'status': 'Errored'}
        one.alyx.rest('tasks', 'partial_update', id=tdict['id'], data=patch_data)
        raise FileNotFoundError(f"Found only {len(dsets)} video files, user required {n_cams}.")
        status = -1
    else:
        # Download camera files
        one.load(session_id, dataset_types=['_iblrig_Camera.raw'], download_only=True)
        # create the task instance and run it, update task
        task = TaskDLC(session_path, one=one, taskid=tdict['id'])
        status = task.run(version=version)
        patch_data = {'time_elapsed_secs': task.time_elapsed_secs, 'log': task.log,
                      'version': version, 'status': 'Errored' if status == -1 else 'Complete'}
        one.alyx.rest('tasks', 'partial_update', id=tdict['id'], data=patch_data)
        # register the data using the FTP patcher
        if task.outputs:
            # it is safer to instantiate the FTP right before transfer to prevent time-out error
            ftp_patcher = FTPPatcher(one=one)
            ftp_patcher.create_dataset(path=task.outputs)

        shutil.rmtree(session_path.joinpath('raw_video_data'), ignore_errors=True)
    return status


def run_queue(n_sessions=None, version=__version__, delta_query=600):
    """
    Run the entire queue of DLC tasks on Alyx.

    :param n_sessions: Number of sessions to run from queue. If 'None' run all.
    :param version: Version of iblvideo / DLC weights to use (default is current version)
    :param delta_query: Time between querying the database for Empty tasks, in sec
    """

    # Create ONE instance
    one = ONE()

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

        # Run next session in the list
        status = run_session(sessions.pop(0), one=one, version=version)
        count += 1
        # Currently, stop the queue if there is an error
        if status == -1:
            return
    return
