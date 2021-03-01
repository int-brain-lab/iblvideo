import logging

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
        if len(files_mp4) != 3:
            _logger.error(f'Found only {len(files_mp4)} video files.')
            return
        else:
            _logger.info('Found 3 video files.')
            # Run dlc on all three videos and upload data, set status to complete
            out_files = []
            for cam in range(len(files_mp4)):
                dlc_result = dlc(files_mp4[cam], path_dlc)
                out_files.append(dlc_result)
                _logger.info(dlc_result)
            [_logger.info(f) for f in out_files]
            pqts = list(self.session_path.joinpath('alf').glob('*.pqt'))
            return pqts


def run_session(session_id, one=None, version=__version__):
    """
    Run DLC on a single session in the database.

    :param session_id: Alyx eID of session to run
    :param one: ONE instance to use for query (optional)
    :param version: Version of iblvideo / DLC weights to use (default is current version)
    :return task: ibllib task instance
    """
    # Create ONE instance if none are given
    if one is None:
        one = ONE()
    # Download camera files
    one.load(session_id, dataset_types=['_iblrig_Camera.raw'], download_only=True)
    session_path = one.path_from_eid(session_id)
    # Find task for session
    tdict = one.alyx.rest('tasks', 'list',
                          django=f"name__icontains,DLC,session__id,{session_id}")[0]
    # create the task instance and run it
    task = TaskDLC(session_path, one=one, taskid=tdict['id'])
    status = task.run(version=version)
    patch_data = {'time_elapsed_secs': task.time_elapsed_secs, 'log': task.log,
                  'version': version, 'status': 'Errored' if status == -1 else 'Complete'}
    # register the data using the FTP patcher
    one.alyx.rest('tasks', 'partial_update', id=tdict['id'], data=patch_data)
    if task.outputs:
        # it is safer to instantiate the FTP right before transfer as there may be a time-out error
        ftp_patcher = FTPPatcher(one=one)
        ftp_patcher.create_dataset(path=task.outputs)
    return task


def run_queue(version=__version__):
    """
    Run the entire queue of DLC tasks on Alyx.

    :param version: Version of iblvideo / DLC weights to use (default is current version)
    """

    # Create ONE instance
    one = ONE()

    # Find EphysDLC tasks that have not been run
    tasks = one.alyx.rest('tasks', 'list', status='Empty', name='EphysDLC')

    # Find those that have all three cameras
    sessions = []
    for task in tasks:
        session = task['session']
        dsets = one.alyx.rest('datasets', 'list', django=(f'session__id,{session},'
                                                          'data_format__name,mp4,'
                                                          'name__icontains,camera'))

        # If there are three cam datasets, add to session list
        if len(dsets) == 3:
            sessions.append(session)

        # Else set status to errored and write in log
        else:
            one.alyx.rest('tasks', 'partial_update', id=task['id'],
                          data={'status': 'Errored',
                                'log': f"Found only {len(dsets)} video files."})

    # On list of sessions, download, run DLC, upload
    for session_id in sessions:
        run_session(session_id, one=one, version=version)
