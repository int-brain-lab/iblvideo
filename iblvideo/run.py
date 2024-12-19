import logging
import traceback
import shutil

from glob import glob
from datetime import datetime
from packaging.version import parse

from one.api import ONE
from ibllib.pipes.video_tasks import DLC as EphysDLC
from ibllib.oneibl.patcher import FTPPatcher
from ibllib.misc import check_nvidia_driver
from iblvideo import __version__

_logger = logging.getLogger('ibllib')

status_dict = {0: 'Complete',
               -1: 'Errored',
               -2: 'Waiting',
               -3: 'Incomplete'}

# TODO: make compatible with Lightning Pose


def run_session(session_id, machine=None, cams=None, one=None, remove_videos=True, overwrite=True,
                location='remote', **kwargs):
    """
    Run DLC on a single session in the database.

    :param session_id: Alyx eID of session to run
    :param machine: Tag for the machine this job ran on (string)
    :param cams: List of labels of videos to run dlc and motion energy on. Valid labels are 'left', 'body' and 'right'.
                 Default is to run these three.
    :param one: ONE instance to use for query (optional)
    :param remove_videos: Whether to remove the local raw_video_data after DLC (default is True)
    :param overwrite: whether to overwrite existing outputs of previous runs (default is True)
    :param kwargs: Additional keyword arguments to be passed to EphysDLC

    :return status: final status of the task
    """
    # Check if the GPU is addressable
    check_nvidia_driver()
    try:
        # Create ONE instance if none is given
        one = one or ONE()
        session_path = one.eid2path(session_id)
        tdict = one.alyx.rest('tasks', 'list', django=f"name__icontains,DLC,session__id,{session_id}", no_cache=True)[0]
    except IndexError:
        print(f"No DLC task found for session {session_id}")
        return -1

    # Catch all errors that are not caught inside run function and put them in the log
    try:
        # set a flag in local session folder to later resume if interrupted
        session_path.mkdir(parents=True, exist_ok=True)
        session_path.joinpath('dlc_started').touch(exist_ok=True)
        # create the task instance
        task = EphysDLC(session_path, one=one, taskid=tdict['id'], machine=machine, location=location, **kwargs)
        # Overwrite the signature with the actual cameras needed. Since this is a class attribute, once we start
        # doing this we have to do it every single time as it will transfer to the next task run
        cams = cams or ['left', 'right', 'body']
        task.signature['input_files'] = [(f'_iblrig_{cam}Camera.raw.mp4', 'raw_video_data', True) for cam in cams]
        task.signature['output_files'] = [(f'_ibl_{cam}Camera.dlc.pqt', 'alf', True) for cam in cams]
        task.signature['output_files'].extend([(f'{cam}Camera.ROIMotionEnergy.npy', 'alf', True) for cam in cams])
        task.signature['output_files'].extend([(f'{cam}ROIMotionEnergy.position.npy', 'alf', True) for cam in cams])
        # Run the task and update on Alyx
        status = task.run(cams=cams, overwrite=overwrite)
        patch_data = {'time_elapsed_secs': task.time_elapsed_secs, 'log': task.log, 'status': status_dict[status]}
        one.alyx.rest('tasks', 'partial_update', id=tdict['id'], data=patch_data, no_cache=True)
        # register the data using the FTP patcher
        if task.outputs:
            # it is safer to instantiate the FTP right before transfer to prevent time-out
            ftp_patcher = FTPPatcher(one=one)
            if len(task.outputs) > 0:
                for output in task.outputs:
                    ftp_patcher.create_dataset(path=output)
                # Update the version only now and only if new outputs are uploaded
                one.alyx.rest('tasks', 'partial_update', id=tdict['id'], data={'version': task.version})
            else:
                _logger.warning("No new outputs computed.")
                one.alyx.rest('tasks', 'partial_update', id=tdict['id'], data={'status': 'Empty'})
        if remove_videos is True:
            shutil.rmtree(session_path.joinpath('raw_video_data'), ignore_errors=True)

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
    return status


def run_queue(machine=None, min_version=__version__, statuses=None, restart_local=True, overwrite=True,
              n_sessions=1000, delta_query=600, **kwargs):
    """
    Run all tasks that have a version below min_version and a status in statuses. By default this
    overwrites pre-exisitng results.

    :param machine: str, tag for the machine this job ran on
    :param min_version: str, rerun all tasks lower than this version (default is current version)
    :param statuses: list, task statuses which should be (re)run
                     (default is ['Empty', 'Waiting', 'Complete', 'Incomplete'])
    :param restart_local: bool, whether to restart interrupted local jobs (default is True)
    :param overwrite: bool, whether overwrite existing outputs of previous runs (default is True)
    :param n_sessions: int, number of sessions to run from queue
    :param delta_query: int, time between querying the database in sec
    :param kwargs: Keyword arguments to be passed to run_session
    """

    one = ONE()
    statuses = statuses or ['Empty', 'Waiting', 'Complete', 'Incomplete']

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
                          (t['version'] is None or
                           parse(t['version'].split('_')[-1]) < parse(min_version.split('_')[-1]))]
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
        status_dict[eid] = run_session(sessions.pop(0), machine=machine, overwrite=overwrite, one=one, **kwargs)
        count += 1

    return status_dict
