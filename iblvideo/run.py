import logging
import os
import traceback
import time
import cv2
import shutil

from glob import glob
from datetime import datetime
from collections import OrderedDict

import numpy as np

from one.api import ONE
from ibllib.pipes.ephys_preprocessing import EphysDLC
from ibllib.oneibl.patcher import FTPPatcher
from ibllib.misc import check_nvidia_driver

_logger = logging.getLogger('ibllib')


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
        # create the task instance and run it, update task
        task = EphysDLC(session_path, one=one, taskid=tdict['id'], machine=machine, location=location, **kwargs)
        status = task.run(cams=cams, overwrite=overwrite)
        patch_data = {'time_elapsed_secs': task.time_elapsed_secs, 'log': task.log,
                      'status': 'Errored' if status == -1 else 'Complete'}
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
