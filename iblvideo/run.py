from datetime import datetime
from oneibl.one import ONE
from oneibl.patcher import FTPPatcher
from .choiceworld import dlc
from .weights import download_weights
from . import __version__


def run_session(session_id, version=__version__, one=None, ftp_patcher=None):
    # Create ONE and FTPPatcher instance if none are given
    if one is None:
        one = ONE()
    if ftp_patcher is None:
        ftp_patcher = FTPPatcher(one=one)

    # Download weights into ONE Cache diretory under 'resources/DLC' if not exist
    path_dlc = download_weights(version=version)

    # Download camera files
    files_mp4 = one.load(session_id, dataset_types=['_iblrig_Camera.raw'], download_only=True)

    # Find task for session
    task = one.alyx.rest('tasks', 'list', django=f"name,EphysDLC,session__id,{session_id}")[0]

    # Log starttime and set status on Alyx to running
    start_time = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    one.alyx.rest('tasks', 'partial_update', id=task['id'],
                  data={'status': 'Started', 'version': f'{version}',
                        'log': f'Started {start_time}'})

    # Run dlc on all three videos and upload data, set status to complete
    try:
        for cam in range(len(files_mp4)):
            dlc_result = dlc(files_mp4[cam], path_dlc)
            end_time = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
            ftp_patcher.create_dataset(path=dlc_result)
        one.alyx.rest('tasks', 'partial_update', id=task['id'],
                      data={'status': 'Complete',
                            'log': f'Started  {start_time}\nFinished {end_time}'})
    # Catch any exceptions and log them, set status to errored
    except Exception as e:
        raise
        one.alyx.rest('tasks', 'partial_update', id=task['id'],
                      data={'status': 'Errored', 'log': f'{e}'})


def run_queue(version=__version__):
    """Run the entire queue of DLC tasks on Alyx."""

    # Create ONE and FTPPatcher instances
    one = ONE()
    ftp_patcher = FTPPatcher(one=one)

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
                          data={'status': 'Errored', 'log': f"Only {len(dsets)} camera datasets"})

    # On list of sessions, download, run DLC, upload
    for session_id in sessions:
        run_session(session_id, version=version, one=one, ftp_patcher=ftp_patcher)
