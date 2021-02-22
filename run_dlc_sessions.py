from oneibl.one import ONE
from oneibl.patcher import FTPPatcher
from choiceworld import dlc
from weights import download_weights_flatiron

DLC_VERSION = '2021-02-15'

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

# Download weights into ONE Cache diretory under 'resources/DLC'
path_dlc = download_weights_flatiron(version_date=DLC_VERSION)

# On list of sessions, download, run DLC, upload
for session in sessions:
    files_mp4 = one.load(session, dataset_types=['_iblrig_Camera.raw'], download_only=True)
    for cam in range(len(files_mp4)):
        dlc_result = dlc(files_mp4[cam], path_dlc)
        ftp_patcher.create_dataset(path=dlc_result)
