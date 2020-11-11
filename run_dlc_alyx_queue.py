import logging
from oneibl.one import ONE

_logger = logging.getLogger('ibllib')
one = ONE()



tasks = one.alyx.rest('tasks', 'list', status='Empty', name='EphysDLC')

##
FIRST = 2
for i, task in enumerate(tasks):
    if  i < FIRST:
        continue
    eid = task['session']
    session_path = one.path_from_eid(eid)
    # eid = '4b7fbad4-f6de-43b4-9b15-c7c7ef44db4b'

    vfiles = one.alyx.rest('datasets', 'list', session=eid, django="dataset_type__name,_iblrig_Camera.raw")
    # we only care about the ones that have 3 videos for now
    if len(vfiles) == 0:
        _logger.warning(f"{eid} incorrect number of videos ({len(vfiles)}) {session_path} ")
        continue
    if len(vfiles) != 3:
        _logger.error(f"{eid} incorrect number of videos ({len(vfiles)}) {session_path} ")
        continue

    _logger.info(f"{eid} incorrect number of videos ({len(vfiles)}) {session_path} ")
    break
    one.list(eid)
    vfiles = one.load(eid, dataset_types=['_iblrig_Camera.raw'], download_only=True)

