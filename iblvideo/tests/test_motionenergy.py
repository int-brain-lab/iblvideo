from oneibl.one import ONE
import numpy as np
from iblvideo.motion_energy import motion_energy

one = ONE()
eid = 'cde63527-7f5a-4cc3-8ac2-215d82e7da26'
session_path = one.path_from_eid(eid)

for label in ['body', 'left', 'right']:
    dlc_pqt = session_path.joinpath(f'alf/_ibl_{label}Camera.dlc.pqt')

    me_file, roi_file = motion_energy(eid, dlc_pqt, frames=None)
    all_frames_me = np.load(me_file)
    all_frames_roi = np.load(roi_file)

    me_file, roi_file = motion_energy(eid, dlc_pqt, frames=100)
    some_frames_me = np.load(me_file)
    some_frames_roi = np.load(roi_file)

    assert all(all_frames_me == some_frames_me)
    assert all(all_frames_roi == some_frames_roi)
# check with and without streaming
