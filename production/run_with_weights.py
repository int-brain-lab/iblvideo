"""Script to run DLC on AWS."""
import deeplabcut  # deeplabcut imported first thing to avoid backend error
from oneibl.one import ONE
from dlc_parallel import dlc_parallel
from weights import download_weights_flatiron

path_dlc = download_weights_flatiron(version_date='2021-02-15')
eid = "8a039e2b-637e-45ed-8da6-0641924626f0"

one = ONE()
files_mp4 = one.load(eid, dataset_types=['_iblrig_Camera.raw'],
                     download_only=True)

alf_files = dlc_parallel(files_mp4[0], path_dlc)
print(alf_files)
