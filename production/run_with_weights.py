import deeplabcut
from pathlib import Path
from oneibl.one import ONE
from dlc import dlc
from weights import download_weights_flatiron

dlc_path = download_weights_flatiron(version_date='2021-02-15')
eid = "8a039e2b-637e-45ed-8da6-0641924626f0"

one = ONE()
data_path = one.load(eid, dataset_types=['_iblrig_Camera.raw'], download_only=True)

alf_files = dlc(data_path[1], dlc_path)
