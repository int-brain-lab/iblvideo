# dlc env
from segmentation.choiceworld import dlc, dlc_parallel


dlc_path = "/home/olivier/Documents/PYTHON/iblscripts/deploy/serverpc/dlc/weights/2020-01-24"

# full profiling reports @ /home/olivier/.PyCharm2019.3/system/snapshots

# baseline laptop 5082.22 secs - duration 3360 secs - 55.3% tensorflow - dask 4451.732
# mp4 = "/mnt/s0/Data/Patch/churchlandlab/Subjects/CSHL_003/2019-05-10/003/raw_video_data/_iblrig_leftCamera.raw.mp4"

# baseline laptop 83.847 secs - duration 20 secs - 50% tensorflow - dask 71.053 secs
mp4 = "/datadisk/Data/IntegrationTests/dlc/videos/churchlandlab_CSHL_015_2019-11-12_001__iblrig_leftCamera.raw.short.mp4"


dlc(mp4, dlc_path, parallel=True)

dlc_parallel(mp4, dlc_path)
