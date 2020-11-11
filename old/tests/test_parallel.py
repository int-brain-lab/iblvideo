from pathlib import Path

from segmentation.choiceworld import dlc_parallel, dlc

dlc_path = "/home/olivier/Documents/PYTHON/iblscripts/deploy/serverpc/dlc/weights/2020-01-24"
files_mp4 = list(Path("/datadisk/Data/IntegrationTests/dlc/videos").rglob('*.mp4'))

# # baseline non-parallel 789.543 secs
# for file_mp4 in files_mp4:
#     dlc(file_mp4, dlc_path)

# # baseline parallel 628.589 (80% of original)
dlc_parallel(files_mp4, dlc_path)
