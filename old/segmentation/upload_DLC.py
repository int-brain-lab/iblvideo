from pathlib import Path
import shutil
import alf.io
from brainbox.io import parquet
import numpy as np
from oneibl.patcher import FTPPatcher
from oneibl.one import ONE
# the DMZ is a FTP server installed on the AWS instance test.alyx.internationalbrainlab.org iblftp
one = ONE(base_url="https://alyx.internationalbrainlab.org")
root_path = '/home/mic/DLC_results3'
ftp_patcher = FTPPatcher(one=one)
root_path = Path(root_path)
c = 0
for ses_info in list(root_path.rglob('session_info.txt')):
    raw_video_path = ses_info.parent
    if raw_video_path.joinpath('patched.flag').exists():
        print('skipping: ', raw_video_path)
        continue
    # move all files into a subject/date/number folder for registration
    eid = ses_info.parts[-2]
    sdn = one.path_from_eid(eid).parts[-3:]  # subject date number
    session_path = raw_video_path.joinpath(*sdn)
    alf_path = session_path.joinpath('alf')
    alf_path.mkdir(exist_ok=True, parents=True)
    # convert all the dlc tables in parquet format for registration
    for df in raw_video_path.glob("*.dlc.npy"):
        object = df.name.split('.')[0].replace('_ibl_', '')
        dlc_table = alf.io.load_object(raw_video_path, object)
        parquet.save(alf_path.joinpath(df.name).with_suffix('.pqt'), dlc_table.to_df())
        # shutil.move(df, alf_path.joinpath(df.name))
    files2register = list(alf_path.rglob("*.dlc.pqt"))
    c += len(files2register)
    ftp_patcher.create_dataset(path=files2register)
    raw_video_path.joinpath('patched.flag').touch()
