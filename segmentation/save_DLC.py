from oneibl.one import ONE
from pathlib import Path
import cv2
import time
import numpy as np
import os
import os.path
from shutil import copyfile

one = ONE()

#eids= one.search(task_protocol='ephysChoiceworld', dataset_types=['_iblrig_Camera.raw'], details=False)
eids = [x[4:-4] for x in os.listdir('/home/mic/DLC_log4')]

for eid in eids:
    try:
        D = one.load(eid, dataset_types=['_iblrig_Camera.timestamps'], dclass_output=True)
        video_data = Path(D.local_path[0]).parent.parent / 'raw_video_data'

    except:
        print(eid,"didn't find path") 
        continue

    for video_type in ['body','left','right']: 

        json_path = video_data / str('_ibl_%sCamera.dlc.metadata.json' % video_type)
        dlc_path = video_data / str('_ibl_%sCamera.dlc.npy' % video_type)
        if not os.path.isfile(dlc_path):
            continue
        direct = Path('/home/mic/DLC_results3/%s' %eid)          
        direct.mkdir(parents=True, exist_ok=True)
        txt_file = direct / 'session_info.txt'
        if not os.path.isfile(txt_file):
            with open(txt_file, 'a') as the_file:
                the_file.write(str(video_data))  
        
        if os.path.isfile(direct / json_path.name):
            continue        
        
        copyfile(json_path,direct / json_path.name)
        copyfile(dlc_path,direct / dlc_path.name)
        print(eid, video_type)
