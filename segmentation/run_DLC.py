from oneibl.one import ONE
from pathlib import Path
import cv2
import time
import numpy as np
import os
import os.path

'''
to use both GPU, run in two terminals, in each terminal first set:
os.environ['CUDA_VISIBLE_DEVICES'] = str(gputouse)

Make sure you have a recent ffmpeg:
https://linuxconfig.org/ubuntu-20-04-ffmpeg-installation
'''


fname = '/home/mic/Dropbox/scripts/IBL/DLC_pipeline_olivier.py'
exec(compile(open(fname, "rb").read(),fname, 'exec'))

one = ONE()


dataset_types = ['_iblrig_Camera.raw']# ,
#                 'camera.times',
#                 'wheel.position',
#                 'wheel.timestamps',
#                 'trials.intervals']

#eids, sessions = one.search(task_protocol='ephysChoiceworld', dataset_types=dataset_types, details=True)
#os.environ['CUDA_VISIBLE_DEVICES'] = str(1)
#eids = ['dfd8e7df-dc51-4589-b6ca-7baccfeb94b4', '034e726f-b35f-41e0-8d6c-a22cc32391fb','4b00df29-3769-43be-bb40-128b1cba6d35']

#eids = [ '83e77b4b-dfa0-4af9-968b-7ea0c7a0c7e4','3663d82b-f197-4e8b-b299-7b803a155b84']
# Karo eids = [

Path_dlc='/home/mic/DLC_labels'
logD = {} 

#eids = ['dfd8e7df-dc51-4589-b6ca-7baccfeb94b4']

def run_dlc(eid):

    a = one.list(eid, 'dataset-types')

    if not all([i in a for i in dataset_types]):
        print(eid, 'No vids for this sessions')
        return 


    D = one.load(eid, dataset_types=dataset_types, dclass_output=True)
    alf_path = Path(D.local_path[0]).parent.parent / 'alf'

    video_data = alf_path.parent / 'raw_video_data'

    d2 = {}
    for video_type in ['body','left','right']: 

        video_path = video_data / str('_iblrig_%sCamera.raw.mp4' % video_type)
        if not os.path.isfile(video_path):
            continue

        cap = cv2.VideoCapture(video_path.as_uri())
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(3)), int(cap.get(4)))
        cap.release()

        start_T = time.time() 
        dlc(video_path, path_dlc=Path_dlc)
        os.remove(video_path)
        end_T = time.time()         
        d2[video_type] = [length, fps, size, end_T - start_T]
                
    np.save('/home/mic/DLC_log4/DLC_%s.npy' %eid, d2)


#eids = one.search(task_protocol='ephysChoiceworld', dataset_types=['_iblrig_Camera.raw'])
#['dfd8e7df-dc51-4589-b6ca-7baccfeb94b4',
# 'd119b389-c40d-4e7c-8a07-0c7f35edef43',
# '034e726f-b35f-41e0-8d6c-a22cc32391fb',
# '4b00df29-3769-43be-bb40-128b1cba6d35',
# '83e77b4b-dfa0-4af9-968b-7ea0c7a0c7e4',
# '5386aba9-9b97-4557-abcd-abc2da66b863',
# '85dc2ebd-8aaf-46b0-9284-a197aee8b16f',
# '3663d82b-f197-4e8b-b299-7b803a155b84',
# '79de526f-aed6-4106-8c26-5dfdfa50ce86',
# 'd42bb88e-add2-414d-a60a-a3efd66acd2a',
# 'ab583ab8-08bd-4ebc-a0b8-5d40af551068',
# 'ecb5520d-1358-434c-95ec-93687ecd1396',
# '74bae29c-f614-4abe-8066-c4d83d7da143',
# '810b1e07-009e-4ebe-930a-915e4cd8ece4',
# 'eef82e27-c20e-48da-b4b7-c443031649e3',
# 'c8e60637-de79-4334-8daf-d35f18070c29',
# '0deb75fb-9088-42d9-b744-012fb8fc4afb',
# '12dc8b34-b18e-4cdd-90a9-da134a9be79c',
# '3ce452b3-57b4-40c9-885d-1b814036e936',
# '1538493d-226a-46f7-b428-59ce5f43f0f9',
# '193fe7a8-4eb5-4f3e-815a-0c45864ddd77',
# '510b1a50-825d-44ce-86f6-9678f5396e02',
# '2d5f6d81-38c4-4bdc-ac3c-302ea4d5f46e',
# 'cb2ad999-a6cb-42ff-bf71-1774c57e5308',
# 'b52182e7-39f6-4914-9717-136db589706e',
# 'd33baf74-263c-4b37-a0d0-b79dcb80a764',
# '89f0d6ff-69f4-45bc-b89e-72868abb042a',
# '68dbaf15-9d48-452c-85da-ba43f118045a',
# '2ffd3ed5-477e-4153-9af7-7fdad3c6946b',
# '08e25a6c-4223-462a-a856-a3a587146668',
# 'd839491f-55d8-4cbe-a298-7839208ba12b',
# 'd2918f52-8280-43c0-924b-029b2317e62c',
# 'c99d53e6-c317-4c53-99ba-070b26673ac4',
# '4b7fbad4-f6de-43b4-9b15-c7c7ef44db4b',
# '7e86c0dd-4de2-4052-bf81-9e57684ccb59',
# '53738f95-bd08-4d9d-9133-483fdb19e8da',
# '21e16736-fd59-44c7-b938-9b1333d25da8',
# 'e5fae088-ed96-4d9b-82f9-dfd13c259d52',
# '266c32c3-4f75-4d44-9337-ef12f2980ecc',
# 'c607c5da-534e-4f30-97b3-b1d3e904e9fd',
# 'ee76b915-f649-4156-827a-ab661b761207',
# '920316c5-c471-4196-8db9-0d52cbe55830',
# 'dd4da095-4a99-4bf3-9727-f735077dba66',
# '46b0d871-23d3-4630-8a6b-c79f99b2958c',
# 'b985b86f-e0e1-4d63-afaf-448b91cb4d74',
# 'f3f406bd-e138-44c2-8a02-7f11bf8ce87a',
# 'af5a1a37-9209-4c1e-8d7a-edf39ee4420a',
# '63b83ddf-b7ea-40db-b1e2-93c2a769b6e5',
# '8c2e6449-57f0-4632-9f18-66e6ca90c522',
# '31087c1d-e5b0-4a01-baf0-b26ddf03f3ca',
# '1b966923-de4a-4afd-8ed3-5f6842d9ec29',
# '40e2f1bd-6910-4635-b9a7-1e76771a422e',
# '1d364d2b-e02b-4b5d-869c-11c1a0c8cafc',
# '17231390-9b95-4ec6-806d-b3aae8af76ac',
# 'f354dc45-caef-4e3e-bd42-2c19a5425114',
# '036f7740-08a0-4b27-966c-7b0faa0bef08',
# 'e429f004-5c8e-4798-9a9d-6cb560885b42',
# '6756a3b4-5564-4005-93b5-f74b91f3e494',
# '441f1b7e-05fd-43fd-a9a3-0845097cce87',
# '9ee12372-7945-4cae-a222-97fcc883d014',
# 'b7998d33-ad2c-4acf-a1ca-5624411e2118']

# done:
#['dfd8e7df-dc51-4589-b6ca-7baccfeb94b4',
# 'd119b389-c40d-4e7c-8a07-0c7f35edef43',
# '034e726f-b35f-41e0-8d6c-a22cc32391fb',
# '4b00df29-3769-43be-bb40-128b1cba6d35',
# '83e77b4b-dfa0-4af9-968b-7ea0c7a0c7e4',
# '5386aba9-9b97-4557-abcd-abc2da66b863',
# '85dc2ebd-8aaf-46b0-9284-a197aee8b16f',
# '3663d82b-f197-4e8b-b299-7b803a155b84',
# '79de526f-aed6-4106-8c26-5dfdfa50ce86',
# 'd42bb88e-add2-414d-a60a-a3efd66acd2a',
# 'ab583ab8-08bd-4ebc-a0b8-5d40af551068',
# 'ecb5520d-1358-434c-95ec-93687ecd1396',
# '74bae29c-f614-4abe-8066-c4d83d7da143',
# '810b1e07-009e-4ebe-930a-915e4cd8ece4',
# 'eef82e27-c20e-48da-b4b7-c443031649e3',
# 'c8e60637-de79-4334-8daf-d35f18070c29',
# '0deb75fb-9088-42d9-b744-012fb8fc4afb',
# '12dc8b34-b18e-4cdd-90a9-da134a9be79c',
# '3ce452b3-57b4-40c9-885d-1b814036e936',
# '1538493d-226a-46f7-b428-59ce5f43f0f9',
# '193fe7a8-4eb5-4f3e-815a-0c45864ddd77',
# '510b1a50-825d-44ce-86f6-9678f5396e02',
# '2d5f6d81-38c4-4bdc-ac3c-302ea4d5f46e',
# 'cb2ad999-a6cb-42ff-bf71-1774c57e5308',
# 'b52182e7-39f6-4914-9717-136db589706e',
# 'd33baf74-263c-4b37-a0d0-b79dcb80a764',
# '89f0d6ff-69f4-45bc-b89e-72868abb042a',
# '68dbaf15-9d48-452c-85da-ba43f118045a',
# '2ffd3ed5-477e-4153-9af7-7fdad3c6946b',
# '08e25a6c-4223-462a-a856-a3a587146668']

def check_dims(eid):


    dataset_types = ['_iblrig_Camera.raw','camera.times','camera.dlc'] 
    a = one.list(eid, 'dataset-types') 

    assert all([i in a for i in dataset_types] 
            ), 'For this eid, not all data available' 

    D = one.load(eid, dataset_types=dataset_types, dclass_output=True) 
    alf_path = Path(D.local_path[0]).parent.parent / 'alf' 

    video_data = alf_path.parent / 'raw_video_data' 

    d2 = {} 
    for video_type in ['body','left','right']: 
        video_path = video_data / str('_iblrig_%sCamera.raw.mp4' % video_type) 
        print(video_path)
        if not os.path.isfile(video_path): 
            continue 
             
        cap = cv2.VideoCapture(video_path.as_uri())
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(3)), int(cap.get(4)))
        cap.release()  
             
        cam0 = alf.io.load_object(alf_path, '_iblrig_%sCamera' % video_type)        
        cam1 = alf.io.load_object(video_path.parent, '_iblrig_%sCamera' % video_type)
        cam = {'times':cam0['times'],**cam1}   
          
        os.remove(video_path)            
        d2[video_type] = [length, fps,length,len(cam['times']),length - len(cam['times'])] 
    return eid, d2
             

















