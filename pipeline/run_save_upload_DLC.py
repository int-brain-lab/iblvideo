from oneibl.one import ONE
from pathlib import Path
import cv2
import time
import numpy as np
import os
import os.path
from shutil import copyfile
import shutil
import alf.io
from brainbox.io import parquet
from oneibl.patcher import FTPPatcher
#import matplotlib
#matplotlib.use('Agg')



def get_sessions0():
    one = ONE()
    traj_traced = one.alyx.rest('trajectories', 'list', provenance='Planned',
                         django='probe_insertion__session__project__name__'
                                'icontains,ibl_neuropixel_brainwide_01,'
                                'probe_insertion__session__qc__lt,50,'
                                'probe_insertion__session__extended_qc__behavior,1,'
                                'probe_insertion__json__extended_qc__tracing_exists,True,'
                                '~probe_insertion__session__extended_qc___task_stimOn_goCue_delays__lt,0.9,'
                                '~probe_insertion__session__extended_qc___task_response_feedback_delays__lt,0.9,'
                                '~probe_insertion__session__extended_qc___task_response_stimFreeze_delays__lt,0.9,'
                                '~probe_insertion__session__extended_qc___task_wheel_move_before_feedback__lt,0.9,'
                                '~probe_insertion__session__extended_qc___task_wheel_freeze_during_quiescence__lt,0.9,'
                                '~probe_insertion__session__extended_qc___task_error_trial_event_sequence__lt,0.9,'
                                '~probe_insertion__session__extended_qc___task_correct_trial_event_sequence__lt,0.9,'
                                '~probe_insertion__session__extended_qc___task_n_trial_events__lt,0.9,'
                                '~probe_insertion__session__extended_qc___task_reward_volumes__lt,0.9,'
                                '~probe_insertion__session__extended_qc___task_reward_volume_set__lt,0.9,'
                                '~probe_insertion__session__extended_qc___task_stimulus_move_before_goCue__lt,0.9,'
                                '~probe_insertion__session__extended_qc___task_audio_pre_trial__lt,0.9')
    eids = [[x['session']['id'],x['probe_name']] for x in traj_traced]
    return eids
    
    
def get_sessions1():
    '''
    updated alignement
    ''' 
    one = ONE()
    sessions = one.alyx.rest('insertions', 'list', django='json__extended_qc__alignment_resolved,True')
    return [[x['session_info']['id'],x['name']] for x in sessions] 



def get_eids_DLC():

    one = ONE()
    s0 = get_sessions0()
    s1 = get_sessions1()
    s0_ = set([x[0] for x in s0])
    s1_ = set([x[0] for x in s1])
    r = list(s0_.union(s1_))
    to_do = []
    for eid in r:
        a = one.list(eid, 'dataset-types')
        if 'camera.dlc' in a:
            continue

        if not '_iblrig_Camera.raw' in a:
            continue
        to_do.append(eid)
    return to_do  






Path_dlc='/home/mic/DLC_labels'
fname = '/home/mic/Dropbox/scripts/IBL/DLC_pipeline/DLC_pipeline_olivier.py'
exec(compile(open(fname, "rb").read(),fname, 'exec'))



'''
to use both GPU, run in two terminals, in each terminal first set:
os.environ['CUDA_VISIBLE_DEVICES'] = str(gputouse)

Make sure you have a recent ffmpeg:
https://linuxconfig.org/ubuntu-20-04-ffmpeg-installation
'''


#one = ONE()
#sessions = one.alyx.rest('insertions', 'list', django='json__extended_qc__alignment_resolved,True')
#eids = [x['session_info']['id'] for x in sessions] 

number = 11 # batch number 
eids = [ '7f6b86f9-879a-4ea2-8531-294a221af5d0',
 'fc14c0d6-51cf-48ba-b326-56ed5a9420c3',
 '7cb81727-2097-4b52-b480-c89867b5b34c',
 '9eec761e-9762-4897-b308-a3a08c311e69',
 '28741f91-c837-4147-939e-918d38d849f2',
 '1928bf72-2002-46a6-8930-728420402e01',
 '9fe512b8-92a8-4642-83b6-01158ab66c3c',
 '4a45c8ba-db6f-4f11-9403-56e06a33dfa4',
 '54238fd6-d2d0-4408-b1a9-d19d24fd29ce',
 'b10ed1ba-2099-42c4-bee3-d053eb594f09',
 '2e22c1fc-eec6-4856-85a0-7dba8668f646',
 'a82800ce-f4e3-4464-9b80-4c3d6fade333',
 '90c61c38-b9fd-4cc3-9795-29160d2f8e55',
 'dcceebe5-4589-44df-a1c1-9fa33e779727',
 'ee5c418c-e7fa-431d-8796-b2033e000b75',
 'cde63527-7f5a-4cc3-8ac2-215d82e7da26',
 'ee8b36de-779f-4dea-901f-e0141c95722b',
 '56b57c38-2699-4091-90a8-aba35103155e',
 'd832d9f7-c96a-4f63-8921-516ba4a7b61f',
 '671c7ea7-6726-4fbe-adeb-f89c2c8e489b',
 '6fb1e12c-883b-46d1-a745-473cde3232c8',
 '064a7252-8e10-4ad6-b3fd-7a88a2db5463',
 '26aa51ff-968c-42e4-85c8-8ff47d19254d']


def upload_dlc(number):
    one = ONE(base_url="https://alyx.internationalbrainlab.org")
    root_path = f'/home/mic/DLC_results{number}'
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



def run_save_upload_dlc(eids, number):



    one = ONE()

    for eid in eids:
        print('STARTING:', eid, eids.index(eid), 'of', len(eids))
        try:                                   
            dataset_types = ['_iblrig_Camera.raw']
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
                         
            Path(f"/home/mic/DLC_log{number}").mkdir(parents=True, exist_ok=True)           
            np.save(f'/home/mic/DLC_log{number}/DLC_%s.npy' %eid, d2)                      
            print(eid, eids.index(eid), 'of', len(eids), 'done')        
        except Exception as e:
            print(f'{eid} did not run', e)
            continue  
            
    '''
    Next save the results in another format (historical accident)
    then upload
    '''            

    eids = [x[4:-4] for x in os.listdir(f'/home/mic/DLC_log{number}')]

    for eid in eids:

        try:        
            video_data = one.path_from_eid(eid) / 'raw_video_data'
            for video_type in ['body','left','right']: 

                json_path = video_data / str('_ibl_%sCamera.dlc.metadata.json' % video_type)
                dlc_path = video_data / str('_ibl_%sCamera.dlc.npy' % video_type)
                if not os.path.isfile(dlc_path):
                    continue
                Path(f"/home/mic/DLC_results{number}").mkdir(parents=True, exist_ok=True)  
                direct = Path(f'/home/mic/DLC_results{number}/%s' %eid)          
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
        except:
            print(eid,"didn't find path") 

    upload_dlc(number)



run_save_upload_dlc(eids, number)





