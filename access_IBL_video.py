from oneibl.one import ONE
from ibllib.misc import pprint

one = ONE()

def get_most_recent_video_from_all_labs():

 '''
 this assumes you have oneibl installed and access;
 a couple recent IBL trainig-rig videos will be downloaded 
 '''
 
 for lab in one.list(keyword='lab'):
  sessions, details = one.search(dataset_types='_iblrig_leftCamera.raw', lab=lab, 
  details=True,date_range=['2019-03-01', '2019-04-10'])
  try:                                             
   dates=[i['end_time'] for i in details]
   for jj in range(1,5):
    idx=argsort(dates)[-jj]   
    one.load(sessions[argsort(dates)[idx]],dataset_types='_iblrig_leftCamera.raw')
   print(lab,len(sessions)) 
  except:                     
   print(lab,len(sessions))  
  
