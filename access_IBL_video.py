from oneibl.one import ONE
one = ONE()
from ibllib.misc import pprint


def get_most_recent_video_from_all_labs():
 D=[]
 for lab in one.list(keyword='lab'):
  sessions, details = one.search(dataset_types='_iblrig_leftCamera.raw', lab=lab, 
  details=True,date_range=['2019-03-01', '2019-04-10'])
  try:                                             
   dates=[i['end_time'] for i in details]
   for jj in range(1,5):
    idx=argsort(dates)[-jj] 
    D.append({'lab':lab,'latest session':sessions[idx],'date':dates[idx],'details':details[idx]})  
    one.load(sessions[argsort(dates)[idx]],dataset_types='_iblrig_leftCamera.raw')
   print(lab,len(sessions)) 
  except:                     
   print(lab,len(sessions))  
  
 return D
