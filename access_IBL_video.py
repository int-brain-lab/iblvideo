from oneibl.one import ONE
from pylab import *
one = ONE()

def get_latest_n_videos_from_all_labs():
 
 '''
 This function gets the n most recent videos for each lab
 and downloads them locally in the ONE folder,
 using one.load
 '''

 n=2 #number of most recent videos to download 
 
 for lab in one.list(keyword='lab'):
  
  sessions, details = one.search(dataset_types='_iblrig_leftCamera.raw', lab=lab, 
  details=True,date_range=['2019-03-01', '2030-04-10'])
  
  try:
   sessions_with_vids=[]
   for i in range(len(sessions)): #check if there are downloadable videos
    urls=[d['data_url'] for d in details[i]['data_dataset_session_related']]
    urls_=[x for x in urls if x is not None]
    vids=[x for x in urls_ if '_iblrig_leftCamera.raw' in x]
    if vids!=[]:
     sessions_with_vids.append([sessions[i],details[i]['end_time']])
    
   print('%s vids to download for %s' %(len(sessions_with_vids),lab))
   #download only the n most recent videos
   s=list(reversed(sorted(sessions_with_vids,key=lambda x: x[1])))
   print(sessions_with_vids[:n])

   for j in s[:n]:
    one.load(j[0],dataset_types='_iblrig_leftCamera.raw')
   
   
  except:
   print('no vids for %s' %(lab)) 
  
