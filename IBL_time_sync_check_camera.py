from pylab import *
from ibllib.io import raw_data_loaders as raw
import csv

def convert_pgts(time):
    """Convert PointGray cameras timestamps to seconds.
    Use convert then uncycle"""
    #offset = time & 0xFFF
    cycle1 = (time >> 12) & 0x1FFF
    cycle2 = (time >> 25) & 0x7F
    seconds = cycle2 + cycle1 / 8000.
    return seconds

def uncycle_pgts(time):
    """Unwrap the converted seconds of a PointGray camera timestamp series."""
    cycles = np.insert(np.diff(time) < 0, 0, False)
    cycleindex = np.cumsum(cycles)
    return time + cycleindex * 128

def get_time_stamp_series(port1_source,ssv_source):

 #port1_source='/home/mic/time_sync/' #jasonable needs to be some  
 #ssv_source='/home/mic/time_sync/_iblrig_leftCamera.timestamps.551d3ebb-5061-443f-9d2f-28fa0d3778b8.ssv'

 '''
 get port1 camera time stamps in [sec] for each video frame (came from ttl signal from camera to bpod
 '''
 d=raw.load_data(port1_source)
 port1_times=[]
 for trial in d:
  port1_times.append(trial['behavior_data']['Events timestamps']['Port1In']) 

 ''' 
 get time stamps from ssv file, from camera clock, saved by bonsai
 '''

 with open(ssv_source,'r') as csv_file: 
  csv_reader = csv.reader(csv_file, delimiter=' ') 
  ssv_times=array([line for line in csv_reader])

 ssv_times_sec=[convert_pgts(int(time)) for time in ssv_times[:,0]]

 return list(flatten(port1_times)),uncycle_pgts(ssv_times_sec)


def find_nearest(array, value):
 array = asarray(array)
 idx = (abs(array - value)).argmin()
 return array[idx]


def check_time_delay_between_camerastamps_and_bpodstamps(ssv,port1):

  port1,ssv=get_time_stamp_series()
  start_idx=port1.index(find_nearest(array(port1), ssv[0])) #where camera recording begins
  matched_ssv=[find_nearest(array(ssv), value) for value in port1[start_idx:]]

  length_port1=port1[start_idx:][-1]-port1[start_idx:][0]
  length_matched_ssv=matched_ssv[-1]-matched_ssv[0]

  print('time difference of port1 and matched_ssv for %s sec is %s sec' %(length_port1,length_matched_ssv - length_port1)
