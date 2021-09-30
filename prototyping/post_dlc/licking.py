import numpy as np
from one.api import ONE


def get_dlc_XYs(eid, video_type, query_type='remote'):

    #video_type = 'left'    
    Times = one.load_dataset(eid,f'alf/_ibl_{video_type}Camera.times.npy',
                             query_type=query_type) 
    cam = one.load_dataset(eid,f'alf/_ibl_{video_type}Camera.dlc.pqt', 
                           query_type=query_type)
    points = np.unique(['_'.join(x.split('_')[:-1]) for x in cam.keys()])

    # Set values to nan if likelyhood is too low # for pqt: .to_numpy()
    XYs = {}
    for point in points:
        x = np.ma.masked_where(
            cam[point + '_likelihood'] < 0.9, cam[point + '_x'])
        x = x.filled(np.nan)
        y = np.ma.masked_where(
            cam[point + '_likelihood'] < 0.9, cam[point + '_y'])
        y = y.filled(np.nan)
        XYs[point] = np.array(
            [x, y])    

    return Times, XYs  


def get_licks(XYs):

    '''
    define a frame as a lick frame if
    x or y for left or right tongue point
    change more than half the sdt of the diff
    '''  
    
    licks = []
    for point in ['tongue_end_l', 'tongue_end_r']:
        for c in XYs[point]:
           thr = np.nanstd(np.diff(c))/4
           licks.append(set(np.where(abs(np.diff(c))>thr)[0]))
    return sorted(list(set.union(*licks))) 



if __name__ == "__main__":    

    # New dataset type 'alf/_ibl_post_dlc.pqt'
    # that should contain camera specific outputs and also
    # non-camera specific ones, such as 3d paw location and licks
    one = ONE()    
    eid = '572a95d1-39ca-42e1-8424-5c9ffcb2df87'

    times_l, XYs_l = get_dlc_XYs(eid, 'left')
    times_r, XYs_r = get_dlc_XYs(eid, 'right')    
    
    DLC = {'left':[times_l, XYs_l], 'right':[times_r, XYs_r]}
    
    # get licks using both cameras    
    lick_times = []
    for video_type in ['right','left']:
        times, XYs = DLC[video_type]
        r = get_licks(XYs)
        idx = np.where(np.array(r)<len(times))[0][-1]            
        lick_times.append(times[r[:idx]])
    
    lick_times = sorted(np.concatenate(lick_times))
