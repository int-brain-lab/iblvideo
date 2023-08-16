from ibllib.plots.figures import dlc_qc_plot
from one.api import ONE
from brainbox.io.one import SessionLoader

import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.cross_decomposition import CCA
from scipy.stats import pearsonr, gaussian_kde
from scipy.interpolate import interp1d
from pathlib import Path


'''
to get full dlc QC plot (trial averaged body part/feature positions, 
replace manually dlc files with lp files in correct path position,
then
dlc_qc_plot(one.eid2path(eid),one=one)
'''

plt.ion()

one = ONE(base_url='https://openalyx.internationalbrainlab.org',
          password='international', silent=True)


def load_lp(eid, cam):

    '''
    for a given session and cam, load all lp tracked points
    '''

    pth = one.eid2path(eid) / 'alf'

    pqt = pd.read_parquet(pth / 'lp' / f'_ibl_{cam}Camera.dlc.pqt')
    pqt['times'] = np.load(pth / f'_ibl_{cam}Camera.times.npy')

    return pqt
    
      
def load_dlc(eid, cam):

    '''
    cam in left, right, body
    '''

    # load DLC
    sess_loader = SessionLoader(one, eid)
    sess_loader.load_pose(views=[cam])

    return sess_loader.pose[f'{cam}Camera']


def interp_cca(eid, feature_l, feature_r, lp=False):

    '''
    interpolate the values from the right cam to be at the 
    lower resolution left cam, 
    run cca on the x,y coords and 
    measure the correlation in the top cca dimension
    '''

    if lp:
        dleft = load_lp(eid, 'left')        
        dright = load_lp(eid, 'right')
                
    else:
        dleft = load_dlc(eid, 'left')        
        dright = load_dlc(eid, 'right')
        
    # body part from left cam
    paw_lcam = dleft[[f'{feature_l}_x', f'{feature_l}_y']].to_numpy()
    
    # body part from right cam
    paw_rcam_tmp = dright[[f'{feature_r}_x', f'{feature_r}_y']].to_numpy()

    ifcn0 = interp1d(dright['times'], 
                                 paw_rcam_tmp[:, 0], 
                                 fill_value="extrapolate")
                                 
    ifcn1 = interp1d(dright['times'], 
                                 paw_rcam_tmp[:, 1], 
                                 fill_value="extrapolate")
                                 
    paw_rcam = np.vstack([ifcn0(dleft['times']), 
                          ifcn1(dleft['times'])]).T

    cca = CCA(n_components=1)
    
    # mask away observations with any NaN value
    good = np.bitwise_and(np.sum(np.isnan(paw_lcam),axis=1) == 0,
                          np.sum(np.isnan(paw_rcam),axis=1) == 0)     
    
    paw_lcam_cca, paw_rcam_cca = cca.fit_transform(paw_lcam[good], 
                                                   paw_rcam[good])

    return paw_lcam_cca[:,0], paw_rcam_cca[:,0]


def plot_compare(eid, feature_l, feature_r):

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(5,3))
    
    k = 0
    for lp in [False, True]:
        paw_lcam_cca, paw_rcam_cca = interp_cca(eid, feature_l, 
                                                feature_r, lp=lp)
        
                                      
        ax[k].plot(paw_lcam_cca,paw_rcam_cca, linestyle='', 
                marker='o', markersize=0.01,alpha=0.5, c='k',
                markeredgecolor=None) 
                 
        r, p = pearsonr(paw_lcam_cca,paw_rcam_cca)         
                 
        ax[k].set_title(f'Pearson: {np.round(r,2)}, LP: {lp}')
         
        ax[k].set_xlabel('left cam cca dim')
        ax[k].set_ylabel('right cam cca dim')              
        k += 1

    fig.suptitle(f'leftCam: {feature_l}, rightCam: {feature_r}')
    fig.tight_layout()
    return fig


def plot_bulk():

    '''
    for some eids, save cca plots in folders
    '''

    fpairs = [['nose_tip','nose_tip'],
            ['pupil_top_r','pupil_top_r'],
            ['pupil_right_r','pupil_right_r'],
            ['pupil_bottom_r','pupil_bottom_r'],
            ['pupil_left_r','pupil_left_r'],
            ['paw_l','paw_r'],
            ['paw_r','paw_l'],
            ['tongue_end_l','tongue_end_r'],
            ['tongue_end_r','tongue_end_l']] 

    eids = ['51e53aff-1d5d-4182-a684-aba783d50ae5', 
            'dda5fc59-f09a-4256-9fb5-66c67667a466',
            'f312aaec-3b6f-44b3-86b4-3a0c119c0438', 
            'ee40aece-cffd-4edb-a4b6-155f158c666a']

    plt.ioff()
    for eid in eids:
    
        p = Path(f'/home/mic/DLC_LP/{eid}')
        p.mkdir(parents=True, exist_ok=True)
        
        for fpair in fpairs:
            fig = plot_compare(eid, fpair[0], fpair[1]) 
            fig.savefig(p / f'{fpair[0]}_{fpair[1]}.png')
            plt.close()

    plt.ion()






