from ibllib.plots.figures import dlc_qc_plot
from one.api import ONE
from brainbox.io.one import SessionLoader
from brainbox.processing import bincount2D

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.cross_decomposition import CCA
from scipy.stats import pearsonr, gaussian_kde
from scipy.interpolate import interp1d
from pathlib import Path
import math


'''
to get full dlc QC plot (trial averaged body part/feature positions, 
replace manually dlc files with lp files in correct path position,
then
dlc_qc_plot(one.eid2path(eid),one=one)
'''

plt.ion()

one = ONE(base_url='https://openalyx.internationalbrainlab.org',
          password='international', silent=True)
          
          
eids = ['51e53aff-1d5d-4182-a684-aba783d50ae5', 
        'dda5fc59-f09a-4256-9fb5-66c67667a466',
        'f312aaec-3b6f-44b3-86b4-3a0c119c0438', 
        'ee40aece-cffd-4edb-a4b6-155f158c666a']


def load_lp(eid, cam, masked=True):

    '''
    for a given session and cam, load all lp tracked points
    '''

    pth = one.eid2path(eid) / 'alf'

    d = pd.read_parquet(pth / 'lp' / f'_ibl_{cam}Camera.dlc.pqt')
    d['times'] = np.load(pth / f'_ibl_{cam}Camera.times.npy')

    if masked:
        points = np.unique(['_'.join(x.split('_')[:-1]) 
                            for x in d.keys()])[1:]
    
        for point in points:
            cond = d[f'{point}_likelihood'] < 0.9
            d.loc[cond, [f'{point}_x', f'{point}_y']] = np.nan

    return d
    
      
def load_dlc(eid, cam, masked=False):

    '''
    cam in left, right, body
    '''

    # load DLC
    sess_loader = SessionLoader(one, eid)
    sess_loader.load_pose(views=[cam])
    d = sess_loader.pose[f'{cam}Camera']

    return d


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


def plot_compare(eid, feature_l, feature_r, ax=None):

    '''
    plot cca for a given session and feature pair
    '''
    

    alone = False
    if not ax:
        alone = True
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(5,3))
    
    k = 0
    for lp in [False, True]:
        paw_lcam_cca, paw_rcam_cca = interp_cca(eid, feature_l, 
                                                feature_r, lp=lp)
                                      
        ax[k].plot(paw_lcam_cca,paw_rcam_cca, linestyle='', 
                marker='o', markersize=0.01,alpha=0.5, c='k',
                markeredgecolor=None) 
                 
        r, p = pearsonr(paw_lcam_cca,paw_rcam_cca)         
        
        if alone:
            ax[k].set_title(f'r: {np.round(r,2)}, LP: {lp}')
        else:         
            ax[k].set_title(f'[{feature_l}, {feature_r}]'
                            f'\n r: {np.round(r,2)}, LP: {lp}')
         
        ax[k].set_xlabel('left cam cca dim')
        ax[k].set_ylabel('right cam cca dim')              
        k += 1

    if alone: 
        fig.suptitle(f"{eid},"
                     f" \n {'/'.join(str(one.eid2path(eid)).split('/')[-5:])}"
                     f" \n leftCam: {feature_l}, rightCam: {feature_r}")
        fig.tight_layout()
        return fig


def plot_bulk(combine=True):

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

    if combine:
        fig, ax = plt.subplots(nrows=3, ncols=6, figsize = (15, 8))


    plt.ioff()    

    for eid in eids:

        p = Path(f'/home/mic/DLC_LP/{eid}')
        p.mkdir(parents=True, exist_ok=True)
        
        k = 0
        col = 0        
        row = 0
        
        for fpair in fpairs:
            if combine:

                plot_compare(eid, fpair[0], fpair[1], 
                             ax=[ax[row,col], ax[row, col+1]])
            
                k += 1
                col += 2
                
                if col%6 == 0:
                    col = 0

                if k%3 == 0:
                    row += 1
                
                            
            else:
                fig = plot_compare(eid, fpair[0], fpair[1]) 
                fig.savefig(p / f'{fpair[0]}_{fpair[1]}.png')
                plt.close()

        if combine:
            fig.suptitle(f"{eid}, cca analysis for left/right cam"
               f" \n {'/'.join(str(one.eid2path(eid)).split('/')[-5:])}")
                            
            fig.tight_layout()
            fig.savefig(p / f'cca_{eid}.png')    

    plt.ion()



def get_licks(d):

    '''
    get lick times from single cam
    '''

    licks = []
    for point in ['tongue_end_l', 'tongue_end_r']:
        for cord in ['x','y']:
           c = d[f'{point}_{cord}']
           thr = np.nanstd(np.diff(c))/4
           licks.append(set(np.where(abs(np.diff(c))>thr)[0]))
           
           
    return d['times'][sorted(list(set.union(*licks)))]
     

def lick_corr(eid, lp=False, plo=False, ax=None):
    
    alone = False
    
    if lp:
        dleft = load_lp(eid, 'left')        
        dright = load_lp(eid, 'right')
                
    else:
        dleft = load_dlc(eid, 'left')        
        dright = load_dlc(eid, 'right')
        
    lt_l = get_licks(dleft)
    lt_r = get_licks(dright)

    # bin licks in bins of size 0.1 sec (empirically determined)
    T_BIN = 0.1
    Rl, tlb, _ = bincount2D(lt_l, np.ones(len(lt_l)), T_BIN)
    Rr, trb, _ = bincount2D(lt_r, np.ones(len(lt_r)), T_BIN)
    
    co = (max(np.correlate(Rr[0], Rl[0], mode='valid')) /
            (np.linalg.norm(Rr[0]) * np.linalg.norm(Rl[0])))
    
    
    if plo:
        if not ax:
            alone = True 
            fig, ax = plt.subplots()
        ax.plot(lt_l, np.ones(len(lt_l)), linestyle='', 
                 marker='o', color='k', label='left')    
        ax.plot(lt_r, np.ones(len(lt_r))*2, linestyle='', 
                 marker='o', color='b', label='right')


        ax.set_xlabel('time [sec]')
        ax.set_ylabel('lick present or not') 
        ax.set_xlim([100,105])
        ax.legend(loc='best')
        
        if alone:
            ax.set_title(f"{eid},"
                         f" \n {'/'.join(str(one.eid2path(eid)).split('/')[-5:])}"
                         f" \n LP:{lp}, right/left corr: {np.round(co,2)}")        
            fig.tight_layout()
            
        else:
            ax.set_title(f" LP:{lp} \n "
                         f"right/left corr: {np.round(co,2)}")            
                    
        
    return co
    

def plot_lick_corr():

    '''
    each 2d point a session, x(y) coord is left/right lick corr for dlc(lp)
    '''
    
    fig, ax = plt.subplots()
    
    subs = [str(one.eid2path(eid)).split('/')[6] for eid in eids]
    
    x = []
    y = []
    for eid in eids:
        x.append(lick_corr(eid, lp=False))  
        y.append(lick_corr(eid, lp=True))    
        
    ax.scatter(x,y)        
    ax.axline((0, 0), slope=1, color="k", linestyle=(0, (5, 5)))
    
    k = 0 
    for sub in subs:
        ax.annotate(sub, xy=(x[k], y[k])) 
        k += 1

    ax.set_xlabel('DLC')
    ax.set_ylabel('LP')
    ax.set_title('lick correlation of left/right cams')


def plot8(eid):

    '''
    For a given session, plot 4 panels for DLC and 4 for LP;
    top row of panels DLC, bottom LP
    '''
    
    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(8,5))
    
    # lick correlation left/right cam
    lick_corr(eid, lp=False, plo=True, ax=axs[0,0])
    lick_corr(eid, lp=True, plo=True, ax=axs[1,0])    

    # cca scatters for body parts seen by both side cams
    plot_compare(eid, 'paw_l', 'paw_r', ax=[axs[0,1], axs[1,1]])
    plot_compare(eid, 'paw_r', 'paw_l', ax=[axs[0,2], axs[1,2]])
    plot_compare(eid, 'nose_tip', 'nose_tip', ax=[axs[0,3], axs[1,3]])
    
    fig.suptitle(f"{eid},"
                 f" \n {'/'.join(str(one.eid2path(eid)).split('/')[-5:])}")
    fig.tight_layout()
    
    p = Path(f'/home/mic/DLC_LP/{eid}')
    p.mkdir(parents=True, exist_ok=True)
       
    s = (f'8pans_{eid}_'
         f"{str(one.eid2path(eid)).split('/')[-3]}.png")    
        
    fig.savefig(p / s)


def eks_z(eid, cam='left'):

    '''
    Ensemble Kalman Smoother (EKS) qc metric
    compute a z-score between the EKS prediction 
    and the ensemble for each keypoint/view:
    ||eks_prediction - ensemble_mean)|| / max(ensemble_std,c),
    where ||.|| is euclidean norm and c>0 is a rough estimate of the label noise,   
    so we don’t divide by any tiny numbers.
    '''
    
    p = Path(f'/home/mic/DLC_LP/{eid}/{eid}/ensembles_{cam}Camera/')

    D = {}

    for net in range(5):   

        df = pd.read_csv(p / f'_iblrig_{cam}Camera.raw.eye{net}.csv')

        D = D | {'_'.join([df[x][0],df[x][1],str(net)]): list(map(float, df[x][2:])) 
                 for x in df.keys() if df[x][1] in ['x','y']}

        df = pd.read_csv(p / f'_iblrig_{cam}Camera.raw.paws{net}.csv')

        D = D | {'_'.join([df[x][0],df[x][1],str(net)]): list(map(float, df[x][2:])) 
                 for x in df.keys() if df[x][1] in ['x','y']}        

    lst = [len(D[x]) for x in D]
    assert all(element == lst[0] for element in lst), 'length mismatch'

    parts = np.unique([x[:-4] for x in D])

    r = {}
    trace = {}

    for part in parts:
    
        m = []
        for net in range(5):
            m.append([D[f'{part}_x_{net}'], D[f'{part}_y_{net}']])
              
        m = np.array(m)      
        ens_mean = np.mean(m, axis=0)
        ens_var = np.var(m, axis=0)
        geo_ens_std = np.sqrt(ens_var[0,:] + ens_var[1,:])
        
        # compute for each net/obs the normed distance to ensemble mean
        u = []
        for net in range(5):
            u.append(np.sqrt(np.sum(np.square(m[net] - ens_mean),axis=0))
                     /geo_ens_std)
                     
        u = np.array(u)
        trace[part] = u 
                    
        obs_mean = np.mean(u,axis=1)        
        obs_std = np.std(u,axis=1)

        r[part] = [obs_mean, obs_std]

     
    fig, ax = plt.subplots(ncols=2, figsize=(8,6))
    
    ax[0].bar(r.keys(), [np.mean(r[x][0]) for x in r], 
              yerr = [np.std(r[x][0]) for x in r], color='k', width=0)
              
    ax[0].plot(r.keys(), [np.mean(r[x][0]) for x in r], linestyle='', 
              color='k', marker = 'o', markersize=10)                            
    
    for net in range(5):
        ax[0].plot(r.keys(), [r[x][0][net] for x in r], marker = 'o', 
        linestyle='', color='gray', markersize=5)

    ax[0].set_ylabel('normed dist from ens_mean \n meaned over frames')
    ax[0].set_xlabel('tracked points')
    ax[0].set_xticklabels(list(r.keys()), rotation = 45)
    ax[0].annotate('5 nets', xy=(0, 0), xytext=(0.05, 0.05),
             textcoords='axes fraction', fontsize=12, color='gray')
                 
    shift = max([max(np.mean(trace[part],axis=0)) - 
           min(np.mean(trace[part],axis=0)) for part in trace])
           
    cs = list(mcolors.BASE_COLORS.keys())
    
    k = 0
    for part in parts:
        tr = np.mean(trace[part],axis=0) - k*shift
        ax[1].plot(tr, label=part, c=cs[k])
                
        k += 1
     
    ax[1].legend().set_draggable(True)    
    ax[1].set_ylabel('dist from ens_mean [a.u.]')
    ax[1].set_xlabel('frames')    
    
    fig.suptitle(f"{cam}Camera \n {eid} "
                    f"\n {str(one.eid2path(eid)).split('/')[-3]}")            
    fig.tight_layout()

    p = Path(f'/home/mic/DLC_LP/{eid}')
    p.mkdir(parents=True, exist_ok=True)
       
    s = (f'ensemble_{cam}_{eid}_'
         f"{str(one.eid2path(eid)).split('/')[-3]}.png")    
        
    fig.savefig(p / s)


    





