"""
Plot timestamps and sync pulses
"""

from CameraFunctionsS import CaptureSettings
from os.path import join
from os import listdir
import matplotlib.pyplot as plt
import numpy as np

dataPath,_,_ = CaptureSettings()
filesTS = [x for x in listdir(dataPath) if 'TS' in x]

f, (ax1, ax2) = plt.subplots(2, 1) 
for i in filesTS:
    TS = np.load(join(dataPath, i))
    #TTL = np.load(join(dataPath, 'SyncPulses_%s.npy'%i[11:19]))
    ax1.plot(TS, label=i[11:19])
    #ax2.plot(TS/1000, TTL, label=i[11:19])    

ax1.set_xlabel('Video frames')
ax1.set_ylabel('Timestamps (ms)')
ax1.set_title('Timestamps')
ax1.legend()

#ax2.set_xlabel('Time (s)')
#ax2.set_ylabel('TTL input')
#ax2.set_title('Sync pulses from Bpod in time')
#ax2.legend()
    
plt.tight_layout()
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()

fig = plt.gcf()
fig.set_size_inches((10, 8), forward=False)
fig.savefig(join(dataPath, 'VideoData.png'), dpi=300)

plt.show()
