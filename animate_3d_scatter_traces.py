import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import cnames
from matplotlib import animation
import matplotlib.patches as mpatches


pts3d = np.load('/home/mic/3D-Animal-Pose-master/IBL_example/pts3d.npy', allow_pickle=True).flat[0] 
nframes, npts = pts3d['x_coords'].shape # number of frames and number of tracked parts

data = np.zeros((nframes, npts, 3)) # (n_iterations, n_particles, 3)
for i in range(npts):
    data[:,i] = np.array([pts3d['x_coords'][:,i], pts3d['y_coords'][:,i], pts3d['z_coords'][:,i]]).T

x_min = np.nanmin(pts3d['x_coords'])
x_max = np.nanmax(pts3d['x_coords'])
y_min = np.nanmin(pts3d['y_coords'])
y_max = np.nanmax(pts3d['y_coords'])
z_min = np.nanmin(pts3d['z_coords'])
z_max = np.nanmax(pts3d['z_coords'])

# Set up figure & 3D axis for animation
fig = plt.figure()
ax = Axes3D(fig)

ax.set_xlim3d([x_min,x_max])
ax.set_xlabel('X')

ax.set_ylim3d([y_min,y_max])
ax.set_ylabel('Y')

ax.set_zlim3d([z_min,z_max])
ax.set_zlabel('Z')
ax.set_title('IBL: paws and nose')

# set point-of-view: specified by (altitude degrees, azimuth degrees)
ax.view_init(30, 0)

# choose a different color for each trajectory
colors = plt.cm.jet(np.linspace(0, 1, npts))
# set up trajectory lines
lines =[ax.plot([], [], [], '-', c=c, lw=0.5)[0] for c in colors]
# set up points
pts = [ax.plot([], [], [], 'x', c=c, markersize = 4)[0] for c in colors]

ttl = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)

labels = ['paw1', 'paw2', 'nose']
patches = [mpatches.Patch(color=colors[u], label=labels[u]) for u in range(npts)]

# initialization function: plot the background of each frame
def init():

    for line, pt in zip(lines, pts): 
        line.set_data([], [])
        pt.set_data([], [])     
        
    plt.legend(handles=patches)       
    return lines, pts

def animate(i):
    for line, pt, j in zip(lines, pts, range(npts)):    
        line.set_data(data[:i,j,0], data[:i,j,1])
        line.set_3d_properties(data[:i,j,2])
        
        pt.set_data(data[i,j,0], data[i,j,1])
        pt.set_3d_properties(data[i,j,2])      
         
    ttl.set_text('frame %s/%s' %(i,nframes))
    
    ax.view_init(30, 0.3 * i)
    fig.canvas.draw()
    return lines, pts

# instantiate the animator.
anim = animation.FuncAnimation(fig, animate, init_func=init, blit=False, interval=16.66666666, frames=nframes)

Writer = animation.writers['ffmpeg']
writer = Writer(fps=60, metadata=dict(artist='Me'), bitrate=1800)        
        
anim.save('/home/mic/3D-Animal-Pose-master/IBL_example/im_trace.mp4', writer=writer)    

plt.show()
