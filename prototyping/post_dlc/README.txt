The post_dlc task takes dlc traces as input and computes useful quantities from it, 
some of which combine the info oth both side cameras, such as lick times while others are camera specific.

For the dataset type, I recommend having just one file for all post_dlc, 
called 'alf/_ibl_post_dlc.pqt' which includes so far the following keys:

pupil_dia_raw_left  (shape is 1 x #frames of left cam)
pupil_dia_smooth_left  (shape is 1 x #frames of left cam)
pupil_dia_raw_right  (shape is 1 x #frames of right cam)
pupil_dia_smooth_right  (shape is 1 x #frames of right cam)
lick_times  (shape is 1 x #licks)

The task should further compute all dlc qc metrics (which all only take dlc traces as input) and 
in the future we'll include 3d paw positions and behavioral segmentation via ARHMM, 
again both using only dlc traces as input).
