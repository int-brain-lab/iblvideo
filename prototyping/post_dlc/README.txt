The post_dlc task takes dlc traces as input and computes useful quantities from it, 
camera specific. (Paw 3d will get its own file one day, and also behavioral segmentation via ARHMM).

There should be one pqt file per camera, e.g. _ibl_leftCamera.features.pqt 
and it will contain columns named in Pascal case, 
the same way you would name an ALF attribute, e.g. pupilDiameter_raw and 
lick_times.

pupilDiameter_raw  (shape is 1 x #frames of left cam)
pupilDiameter_smooth  (shape is 1 x #frames of left cam)
lick_times (shape is 1 x #licks)

The task should further compute all dlc qc metrics (which all only take dlc traces as input), including 
the new SNR score for the pupil diameter, as is computed in pupil_diameter.py
