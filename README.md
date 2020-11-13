
## Video acquisition in IBL 

Mice are filmed in training rigs and recording rigs. In training rigs there is only one side camera recording at full resolution (1280x1024) and 30 Hz. In the recording rigs, there are three cameras, one called 'left' at full resolution 1280x1024 and 60 Hz filming the mouse from one side, one called 'right' at half resolution (640x512) and 150 Hz filming the mouse symmetrically from the other side, and one called 'body' filming the trunk of the mouse from above.    
 
## Feature-tracking in side view videos using DeepLabCut	 	 

DeepLabCut (DLC) is used for markerless tracking of animal parts in these videos. For each side video we track the following points:	

`'pupil_top_r', 'pupil_right_r', 'pupil_bottom_r', 'pupil_left_r', 'nose_tip', 'tube_top', 'tube_bottom', 'tongue_end_r', 'tongue_end_l', 'paw_r', 'paw_l'`

![side_view_points](https://github.com/int-brain-lab/iblvideo/blob/master/DLC_IBL.png | width=100)


