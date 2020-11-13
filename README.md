
## Video acquisition in IBL 

Mice are filmed in training rigs and recording rigs. In training rigs there is only one side camera recording at full resolution (1280x1024) and 30 Hz. In the recording rigs, there are three cameras, one called 'left' at full resolution 1280x1024 and 60 Hz filming the mouse from one side, one called 'right' at half resolution (640x512) and 150 Hz filming the mouse symmetrically from the other side, and one called 'body' filming the trunk of the mouse from above.    
 
## Feature-tracking in side view videos using DeepLabCut	 	 

DeepLabCut (DLC) was used fro markerless tracking of animal parts in these videos. For each side video we track the following points:	

`'pupil_top_r', 'pupil_right_r', 'pupil_bottom_r', 'pupil_left_r', 'nose_tip', 'tube_top', 'tube_bottom', 'tongue_end_r', 'tongue_end_l', 'paw_r', 'paw_l'`



![side_view_points](https://user-images.githubusercontent.com/17218515/52708624-ea099680-2f8a-11e9-884b-6c82b1a54ce7.png)


