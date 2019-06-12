
## Image acquisition 

The behavioral rig camera will be recorded at 30 Hz on the Windows computer that runs the behavioral task. The recording software is Bonsai which allows for a video stream window to be displayed to the user during behavioral training. For the recording rig, Bonsai will also run on Windows, with one side view camera and the mouse trunk camera at 60 Hz, and the other side camera at 150 Hz. 
 
## Feature-tracking in side view videos using DeepLabCut	 	 
	
We next present tracking body parts using machine learning and how performance depends on video compression. DeepLabCut (DLC, https://github.com/AlexEMG/DeepLabCut) was used to predict the location of 27 features seen from at least one side. The points include 4 for tongue tracking (2 static points on the lick port, 2 points at the edge of the tongue), 4 for pupil tracking for each pupil and the tips of 4 fingers per paw. In addition, nose, chin and ear tip were tracked (Figure 5). The features are as follows, with “r” or “l” indicating the right or left type of the feature, as seen from the right/left camera:

`'pupil_top_r', 'pupil_right_r', 'pupil_bottom_r', 'pupil_left_r', 'nose_tip', 'tube_top', 'tongue_end_r', 'tube_bottom', 'tongue_end_l', 'pinky_r', 'ring_finger_r', 'middle_finger_r', 'pointer_finger_r', 'pupil_top_l', 'pupil_right_l', 'pupil_bottom_l', 'pupil_left_l', 'pinky_l', 'ring_finger_l', 'middle_finger_l', 'pointer_finger_l'`


In order to create a training set for the DLC network, the positions of these features were manually annotated in 400 images of one video, and additional 100 sampled from 6 other videos (other mice and rigs). Some of them were further used as “ground truth” for testing of network prediction accuracy. Note that some features are seen from both side views. 300 of the images were uniformly sampled from one of the two 10 min side videos (both 10 min long, at 60 Hz, compressed at 29 crf, 1280 x 1024 px) and 100 were manually selected in order to provide sufficient samples that display the mouse’ tongue. I.e. one DLC network was trained jointly on images from two side views. The labeling was done using the DLC GUI. This network is only used for automatic cropping, i.e. finding regions of interest (eye, nose, tongue, paws) which are cut out automatically from the videos and then specialised DLC network are applied to these crops (see `DLC_crop.py`). Smaller videos accelerate processing time considerably and the specialised networks can be independently tuned. It is further much faster to create hand-annotated training images if no zooming is required - as would be the case when having all points in one frame.
	 
Accurate lick-detection with DLC can be achieved with two points on the tongue tip and two points on the water-port (when the two points of the tube are within a small distance to one of the tongue tip points).

![side_view_points](https://user-images.githubusercontent.com/17218515/52708624-ea099680-2f8a-11e9-884b-6c82b1a54ce7.png)


