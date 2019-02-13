
##Image acquisition 

The behavioral rig camera will be recorded at 30 Hz on the Windows computer that runs the behavioral task. The recording software is Bonsai which allows for a video stream window to be displayed to the user during behavioral training. For the recording rig, Bonsai will also run on Windows, with one side view camera and the mouse trunk camera at 60 Hz, and the other side camera at 150 Hz. 
 
##Feature-tracking in side view videos using DeepLabCut	 	 
	
We next present tracking body parts using machine learning and how performance depends on video compression. DeepLabCut (DLC, https://github.com/AlexEMG/DeepLabCut) was used to predict the location of 27 features seen from at least one side. The points include 4 for tongue tracking (2 static points on the lick port, 2 points at the edge of the tongue), 4 for pupil tracking for each pupil and the tips of 4 fingers per paw. In addition, nose, chin and ear tip were tracked (Figure 5). The 27 features are as follows, with “r” or “l” indicating the right or left type of the feature, as seen from the right/left camera:

'pupil_top_r', 'pupil_right_r', 'pupil_bottom_r', 'pupil_left_r', 'whisker_pad_r', 'nose_tip', 'tube_top', 'tongue_end_r', 'tube_bottom', 'tongue_end_l', 'chin_r', 'pinky_r', 'ring_finger_r', 'middle_finger_r', 'pointer_finger_r', 'ear_tip_r', 'pupil_top_l', 'pupil_right_l', 'pupil_bottom_l', 'pupil_left_l', 'whisker_pad_l', 'chin_l', 'pinky_l', 'ring_finger_l', 'middle_finger_l', 'pointer_finger_l', 'ear_tip_l'


In order to create a training set for the DLC network, the positions of these features were manually annotated in 400 images and some of them were further used as “ground truth” for testing of network prediction accuracy. Note that some features are seen from both side views. 300 of the images were uniformly sampled from one of the two 10 min side videos (both 10 min long, at 60 Hz, compressed at 29 crf, 1280 x 1024 px) and 100 were manually selected in order to provide sufficient samples that display the mouse’ tongue. I.e. one DLC network was trained jointly on images from two side views. The labeling was done using the DLC GUI. 

Performing cross-validation, the manually-labeled images were split randomly into 280 training and 120 testing images, repeated 4 times for different random splits, in order to evaluate the prediction precision of the DLC network, as suggested by Mathis et al.5. The error was computed as the Euclidean distance in pixels from “ground truth” to the position predicted by DLC. Figure 5 shows the test error across 120 test images for each tracked feature. Note that these results can easily be improved by the inclusion of more hand-selected training images (we plan to generate one from videos of several labs and animals, for generality). The high accuracy of the position prediction for the 4 pupil points (test errors below 2 px = 0.1 mm) allows accurately estimating the pupil diameter and position, thereby replacing dedicated eye-tracking software and obviating the need for an extra eye-camera. Tracking 4 fingers per paw allows further the reliable distinction and tracking of each paw. 
	 
Accurate lick-detection with DLC can be achieved with two points on the tongue tip and two points on the water-port (when the two points of the tube are within a small distance to one of the tongue tip points). We plan to compare these DLC-based lick detections to the lick detections read from a dedicated lick-detector. 




