import cv2 
import numpy as np

video_path = '/home/mic/body_cam_background/body_2min.mp4'

cap = cv2.VideoCapture(video_path)

_, first_frame = cap.read()
first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
first_gray = cv2.GaussianBlur(first_gray, (5, 5), 0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
size = (int(cap.get(3)), int(cap.get(4)))
# you need thi ,0 option here for greyscale; for RGB not. 
out = cv2.VideoWriter('output.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 30.0, size, 0)


total_frames = cap.get(7) 

while cap.isOpened():

    ret, frame = cap.read()
    if ret==True:

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
        difference = cv2.absdiff(first_gray, gray_frame)
        _, difference = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)        
        
        out.write(difference)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
