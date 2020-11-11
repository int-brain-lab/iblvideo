import cv2 
import numpy as np


def get_average_hist(video_path):

    '''
    get the average histogram of a video
    '''

    cap = cv2.VideoCapture(video_path)
    D = []
    while cap.isOpened():     
 
        ret1, frame1 = cap.read()        

        if ret1==True:
          
            D.append(frame1)
            #out.write(difference)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()    
    
    return cv2.calcHist([D],[0],None,[256],[0,256])



def histogram_diff(video_path):

    '''
    compute euclidean distance of histogram of each frame to a reference hist
    '''
    
    histM = np.load('/home/mic/DLC_ephys_IBL/QC_videos/average_hist.npy')

    cap = cv2.VideoCapture(video_path)
    total_frames = cap.get(7) 
    D = []

    k = 0
    while cap.isOpened():     

        ret1, frame1 = cap.read()        
        
        if k%1000 == 0:
            print(k)

        if ret1==True:

            hist1 = cv2.calcHist([frame1],[0],None,[256],[0,256])
            a = cv2.compareHist(hist1,histM,cv2.HISTCMP_BHATTACHARYYA)
            D.append(a)
            #out.write(difference)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

        k+=1

    cap.release()
    cv2.destroyAllWindows()    

    return D
