import numpy as np
import cv2
import glob
import argparse
from pylab import *
# adapted from https://github.com/bvnayak/stereo_calibration
import os, fnmatch, pandas

'''
This assumes good checkeboard images seen from both cameras in folder RIGHT, LEFT
'''

class StereoCalibration(object):
    def __init__(self, filepath):
        # termination criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS +
                         cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.criteria_cal = (cv2.TERM_CRITERIA_EPS +
                             cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.objp = np.zeros((9*6, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        self.objpoints = []  # 3d point in real world space
        self.imgpoints_l = []  # 2d points in image plane.
        self.imgpoints_r = []  # 2d points in image plane.

        self.cal_path = filepath
        self.read_images(self.cal_path)

    def read_images(self, cal_path):
        images_right = glob.glob(cal_path + 'RIGHT/*.png')
        images_left = glob.glob(cal_path + 'LEFT/*.png')
        print(len(images_right))
        images_left.sort()
        images_right.sort()

        for i, fname in enumerate(images_right):
            img_l = cv2.imread(images_left[i])
            img_r = cv2.imread(images_right[i])

            gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret_l, corners_l = cv2.findChessboardCorners(gray_l, (9,6), None)
            ret_r, corners_r = cv2.findChessboardCorners(gray_r, (9,6), None)

            # If found, add object points, image points (after refining them)
            self.objpoints.append(self.objp)

            if ret_l is True:
                rt = cv2.cornerSubPix(gray_l, corners_l, (11, 11),
                                      (-1, -1), self.criteria)
                self.imgpoints_l.append(corners_l)

                # Draw and display the corners
                ret_l = cv2.drawChessboardCorners(img_l, (9,6),
                                                  corners_l, ret_l)
#                cv2.imshow(images_left[i], img_l)
#                cv2.waitKey(500)
 
                cv2.imwrite(cal_path+'/with_pattern/'+'left_'+images_left[i].split('/')[-1],img_l)
            else: 
                print('%s no pattern detected' %images_left[i])
                return

            if ret_r is True:
                rt = cv2.cornerSubPix(gray_r, corners_r, (11, 11),
                                      (-1, -1), self.criteria)
                self.imgpoints_r.append(corners_r)

                # Draw and display the corners
                ret_r = cv2.drawChessboardCorners(img_r, (9,6),
                                                  corners_r, ret_r)
#                cv2.imshow(images_right[i], img_r)
#                cv2.waitKey(500)
                cv2.imwrite(cal_path+'/with_pattern/'+'right_'+images_left[i].split('/')[-1],img_r)
           
            else: 
                print('%s no pattern detected' %images_right[i])
                return

            img_shape = gray_l.shape[::-1]

        print(len(self.imgpoints_l),len(self.imgpoints_r),len(self.objpoints))
        
        

        rt, self.M1, self.d1, self.r1, self.t1 = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_l, img_shape, None, None)
        rt, self.M2, self.d2, self.r2, self.t2 = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_r, img_shape, None, None)

        self.camera_model = self.stereo_calibrate(img_shape)

    def stereo_calibrate(self, dims):
        flags = 0
        flags |= cv2.CALIB_FIX_INTRINSIC
        # flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_ASPECT_RATIO
        flags |= cv2.CALIB_ZERO_TANGENT_DIST
        # flags |= cv2.CALIB_RATIONAL_MODEL
        # flags |= cv2.CALIB_SAME_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_K3
        # flags |= cv2.CALIB_FIX_K4
        # flags |= cv2.CALIB_FIX_K5

        stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER +
                                cv2.TERM_CRITERIA_EPS, 100, 1e-5)
        ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
            self.objpoints, self.imgpoints_l,
            self.imgpoints_r, self.M1, self.d1, self.M2,
            self.d2, dims,
            criteria=stereocalib_criteria, flags=flags)

        print('Intrinsic_mtx_1', M1)
        print('dist_1', d1)
        print('Intrinsic_mtx_2', M2)
        print('dist_2', d2)
        print('R', R)
        print('T', T)
        print('E', E)
        print('F', F)

        # for i in range(len(self.r1)):
        #     print("--- pose[", i+1, "] ---")
        #     self.ext1, _ = cv2.Rodrigues(self.r1[i])
        #     self.ext2, _ = cv2.Rodrigues(self.r2[i])
        #     print('Ext1', self.ext1)
        #     print('Ext2', self.ext2)

        print('')

        camera_model = dict([('M1', M1), ('M2', M2), ('dist1', d1),
                            ('dist2', d2), ('rvecs1', self.r1),
                            ('rvecs2', self.r2), ('R', R), ('T', T),
                            ('E', E), ('F', F)])

        #cv2.destroyAllWindows()
        #return only what we need below, the camera matrices
        return camera_model

#if __name__ == '__main__':
#    parser = argparse.ArgumentParser()
#    parser.add_argument('filepath', help='String Filepath')
#    args = parser.parse_args()
#    cal_data = StereoCalibration(args.filepath)

#ffmpeg -i '/home/mic/Videos/CameraCalibration/VideoCalibrationCheckerboard/cam_right.avi' -vcodec libx264 -crf 17 -vf scale=640:512 /home/mic/Videos/CameraCalibration/VideoCalibrationCheckerboard/right_640x512.mp4


def get2dpoints_DLC():

 ion()

 h5s=['/home/mic/Videos/CameraCalibration/calib-mic-2019-05-29/videos/leftDeepCut_resnet50_calibMay29shuffle1_150000.h5','/home/mic/Videos/CameraCalibration/calib-mic-2019-05-29/videos/rightDeepCut_resnet50_calibMay29shuffle1_150000.h5']

 L=[]

 for h5 in h5s:

  df=pandas.read_hdf(h5)
  XYs={}
  parts=['middle_black_top']
  XYleft={}
  for part in parts:
   x_values=df[(df.keys()[0][0], part, 'x')].values
   y_values=df[(df.keys()[0][0], part, 'y')].values
   likelyhoods=df[(df.keys()[0][0], part, 'likelihood')].values
   XYs[part]=[x_values,y_values,likelyhoods]

  L.append(XYs)
  
 return L


def select_test(L):

 '''
 select a short test video segment
 '''

 x_left=L[0]['middle_black_top'][0][2880:3060]
 x_right=L[1]['middle_black_top'][0][2880:3060] 
 y_left=L[0]['middle_black_top'][1][2880:3060]
 y_right=L[1]['middle_black_top'][1][2880:3060]

 left=array([x_left,y_left]).T
 right=array([x_right,y_right]).T
 
 return left,right
 
def get3dpoints(pts_l,pts_r,K_l,K_r):

 #Input:

 #pts_l - set of n 2d points in left image. nx2 numpy float array
 #pts_r - set of n 2d points in right image. nx2 numpy float array

 #K_l - Left Camera matrix. 3x3 numpy float array
 #K_r - Right Camera matrix. 3x3 numpy float array
 #K_l=cal_data.M1
 #K_r=cal_data.M2
 
 #Code:

 # Normalize for Esential Matrix calaculation
 pts_l_norm = cv2.undistortPoints(np.expand_dims(pts_l, axis=1), cameraMatrix=K_l, distCoeffs=None)
 pts_r_norm = cv2.undistortPoints(np.expand_dims(pts_r, axis=1), cameraMatrix=K_r, distCoeffs=None)

 E, mask = cv2.findEssentialMat(pts_l_norm, pts_r_norm, focal=1.0, pp=(0., 0.), method=cv2.RANSAC, prob=0.999, threshold=3.0)
 points, R, t, mask = cv2.recoverPose(E, pts_l_norm, pts_r_norm)

 M_r = np.hstack((R, t))
 M_l = np.hstack((np.eye(3, 3), np.zeros((3, 1))))

 P_l = np.dot(K_l,  M_l)
 P_r = np.dot(K_r,  M_r)
 point_4d_hom = cv2.triangulatePoints(P_l, P_r, np.expand_dims(pts_l, axis=1), np.expand_dims(pts_r, axis=1))
 point_4d = point_4d_hom / np.tile(point_4d_hom[-1, :], (4, 1))
 point_3d = point_4d[:3, :].T
 #Output:
 return point_3d
 #point_3d - nx3 numpy array


 #cv2.estimateAffine3D #from camera to real world coordinates, needs measured points  








