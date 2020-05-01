import cv2
import numpy as np
from matplotlib import pyplot as plt

"""
Corner Finding
"""
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 23, 0.001)

# Prepare object points, like (0,0,0), (1,0,0), ....,(6,5,0)
objp = np.zeros((5*5,3), np.float32)
objp[:,:2] = np.mgrid[0:5,0:5].T.reshape(-1,2)

right_objp = np.zeros((5*5,3), np.float32)
right_objp[:,:2] = np.mgrid[0:5,0:5].T.reshape(-1,2)

# Arrays to store object points and image points from all images
left_objpoints = []
left_imgpoints = []

right_objpoints = []
right_imgpoints = []
chessboard = (9,6)

counting = 0
#right_counting = 0
# Import Images
left_images = cv2.VideoCapture(1)
right_images = cv2.VideoCapture(0)

while True:
    ret, left_fname = left_images.read()
    left_gray = cv2.cvtColor(left_fname, cv2.COLOR_BGR2GRAY)

    ret, right_fname = right_images.read()
    right_gray = cv2.cvtColor(right_fname, cv2.COLOR_BGR2GRAY)



    # Find the chess board corners
    ret, left_corners = cv2.findChessboardCorners(left_gray, chessboard)
    ret, right_corners = cv2.findChessboardCorners(right_gray, chessboard)


    # if found, add object points, image points (after refining them)
    if ret == True:
        left_objpoints.append(objp)
        right_objpoints.append(right_objp)

        cv2.cornerSubPix(right_gray, right_corners, (11, 11), (-1, -1), criteria)
        #cv2.cornerSubPix(left_gray, left_corners, (11,11), (-1,-1), criteria)

        right_imgpoints.append(right_corners)
        left_imgpoints.append(left_corners)


        #Draw and display corners point and lines on image
        cv2.drawChessboardCorners(right_fname, chessboard, right_corners, ret)
        cv2.drawChessboardCorners(left_fname, chessboard, left_corners, ret)

        right_corner_points = cv2.goodFeaturesToTrack(right_gray, 3, 0.01, 1)
        right_corner_point = right_corner_points.astype(np.int)[:, 0, :]
        left_corner_points = cv2.goodFeaturesToTrack(left_gray, 3, 0.01, 1)
        left_corner_point = left_corner_points.astype(np.int)[:, 0, :]


        counting += 1
 #       right_counting += 1
        print(str(counting) + ' Viable Image(s)')
        print(left_corner_point)
        print(right_corner_point)

    cv2.imshow('left', left_fname)
    cv2.imshow('right', right_fname)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()


# Calibrate Camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(left_objpoints, left_imgpoints, left_gray.shape[::-1],None,None)