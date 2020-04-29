import cv2
import numpy as np
import glob

"""
Corner Finding
"""
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points, like (0,0,0), (1,0,0), ....,(6,5,0)
objp = np.zeros((5*5,3), np.float32)
objp[:,:2] = np.mgrid[0:5,0:5].T.reshape(-1,2)

# Arrays to store object points and image points from all images
objpoints = []
imgpoints = []

counting = 0

# Import Images
images = cv2.VideoCapture(0)

while True:
    ret, fname = images.read()
    gray = cv2.cvtColor(fname, cv2.COLOR_BGR2GRAY)



    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (5,5))

    # if found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)

        #Draw and display corners
        cv2.drawChessboardCorners(fname, (5,5), corners, ret)
        counting += 1

        print(str(counting) + ' Viable Image(s)')

    cv2.imshow('Harris Corner Detection', fname)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()


# Calibrate Camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)