import cv2
import numpy as np
import glob

"""
Corner Finding
"""
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 23, 0.001)

# Prepare object points, like (0,0,0), (1,0,0), ....,(6,5,0)
objp = np.zeros((5*5,3), np.float32)
objp[:,:2] = np.mgrid[0:5,0:5].T.reshape(-1,2)

# Arrays to store object points and image points from all images
objpoints = []
imgpoints = []
chess = (9,6)
output = []

counting = 0

# Import Images
images = cv2.VideoCapture(1)


while True:
    ret, fname = images.read()
    gray = cv2.cvtColor(fname, cv2.COLOR_BGR2GRAY)

    dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    fname[dst > 0.01 * dst.max()] = [0, 0, 255]
    dist = []
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, chess)

    # if found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)

        #Draw and display corners
        boob = cv2.drawChessboardCorners(fname, chess, corners, ret)
        counting += 1

        #cv2.solvePnPGeneric(corners, objp, fname, distCoeffs=0.23)

        corner_points = cv2.goodFeaturesToTrack(gray, 3, 0.01, 1)
        corner_points = corner_points.astype(np.int)[:, 0, :]
        m = np.matrix(corner_points)
        #dot_product = m[0] * m[1]
        print(m)

        print(str(counting) + ' Viable Image(s)')
    cv2.imshow('Harris Corner Detection', fname)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()


# Calibrate Camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)