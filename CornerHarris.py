import cv2
import numpy as np

images = cv2.VideoCapture(0)

while True:
        ret, fname = images.read()
        gray = cv2.cvtColor(fname, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)

        dst = cv2.cornerHarris(gray, 2, 3, 0.04)

        fname[dst>0.01 * dst.max()] = [0,0,255]

        cv2.imshow('Harris Corner Detection', fname)
        if cv2.waitKey(1) & 0xFF == ord ('q'):
            break

img.release()
cv2.destroyAllWindows()

