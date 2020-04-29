import cv2
import numpy as np

img = cv2.VideoCapture(0)

while True:
        ret, frame = img.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)
        corners = np.int0(corners)


        for i in corners:
            x, y = i.ravel()
            cv2.circle(frame, (x,y), 3, 255, -1)
        cv2.imshow('DST', frame)
        if cv2.waitKey(1) & 0xFF == ord ('q'):
            break

img.release()
cv2.destroyAllWindows()