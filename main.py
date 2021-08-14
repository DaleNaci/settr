import time
import datetime
import argparse

import numpy as np
import cv2
import imutils
from imutils.video import VideoStream


vid = cv2.VideoCapture(0)

while(True):
    ret, frame = vid.read()

    cv2.circle(frame, (50, 50), 30, (0, 0, 255), -1)
    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break



vid.release()
cv2.destroyAllWindows()
