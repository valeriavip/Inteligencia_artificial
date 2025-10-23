import numpy as np
import cv2 as cv
import math
cap = cv.VideoCapture()
i = 0
while True:
    ret, frame = cap.read()
    frame2 = cv.resize(frame, (28, 28), interpolation=cv.INTER_AREA)
    if (i % 5 == 0):
        cv.imwrite(''+str(i)+'.jpg', frame2)
    cv.imshow('salida', frame)
    i += 1
    k = cv.waitKey(1)
    if k == 27:
        break
cap.release()
cv.destroyAllWindows()

