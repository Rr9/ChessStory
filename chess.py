import cv2 as cv
import numpy as np


chess = cv.imread('chess.png', 0)
#cv.imshow('chess',chess)

thresh = cv.adaptiveThreshold(chess, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 111, 1)

corners = np.float32(thresh)
corners = cv.goodFeaturesToTrack(corners, 100, 0.01, 10)
corners = np.int0(corners)

print(corners)

i = 0
for corner in corners:
    i+=1
    x,y = corner.ravel()
    cv.circle(chess, (x,y), 2, (200,15,200), 2, -1)
    cv.putText(chess, str(i), (x+5, y-5), cv.FONT_HERSHEY_SCRIPT_SIMPLEX, .5, (210,15,200))

cv.imshow('cornerChess', chess)
