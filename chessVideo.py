##IMPORTS

import cv2 as cv
import numpy as np
from time import sleep


##CONSTANTS
CAMERA = 1

##FUNCTIONS

def morph(img, iterations):
    kernelE = np.ones((5, 5), np.uint8)
    kernelD = np.ones((5, 5), np.uint8)
    for i in range(0, iterations):
        eImg = cv.erode(img, kernelE, iterations=1)
        eImg = cv.dilate(eImg, kernelD, iterations=1)

    return eImg


def cornerDetect(img, maxNum, quality, minDist):
    #corners = np.float32(thresh)
    corners = cv.goodFeaturesToTrack(img, maxNum, quality, minDist)
    #print(corners)
    #print(len(corners))
    return np.int0(corners)


def cornerOverlay(img, corners, withNumber=0):
    i = 0
    for corner in corners:
        currentColor = (153,51,255)  #200,15,200
        x, y = corner.ravel()
        cv.circle(img, (x, y), 2, currentColor, 2, -1)
        if (withNumber == 1):
            cv.putText(img, str(i), (int(x) + 5, int(y) - 5), cv.FONT_HERSHEY_SCRIPT_COMPLEX, .7, currentColor)
            #scv.putText(img, str(int(x))+","+str(int(y)), (int(x), int(y)+15), cv.FONT_HERSHEY_SCRIPT_SIMPLEX, .4, currentColor)
        i += 1
    return img

#def shapeOverlay(img, fourCourners):


def getShots(shots, feed, greyScale=0):
    images = []
    for i in range(0, shots):
        a, frame = feed.read()
        if a:
            if greyScale == 1:
                frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            images.append(frame)
            # cv.imshow('1', frame)
    return images


def reduceNoise(images, greyScale):
    if (len(images) > 1):
        if greyScale == 1:
            return cv.fastNlMeansDenoisingMulti(images, 0, len(images))
        else:
            return cv.fastNlMeansDenoisingColoredMulti(images, 1, len(images))
    elif (len(images) == 1):
        if greyScale == 1:
            return cv.fastNlMeansDenoising(images[0], None, 10)


def thresholdAll(images, value):
    ret = []
    for image in images:
        ret.append(cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, value, 1))
    return ret


##MAIN

feed = cv.VideoCapture(CAMERA)
sleep(1)


while (True):
    # _, frame = feed.read()
    # frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    '''
    images = getShots(5, feed, 1)
    frame = cv.fastNlMeansDenoisingMulti(images, 1, 3, None, 10, 7, 21)
    '''
    frame = cv.imread('chess.png',0)

    inp = cv.imread('largeDownTilt.png', 0)
    frame = cv.resize(inp, (640, 640))

    cv.imshow('og', frame)

    # images = thresholdAll(images, 115)
    # thresh = cv.GaussianBlur(frame, (5, 5),0)
    _, thresh = cv.threshold(frame, 120, 255, cv.THRESH_BINARY)
    # thresh = cv.adaptiveThreshold(frame , 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 111, 1)
    #_, thresh = cv.threshold(frame, 50, 200, cv.THRESH_BINARY+cv.THRESH_OTSU)

    thresh = morph(thresh, 5)
    cv.imshow("thres", thresh)
    tempCorners = cornerDetect(thresh, 100, 0.025, 40)
    corners = []

    print(len(tempCorners))

    if len(tempCorners)>=80:

        tempCorners = sorted(tempCorners, key=lambda element: (element[0][1]))#, element[0][0]))

        for i in range(0, 80, 9):
            tempLine = sorted(tempCorners[0+i:9+i], key=lambda element: (element[0][0]))
            corners.extend(tempLine)


    thresh = cv.cvtColor(thresh, cv.COLOR_GRAY2RGB)
    frame = cornerOverlay(thresh, corners, 1)

    cv.imshow("forstr", thresh)


    '''
    if len(tempCorners) >= 80:

        for i in range(8):
            for j in range(8):
                cur = (i*9)+j
                points = [corners[cur], corners[cur+1], corners[cur+9], corners[cur+10]]
                print(points)
                cv.fillPoly(thresh, np.array(points), (40,50,60))

        cv.imshow("forstr", thresh)

    '''


    if (cv.waitKey(0) & 0xFF == ord('q')):
        break
    # sleep(1)

feed.release()
cv.destroyAllWindows();
