from __future__ import print_function
import cv2
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import sys

imagepath = sys.argv[1]
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
image = cv2.imread(imagepath)
#image = imutils.resize(image, width=min(400, image.shape[1]))
(rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),padding=(8, 8), scale=1.05)
print(rects, weights)
rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
nweights = []
for x in pick:
    x[]
for x in pick:
    if x in rects:
        nweights.append(weights[rects.index(x)])
index = np.array(nweights).argmax()
(x, y, w, h) = pick[index]
image = image[y:y+h,]
cv2.imwrite("newimg5.jpg", image)


