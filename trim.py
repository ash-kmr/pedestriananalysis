import cv2
import os
import sys

imagedir = sys.argv[1]
savedir = sys.argv[2]
for filename in os.listdir(imagedir):
    path = os.path.join(imagedir, filename)
    image = cv2.imread(path)
    image = image[int(image.shape[0]/2):, :, :]
    savepath = os.path.join(savedir, filename)
    cv2.imwrite(savepath, image)