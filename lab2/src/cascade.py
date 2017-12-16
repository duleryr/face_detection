#!/usr/bin/env python

import sys
import numpy as np
import cv2
import time
import re
from matplotlib import pyplot as plt

# command line arguments
if(len(sys.argv)<2):
    print("Usage: python3 cascade.py image_file_name.ext scale_factor min_neighbors")
    print("The last two parameters are optional")
    exit(1)

if(len(sys.argv) < 4):
    min_neighbors = 3
else:
    min_neighbors = int(sys.argv[3])

if(len(sys.argv) < 3):
    scale_factor = 1.4
else:
    scale_factor = float(sys.argv[2])

# load necessary files
face_cascade = cv2.CascadeClassifier("../haarcascades/haarcascade_frontalface_default.xml")
img_name = sys.argv[1]
img = cv2.imread(img_name)
final_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# the actual face detection
start_time = time.time()
# a scale-factor of 1.10 means that the image will be downscaled by 10% each time
# => there will be 10 images to go trough

#[faces,rejectLevels,levelWeights] = face_cascade.detectMultiScale3(gray,1.1,3,outputRejectLevels=True)
#print("len(faces): " + str(len(faces)))
#print(rejectLevels)
#print(levelWeights)
print(scale_factor, min_neighbors)
faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)
end_time = time.time()
print("time=", end_time - start_time)

# plot the result
string = str(sys.argv[1])
res = string.split("/")
title = res[len(res) - 1]
final_title = title + " - scaleFactor : " + str(scale_factor) + ", minNeighbours : " + str(min_neighbors)
filename = title.split(".")[0] + "_" + str(scale_factor).replace('.', ',') + "_" + str(min_neighbors) + ".png"
print(filename)
for(x,y,w,h) in faces:
    cv2.rectangle(final_img,(x,y),(x+w,y+h),(255,0,0),2)
#cv2.namedWindow(str(title), cv2.WINDOW_NORMAL)
#cv2.resizeWindow(str(title), 800, 600)
#cv2.imshow(str(title), img)
#cv2.imwrite(str(legend)+".jpg", img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

plt.gcf().clear()
plt.imshow(final_img)
plt.title(final_title)
plt.savefig("figures/"+filename)
plt.show()
