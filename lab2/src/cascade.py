#!/usr/bin/env python

import sys
import numpy as np
import cv2
import time

# command line arguments
if(len(sys.argv)<2):
    print("Usage: python3 cascade.py image_file_name.ext")
    exit(1)

# load necessary files
face_cascade = cv2.CascadeClassifier("../haarcascades/haarcascade_frontalface_default.xml")
img_name = sys.argv[1]
img = cv2.imread(img_name)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# the actual face detection
start_time = time.time()
# a scale-factor of 1.10 means that the image will be downscaled by 10% each time
# => there will be 10 images to go trough
faces = face_cascade.detectMultiScale(gray,1.1,3)
#[faces,rejectLevels,levelWeights] = face_cascade.detectMultiScale3(gray,1.1,3,outputRejectLevels=True)
print("len(faces): " + str(len(faces)))
#print(rejectLevels)
#print(levelWeights)
end_time = time.time()
print("time=", end_time - start_time)

# plot the result
for(x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
cv2.namedWindow("Danemark", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Danemark", 800, 600)
cv2.imshow("Danemark", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
