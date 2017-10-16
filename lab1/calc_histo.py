#!/usr/bin/env python

import sys
import numpy as np
import cv2
import parse_file
from matplotlib import pyplot as plt

descriptor_file = open(sys.argv[1])
img_info = parse_file.get_img_info(descriptor_file)
#img_info = parse_file.get_img_info(descriptor_file)

#load the image
img = cv2.imread(img_info.img_path)
#DEBUG
cv2.imshow('face', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#creating a mask for the histogram
mask = img.copy()
mask[:] = (0, 0, 0)

for i in range(0, img_info.nb_faces):
	e_tmp = img_info.list_ellipse[i]
	cv2.ellipse(mask,(int(e_tmp.c_x),int(e_tmp.c_y)),(int(e_tmp.r_a),
		int(e_tmp.r_b)),int(e_tmp.theta),0,360,(255, 255, 255), -1)
#DEBUG
masked_img = cv2.bitwise_and(img, img, mask = mask[:,:,0])

color = {"b","g","r"}

hist_total = []
for i,col in enumerate(color): #enumerate returns always a couple, e.g.(0,'r')
	hist = cv2.calcHist([img],[i], mask[:,:,0], [256], [0,256])
	hist_total.append(hist)
	plt.plot(hist, color = col)
	plt.xlim([0,256])
plt.show()


#TODO: accumulate histograms for more than one image


