#!/usr/bin/env python

import sys
import cv2
import numpy as np
import parse_file

descriptor_file = open(sys.argv[1])
img_info = parse_file.get_img_info(descriptor_file)
img_info = parse_file.get_img_info(descriptor_file)

#load the image
img = cv2.imread(img_info.img_path)

for i in range(0, img_info.nb_faces):
    e_tmp = img_info.list_ellipse[i]
    cv2.ellipse(img,(int(e_tmp.c_x),int(e_tmp.c_y)),(int(e_tmp.r_a),
    	int(e_tmp.r_b)),int(e_tmp.theta),0,360,(0, 255, 0), 3)

cv2.imshow("face", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
