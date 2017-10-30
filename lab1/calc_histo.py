#!/usr/bin/env python

import sys
import numpy as np
import cv2
import parse_file
from matplotlib import pyplot as plt


# Return the masked image
# Warning : Redundancy between img and img_info
def calc_mask(img, img_info):
    mask = img.copy()
    #creating a mask for the histogram
    mask[:] = (0, 0, 0)
    
    for i in range(0, img_info.nb_faces):
        e_tmp = img_info.list_ellipse[i]
        cv2.ellipse(mask,(int(e_tmp.c_x),int(e_tmp.c_y)),(int(e_tmp.r_a),
    		int(e_tmp.r_b)),int(e_tmp.theta),0,360,(255, 255, 255), -1)
    return cv2.bitwise_and(img, img, mask = mask[:,:,0])


# Return the hist as [HistB, HistG, HistR]
def calc_hist(img, mask):
    color = {"b","g","r"}
    hist_rgb = []
    for i,col in enumerate(color): #enumerate returns always a couple, e.g.(0,'r')
    	hist = cv2.calcHist([img],[i], mask[:,:,0], [256], [0,256])
    	hist_rgb.append(hist)
    return hist_rgb

# Return the histogram without mask
def calc_normal_hist(img):
	mask = img.copy()
	mask[:] = (255,255,255)
	return calc_hist(img, mask)

def test():
    descriptor_file = open(sys.argv[1])
    img_info = parse_file.get_img_info(descriptor_file)

    img_global = cv2.imread(img_info.img_path)
    
    masked_img = calc_mask(img_global, img_info)
    hist = calc_hist(img_global, masked_img)
    
    color = {"b","g","r"}
    for i,col in enumerate(color): #enumerate returns always a couple, e.g.(0,'r')
        plt.plot(hist[i], color = col)
        plt.xlim([0,256])
    
    plt.show()
    cv2.imshow('face', masked_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
