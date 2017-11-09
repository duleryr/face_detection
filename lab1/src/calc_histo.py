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
    #cv2.imshow('face', img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #cv2.imshow('face', mask)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #return cv2.bitwise_and(img, img, mask = mask[:,:,0])
    return mask

# Return the hist as [HistB, HistG, HistR]
def calc_hist(img, mask):
    n_quantification = 8
    #color = {"b","g","r"}
    #hist_rgb = []
    hist_all_colors = np.array([[[0]*256]*256]*256)
    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
    	    (r,g,b) = img[i,j]
    	    if(mask[i,j][0]==255):
    	        hist_all_colors[r][g][b] += 1
    #DEBUG
    #plt.plot(hist_all_colors.flatten())
    #plt.xlim([0,hist_all_colors.size-1])
    #plt.show()
    #for i,col in enumerate(color): #enumerate returns always a couple, e.g.(0,'r')
    #	hist = cv2.calcHist([img],[i], mask[:,:,0], [256/n_quantification], [0,31])
    #	hist_rgb.append(hist)
    #return hist_rgb
    return hist_all_colors

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

if __name__ == '__main__':
    test()
