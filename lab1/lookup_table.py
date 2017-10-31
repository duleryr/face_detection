#! /usr/bin/env python

import sys
import calc_histo
import parse_file
import numpy as np
import cv2
from matplotlib import pyplot as plt

# Return the likelihood table(r,g,b) for nb_images images
def construct_lookup_table(fd, nb_images):
	# Initialize the sum of histograms with the first image
	img_info = parse_file.get_img_info(fd)
	img_global = cv2.imread(img_info.img_path)
	masked_img = calc_histo.calc_mask(img_global, img_info)
	hist_t = calc_histo.calc_hist(img_global, masked_img)
	hist_all = calc_histo.calc_normal_hist(img_global)
	
	# Initialization
	hist_t_sum = hist_t
	hist_all_sum = hist_all
	
	for i in range(1,nb_images):
		img_info = parse_file.get_img_info(fd)
		img_global = cv2.imread(img_info.img_path)
		masked_img = calc_histo.calc_mask(img_global, img_info)
		hist_t = calc_histo.calc_hist(img_global, masked_img)
		hist_all = calc_histo.calc_normal_hist(img_global)
		# Sum the histograms
		hist_t_sum[0] += hist_t[0] # for loop doesn't seem to work here
		hist_t_sum[1] += hist_t[1]
		hist_t_sum[2] += hist_t[2] 
		hist_all_sum[0] += hist_all[0]
		hist_all_sum[1] += hist_all[1]
		hist_all_sum[2] += hist_all[2]

	lookup_table = hist_t_sum
	lookup_table[0] /= hist_all_sum[0]
	lookup_table[1] /= hist_all_sum[1]
	lookup_table[2] /= hist_all_sum[2]
	
# DEBUG
#	color = {"b","g","r"}
#	for i,col in enumerate(color): #enumerate returns always a couple, e.g.(0,'r')
#	    plt.plot(hist_t_sum[i], color = col)
#	    plt.xlim([0,256])
#	plt.show()
#
#	for i,col in enumerate(color): #enumerate returns always a couple, e.g.(0,'r')
#	    plt.plot(hist_all_sum[i], color = col)
#	    plt.xlim([0,256])
#	plt.show()

#	for i,col in enumerate(color): #enumerate returns always a couple, e.g.(0,'r')
#	    plt.plot(lookup_table[i], color = col)
#	    plt.xlim([0,256])
#	plt.show()

#	cv2.imshow('face', masked_img)
#	cv2.waitKey(0)
#	cv2.destroyAllWindows()
	return lookup_table

def pixel_probability(lookup_table, pixel):
	return (lookup_table[0][pixel[0]],lookup_table[1][pixel[1]],lookup_table[2][pixel[2]])

# Fill the next image with the likelihood-values (Black and white style)
def test_with_image(fd, lookup_table):
	img_info = parse_file.get_img_info(fd)
	img = cv2.imread(img_info.img_path)
	likelihood_img = img.copy()
	for i in range(0, likelihood_img.shape[0]):
		for j in range(0, likelihood_img.shape[1]):
			pixel_prob = pixel_probability(lookup_table,likelihood_img[i,j])
			grey_value = (255)*pixel_prob[0]+(255)*pixel_prob[1]+(255)*pixel_prob[2]
			likelihood_img[i,j] = (grey_value,grey_value, grey_value)
	cv2.imshow('face', likelihood_img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def test():
	descriptor_file = open(sys.argv[1])
	lT = construct_lookup_table(descriptor_file, 40)
	test_with_image(descriptor_file, lT)