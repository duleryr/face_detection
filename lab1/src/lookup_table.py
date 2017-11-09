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
	n_quantification = 8
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
		hist_t_sum += hist_t
		hist_all_sum += hist_all
		print(str(i)+"-th image processed")

	lookup_table = np.full(hist_t_sum.shape, 0.0)
	for i in range(lookup_table.shape[0]):
	    for j in range(lookup_table.shape[1]):
	        for k in range(lookup_table.shape[2]):
	            if(hist_all_sum[i][j][k] != 0):
	                lookup_table[i][j][k] = float(hist_t_sum[i][j][k])/float(hist_all_sum[i][j][k])
	#DEBUG
	plt.plot(hist_t_sum.flatten())
	plt.xlim([0,hist_t_sum.size-1])
	plt.title("histogram target zone")
	plt.show()
	plt.plot(hist_all_sum.flatten())
	plt.xlim([0,hist_all_sum.size-1])
	plt.title("histogram all pixels")
	plt.show()
	plt.plot(lookup_table.flatten())
	plt.xlim([0,lookup_table.size-1])
	plt.title("lookup table")
	plt.show()

	return lookup_table

def pixel_probability(lookup_table, pixel):
	n_quantification = 8
	return (lookup_table[pixel[0]][pixel[1]][pixel[2]])

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

if __name__ == '__main__':
    test()
