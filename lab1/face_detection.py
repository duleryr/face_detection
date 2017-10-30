#! /usr/bin/env python

import sys
import calc_histo
import parse_file
import lookup_table
import numpy as np
import cv2
from matplotlib import pyplot as plt
import math

class region_of_interest:
	def __init__(self,c_i,c_j,w,h):
		self.t = c_j-(h-1)/2
		self.b = c_j+(h-1)/2
		self.l = c_i-(w-1)/2
		self.r = c_i+(w+1)/2
		self.c_i = c_i
		self.c_j = c_j
		self.w = w
		self.h = h
		self.step_size_i = w/2
		self.step_size_j = h/2
	def correct_position(self,img_shape):
		if(self.r>img_shape[0]):
			self.t += self.step_size_j
			self.b += self.step_size_j
			self.l = 0
			self.r = self.w
			self.c_i = (self.l+self.r-1)/2
			self.c_j = (self.b+self.t-1)/2
		if(self.b>img_shape[1]):
			return False
		return True
	# the roi only steps horizontally; 
	# the vertical position will only be corrected!
	def step_forward(self):
		self.l += self.step_size_i
		self.r += self.step_size_i
		self.c_i += self.step_size_i

# Return true if roi detects a face within the roi
def is_face(img, l_t, roi, bias):
	g_sum = 0.0
	for i in range(roi.l, roi.r):
		for j in range(roi.t, roi.b):
			pixel_prob = lookup_table.pixel_probability(l_t,img[i,j])
			mean_pixel_prob = (pixel_prob[0]+pixel_prob[1]+pixel_prob[2])/3.0
			g_sum += mean_pixel_prob
	g_sum /= (roi.w*roi.h)
	return ((g_sum+bias)>0.5)

# Return true if the coordinates are inside the ellipse
# TODO: Debug, doesn't work
def in_ellipse(e, y, x):
	theta = math.radians(float(e.theta))
	return ((pow((x-e.c_x)*math.cos(theta)+(y-e.c_y)*math.sin(theta),2.0)/(e.r_a*e.r_a)+
			 pow((x-e.c_x)*math.sin(theta)+(y-e.c_y)*math.cos(theta),2.0)/(e.r_b*e.r_b)) <= 1.0)
	
def get_ground_truth_mask(img, img_info):
	mask = img.copy()
	mask[:] = (0, 0, 0)
	# we have to extract all the ellipses
	for i in range(0, img_info.nb_faces):
		e_tmp = img_info.list_ellipse[i]
		cv2.ellipse(mask,(int(e_tmp.c_x),int(e_tmp.c_y)),(int(e_tmp.r_a),
			int(e_tmp.r_b)),int(e_tmp.theta),0,360,(255, 255, 255), -1)
	return mask

# Return true if (i,j) is part of a face
# TODO: ground truth function by using only the ellipse parameters
def ground_truth(ground_truth_mask, x, y):
	#is_in_ellipse = False
	# we have to extract all the ellipses
	#is_in_ellipse = is_in_ellipse or in_ellipse(e_tmp,x,y)
	#return is_in_ellipse
	return (ground_truth_mask[x,y][0]==255)
 
			
def test():
	fd = open(sys.argv[1])
	lT = lookup_table.construct_lookup_table(fd, 40)
	img_info = parse_file.get_img_info(fd)
	img = cv2.imread(img_info.img_path)
	
	#let's create a mask that shows us where we got a positive detection
	mask = img.copy()
	mask[:] = (0,0,0)
	roi = region_of_interest(2,2,5,5)
	nb_detections = 0
	nb_true_pos = 0
	#DEBUG
	ground_truth_mask = get_ground_truth_mask(img, img_info)
	while(roi.correct_position(img.shape)):
		if(is_face(img,lT,roi,0.3)):
			nb_detections += 1;
			if(ground_truth(ground_truth_mask,roi.c_i,roi.c_j)):
				nb_true_pos += 1
				for i in range(roi.l, roi.r):
					for j in range(roi.t, roi.b):
						mask[i,j] = (255,255,255)
		roi.step_forward()
	print("nb_detections: "+str(nb_detections))
	print("nb_true_pos: "+str(nb_true_pos))
	
	plt.subplot(221)
	plt.imshow(ground_truth_mask)
	plt.subplot(222)
	plt.imshow(mask)
	plt.subplot(223)
	plt.imshow(img, 'gray')
	plt.show()
test()
