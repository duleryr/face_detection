#! /usr/bin/env python

import sys
import calc_histo
import parse_file
import numpy as np
import cv2
from matplotlib import pyplot as plt

def plot_array(vec, title):
    plt.plot(vec.flatten())
    plt.xlim([0,vec.size-1])
    plt.title(title)
    plt.show()
    
# Return the likelihood table(r,g,b) for nb_images images
def construct_lookup_table(fd, nb_images, n_quantification):
    # Initialize the sum of histograms with the first image
    img_info = parse_file.get_img_info(fd)
    img_global = cv2.imread(img_info.img_path)
    masked_img = calc_histo.calc_mask(img_global, img_info)
    hist_t = calc_histo.calc_hist(img_global, masked_img, n_quantification)
    hist_all = calc_histo.calc_normal_hist(img_global, n_quantification)
    
    # Initialization
    hist_t_sum = hist_t
    hist_all_sum = hist_all
    
    for i in range(1,nb_images):
        img_info = parse_file.get_img_info(fd)
        img_global = cv2.imread(img_info.img_path)
        masked_img = calc_histo.calc_mask(img_global, img_info)
        hist_t = calc_histo.calc_hist(img_global, masked_img, n_quantification)
        hist_all = calc_histo.calc_normal_hist(img_global, n_quantification)
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
    plot_array(hist_t_sum, "histogram target zone")
    plot_array(hist_all_sum, "histogram all pixels")

    return lookup_table

def pixel_probability(lookup_table, pixel, n_quantification):
    r = int(pixel[0]/n_quantification)
    g = int(pixel[1]/n_quantification)
    b = int(pixel[2]/n_quantification)
    return lookup_table[r][g][b]

# Fill the next image with the likelihood-values (Black and white style)
def test_with_image(fd, lookup_table, n_quantification = 8):
    img_info = parse_file.get_img_info(fd)
    img = cv2.imread(img_info.img_path)
    likelihood_img = img.copy()
    for i in range(0, likelihood_img.shape[0]):
        for j in range(0, likelihood_img.shape[1]):
            pixel_prob = pixel_probability(lookup_table,likelihood_img[i,j], n_quantification)
            grey_value = int((255)*pixel_prob)
            likelihood_img[i,j] = (grey_value,grey_value, grey_value)
    cv2.imshow('face', likelihood_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test():
    descriptor_file = open(sys.argv[1])
    quantif = 8
    lT = construct_lookup_table(descriptor_file, 40, quantif)
    test_with_image(descriptor_file, lT, quantif)

if __name__ == '__main__':
    test()
