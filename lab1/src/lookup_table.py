#! /usr/bin/env python

import sys
import calc_histo
import parse_file
import numpy as np
import cv2
from matplotlib import pyplot as plt

class LookupTable:
    def __init__(self,mode=calc_histo.Color.RGB,n_quantification=1):
        self.mode = mode
        self.n_quantification = n_quantification
        nb_color_values = int(256/n_quantification)
        if(self.mode == calc_histo.Color.RGB):
            self.table = np.full((nb_color_values,nb_color_values,nb_color_values),0.0)
        elif(self.mode == calc_histo.Color.RG):
            self.table = np.full((nb_color_values,nb_color_values),0.0)
        self.list_hist_sum = []
   
    def calc_fold_histograms(self, fd, nb_images):
        # Initialize the sum of histograms with the first image
        img_info = parse_file.get_img_info(fd)
        img_global = cv2.imread(img_info.img_path)
        masked_img = calc_histo.calc_mask(img_global, img_info)
        hist_t = calc_histo.calc_hist(img_global, masked_img, self.mode, self.n_quantification)
        hist_all = calc_histo.calc_normal_hist(img_global, self.mode, self.n_quantification)
        
        # Initialization
        hist_t_sum = hist_t
        hist_all_sum = hist_all
        
        for i in range(1,nb_images):
            img_info = parse_file.get_img_info(fd)
            img_global = cv2.imread(img_info.img_path)
            masked_img = calc_histo.calc_mask(img_global, img_info)
            hist_t = calc_histo.calc_hist(img_global, masked_img, self.mode, self.n_quantification)
            hist_all = calc_histo.calc_normal_hist(img_global, self.mode, self.n_quantification)
            hist_t_sum += hist_t
            hist_all_sum += hist_all
            print(str(i)+"-th image processed")
        self.list_hist_sum.append((hist_t_sum, hist_all_sum))
        #DEBUG
        #plot_array(hist_t_sum, "histogram target zone")
        #plot_array(hist_all_sum, "histogram all pixels")

    # Return the likelihood table(r,g,b) for nb_images images
    def construct_lookup_table(self):
    
        global_hist_t_sum = self.list_hist_sum[0][0]
        global_hist_all_sum = self.list_hist_sum[0][1]

        for tuple_hist in self.list_hist_sum[1:]:
            global_hist_t_sum += tuple_hist[0]
            global_hist_all_sum += tuple_hist[1]

        if(self.mode == calc_histo.Color.RGB):
            for i in range(self.table.shape[0]):
                for j in range(self.table.shape[1]):
                    for k in range(self.table.shape[2]):
                        if(global_hist_all_sum[i][j][k] != 0):
                            self.table[i][j][k] = float(global_hist_t_sum[i][j][k])/float(global_hist_all_sum[i][j][k])
        elif(self.mode == calc_global_histo.Color.RG):
            for i in range(self.table.shape[0]):
                for j in range(self.table.shape[1]):
                    if(global_hist_all_sum[i][j] != 0):
                        self.table[i][j] = float(global_hist_t_sum[i][j])/float(global_hist_all_sum[i][j])
    
    def get_pixel_probability(self, pixel):
        if(self.mode == calc_histo.Color.RGB):
            r = int(pixel[2]/self.n_quantification)
            g = int(pixel[1]/self.n_quantification)
            b = int(pixel[0]/self.n_quantification)
            return self.table[r][g][b]
        elif(self.mode == calc_histo.Color.RG):
            r = int(pixel[2]) 
            g = int(pixel[1])
            b = int(pixel[0])
            nb_color_values = int(256/self.n_quantification)
            l = float(r+g+b)
            r_norm = float(r)
            g_norm = float(g)
            if(l != 0.0):
                r_norm /= l 
                g_norm /= l
            return self.table[int((nb_color_values-1)*r_norm)][int((nb_color_values-1)*g_norm)]

    def plot(self, title):
        plt.plot(self.table.flatten())
        plt.xlim([0,self.table.size-1])
        plt.title(title)
        plt.show()

# General plot array function
def plot_array(vec, title):
    plt.plot(vec.flatten())
    plt.xlim([0,vec.size-1])
    plt.title(title)
    plt.show()
 
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
