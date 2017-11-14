#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import pickle
import graphical_tools
import calc_histo
import parse_file
import lookup_table
import numpy as np
from sklearn.metrics import auc
import cv2
from matplotlib import pyplot as plt
import math
from sklearn import mixture

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
        # color-clustering
        self.mean_color = 0
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

class statistics:
    def __init__(self,tp,fp,tn,fn):
        self.tp = tp
        self.fp = fp
        self.tn = tn
        self.fn = fn

# Return true if roi detects a face within the roi
def is_face(img, l_t, roi, bias):
    g_sum = 0.0
    pixel_nb = 0
    #roi.mean_color = 0
    for i in range(int(roi.l), int(roi.r)):
        for j in range(int(roi.t), int(roi.b)):
            pixel_nb += 1
            pixel_prob = l_t.get_pixel_probability(img[i,j])
            #roi.mean_color += img[i,j][0]*255*255+img[i,j][1]*255+img[i,j][2]
            g_sum += pixel_prob
    g_sum /= float(roi.w*roi.h)
    #roi.mean_color /= float(roi.w*roi.h)
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
    return (ground_truth_mask[int(x)][int(y)][0]==255)

def get_ellipse_mask(img, e):
   mask = img.copy()
   mask[:] = (0)
   # we have to extract all the ellipses
   cv2.ellipse(mask,(int(e.c_x),int(e.c_y)),(int(e.r_a), int(e.r_b)),
       int(e.theta),0,360,(255), -1)
   return mask

def get_statistics_one_image(lookup_table, img, img_info, bias, roi_c_i, roi_c_j, roi_w, roi_h, n_quantification):
    roi = region_of_interest(roi_c_i,roi_c_j,roi_w,roi_h)
    ground_truth_mask = get_ground_truth_mask(img, img_info)
    tp = 0 # true positivs, etc.
    fp = 0
    tn = 0
    fn = 0
    face_detected = False # boolean variables to count the aforementioned statistics
    true_face = False
    # while the roi is still in the image
#    detections = []
    while(roi.correct_position(img.shape)):
        face_detected = is_face(img,lookup_table,roi,bias)
        true_face = ground_truth(ground_truth_mask,roi.c_i,roi.c_j)
        # print("face_detected : " + str(face_detected))
        # print("true_face : " + str(true_face))
        if(face_detected): # we detected a face
#            detections.append((roi.c_j,roi.c_i))
            if(true_face): # decision of the ground-truth function
                tp += 1 # it really is a face
            else:
                fp += 1 # false alert
        else: # our algorithm didn't detect a face
            if(true_face): # but it actually was
                fn += 1 # we missed an actual face
            else:
                tn += 1 # our decision was correct
        roi.step_forward()

#    if(len(detections) > 0):
## CLUSTERING
#        x = np.array(detections)
#        bic = []
#        components = range(1,10)
#        for c in components:
#            if(c<len(x)):
#                sys.stdout.flush()
#                gmm = mixture.GaussianMixture(n_components=c,covariance_type="full")
#                gmm.fit(x)
#                bic.append(gmm.bic(x))
#        bic = np.array(bic)
#        best_bic_c = bic.argmin()+1
#        gmm = mixture.GaussianMixture(n_components=best_bic_c).fit(x)
#        labels = gmm.predict(x)
## construct gaussian ellipses
#        detections_ellipses = []
#        for i in range(best_bic_c):
#            mean = gmm.means_[i]
#            covariance = gmm.covariances_[i]
#            # singular value decomposition
#            U, s, Vt = np.linalg.svd(covariance) 
#            angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
#            #width, height = 2 * np.sqrt(3*s) # 3-sigma rule => 99% of all values
#            width, height = 2 * np.sqrt(1*s) 
#            detections_ellipses.append(parse_file.ellipse(height,width,
#                    math.radians(float(angle))+math.pi/2,mean[0],mean[1]))
## construct bi-parted-graph
#        d_masks = [] # masks of detected ellipses
#        l_masks = [] # masks of ground-truth ellipses
#        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#        for d in detections_ellipses:
#            d_masks.append(get_ellipse_mask(gray_img,d))
#        for l in img_info.list_ellipse:
#            l_masks.append(get_ellipse_mask(gray_img,l))
## evaluate graph result
#        bias_graph = 0.3
#        bi_graph = np.full((len(d_masks), len(l_masks)), 0.0)
#        for i,d in enumerate(d_masks):
#            for j,l in enumerate(l_masks):
#                union = cv2.bitwise_or(d,l)
#                intersection = cv2.bitwise_and(d,l)
#                #cv2.imshow('detection_mask', union)
#                union_count = cv2.countNonZero(union)
#                #print("union_count: "+str(union_count))
#                #cv2.waitKey(0)
#                #cv2.destroyAllWindows()
#                #cv2.imshow('detection_mask', intersection)
#                intersection_count = cv2.countNonZero(intersection)
#                #print("intersection_count: "+str(intersection_count))
#                #cv2.waitKey(0)
#                #cv2.destroyAllWindows()
#                bi_graph[i][j] = float(intersection_count)/float(union_count)
# get statistics
#        tp = 0
#        fp = 0
#        fn = 0
#        for d_i in range(bi_graph.shape[0]):
#            max_i = bi_graph[d_i].max()
#            if(max_i == 0.0):
#                fp += 1
#            elif((max_i+bias_graph) < 0.5):
#                fp += 1
#            elif(bi_graph[:,bi_graph[d_i].argmax()].argmax() != d_i):
#                fp += 1
#            else:
#                tp += 1
#        for l_i in range(bi_graph.shape[1]):
#            if(bi_graph[:,l_i].max()+bias_graph<0.5):
#                fn += 1
 
    return statistics(tp,fp,tn,fn)

def test_graphical(quantification):
    fd = open(sys.argv[1])
    charge_lookup_table = int(sys.argv[2])

    if(charge_lookup_table==1):
        try:
            lookup_table_fd = open("lT","rb")
            lT = pickle.load(lookup_table_fd)
            lookup_table_fd.close() 
        except IndexError as err:
            print("IndexError: {0}".format(err))
            exit(1)
    else:
        lT = lT.construct_lookup_table(fd, 1, quantification)

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
        if(is_face(img,lT,roi,0, quantification)):
            nb_detections += 1;
            if(ground_truth(ground_truth_mask,roi.c_i,roi.c_j)):
                nb_true_pos += 1
            for i in range(int(roi.l), int(roi.r)):
                for j in range(int(roi.t), int(roi.b)):
                    mask[i,j] = (255,255,255)
        roi.step_forward()
    print("nb_detections: "+str(nb_detections))
    print("nb_true_pos: "+str(nb_true_pos))
    
    plt.subplot(221)
    plt.imshow(ground_truth_mask)
    plt.subplot(222)
    plt.imshow(mask)
    plt.subplot(223)
    tmp_vec = img[:][2]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            (b,g,r) = img[i,j]
            img[i,j] = (r,g,b)
    plt.imshow(img)
    plt.show()

def view_ground_truth():
    fd = open(sys.argv[1])
    img_info = parse_file.get_img_info(fd)
    img = cv2.imread(img_info.img_path)
    ground_truth_mask = get_ground_truth_mask(img, img_info)
    graphical_tools.showImg("Dataset Image", img)
    graphical_tools.showImg("Ground_truth", ground_truth_mask)

def test_statistics():
    fd = open(sys.argv[1])
    lT = lookup_table.construct_lookup_table(fd, 40)
    img_info = parse_file.get_img_info(fd)
    img = cv2.imread(img_info.img_path)
    s = get_statistics_one_image(lT,img,img_info,0.3,2,2,5,5)
    tpr = float(s.tp)/float(s.tp+s.fn)
    fpr = float(s.fp)/float(s.fp+s.tn)
    print("tp: "+str(s.tp))
    print("fp: "+str(s.fp))
    print("tn: "+str(s.tn))
    print("fn: "+str(s.fn))
    print("tpr: "+str(tpr))
    print("fpr: "+str(fpr))

def test_roc_curve():
    fd = open(sys.argv[1]) # the file containing the ellipse+image information
    lT = lookup_table.construct_lookup_table(fd, 60)
    tpr_vec = [0]*11
    fpr_vec = [0]*11
    tp_vec = [0]*11
    fp_vec = [0]*11
    tn_vec = [0]*11
    fn_vec = [0]*11
    bias_vec = np.arange(-0.5,0.6,0.1)
    nb_images = 10
    for k in range(0,nb_images):
        print(str(k)+"-th image")
        # per-image operations
        img_info = parse_file.get_img_info(fd)
        img = cv2.imread(img_info.img_path)
        for i,bias in enumerate(bias_vec):
            s = get_statistics_one_image(lT,img,img_info,bias,5,5,11,11)
            tp_vec[i] += s.tp
            fp_vec[i] += s.fp
            tn_vec[i] += s.tn
            fn_vec[i] += s.fn
            
    # calculating the tpr, fpr
    for i in range(0,11):
        tpr_vec[i] = float(tp_vec[i])/float(tp_vec[i]+fn_vec[i])
        fpr_vec[i] = float(fp_vec[i])/float(fp_vec[i]+tn_vec[i])
    print("tpr: " + str(tpr_vec))
    print("fpr: " + str(fpr_vec))
    area_under_curve = auc(tpr_vec, fpr_vec)
    print("Area under curve : " + str(area_under_curve))
    plt.plot(fpr_vec,tpr_vec,'ro')
    plt.show()
    
if __name__ == '__main__':
    quantif = 8
    test_graphical(quantif)
