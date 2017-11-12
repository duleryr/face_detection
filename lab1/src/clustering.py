#! /usr/bin/env python
import sys
import pickle
import graphical_tools
import calc_histo
import parse_file
import lookup_table
import face_detection
import numpy as np
from sklearn.metrics import auc
import cv2
from matplotlib import pyplot as plt
from matplotlib import patches
import math
from sklearn import mixture


def test():
    fd = open(sys.argv[1])
    lT_fd = str(sys.argv[2])
    bias = float(sys.argv[3])
    k_th_image = int(sys.argv[4])

    try:
        lookup_table_fd = open(lT_fd,"rb")
        lT = pickle.load(lookup_table_fd)
        lookup_table_fd.close() 
    except IndexError as err:
        print("IndexError: {0}".format(err))
        exit(1)

    for _ in range(k_th_image):
        img_info = parse_file.get_img_info(fd)
    img = cv2.imread(img_info.img_path)
    
    #let's create a mask that shows us where we got a positive detection
    mask = img.copy()
    mask[:] = (0,0,0)
    #roi = face_detection.region_of_interest(4,4,9,9)
    roi = face_detection.region_of_interest(10,10,21,21)
    nb_detections = 0
    nb_true_pos = 0
    #DEBUG
    ground_truth_mask = face_detection.get_ground_truth_mask(img, img_info)
    detections = []
    while(roi.correct_position(img.shape)):
        if(face_detection.is_face(img,lT,roi,bias)):
            nb_detections += 1;
            if(face_detection.ground_truth(ground_truth_mask,roi.c_i,roi.c_j)):
                nb_true_pos += 1
            detections.append((roi.c_j,roi.c_i))
            for i in range(int(roi.l), int(roi.r)):
                for j in range(int(roi.t), int(roi.b)):
                    mask[i,j] = (255,255,255)
        roi.step_forward()
    if(nb_detections == 0):
        print("NO DETECTIONS, PLEASE ADJUST BIAS")
        exit(1)
    #print("nb_detections: "+str(nb_detections))
    #print("nb_true_pos: "+str(nb_true_pos))

    # CLUSTERING
    x = np.array(detections)
    bic = []
    components = range(1,10)
    for c in components:
        if(c<x.size):
            gmm = mixture.GaussianMixture(n_components=c,covariance_type="full")
            gmm.fit(x)
            bic.append(gmm.bic(x))
    bic = np.array(bic)
    best_bic_c = bic.argmin()+1
    #print("best number of components: " + str(best_bic_c+1))

    # PLOT THE RESULTS
    plt.suptitle("Bias= "+str(bias)+", ROI= ("+str(roi.w)+", "+str(roi.h)+")")
    plt.subplot(221)
    tmp_vec = img[:][2]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            (b,g,r) = img[i,j]
            img[i,j] = (r,g,b)
    plt.title("Original Image")
    plt.imshow(img)
    plt.subplot(222)
    plt.title("Ground Truth")
    plt.imshow(ground_truth_mask)
    plt.subplot(223)
    plt.title("Positive Detections")
    plt.imshow(mask)

    # GMM
    window = plt.subplot(224)
    img_gmm = img.copy()
    plt.title("Model Fit")
    plt.xlim(0, img.shape[1])
    plt.ylim(0, img.shape[0])
    gmm = mixture.GaussianMixture(n_components=best_bic_c).fit(x)
    labels = gmm.predict(x)
    for i in range(best_bic_c):
        mean = gmm.means_[i]
        covariance = gmm.covariances_[i]
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
        window.add_artist(patches.Ellipse(mean, width, height, angle,color="r"))
    plt.scatter(x[:,0],x[:,1],c=labels)
    plt.gca().invert_yaxis()
    plt.imshow(img_gmm)

    # TODO: Maximize window
    plt.show()

test()
# example command: python clustering.py ../dataset/FDDB_dataset/FDDB-folds/FDDB-fold-02-ellipseList.txt lT_1_280_RG 0.03 6
