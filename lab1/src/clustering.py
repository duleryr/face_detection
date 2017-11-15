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

def get_ellipse_mask(img, e):
    mask = img.copy()
    mask[:] = (0)
    # we have to extract all the ellipses
    cv2.ellipse(mask,(int(e.c_x),int(e.c_y)),(int(e.r_a), int(e.r_b)),
        int(e.theta),0,360,(255), -1)
    return mask

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
    color_detections = []
    while(roi.correct_position(img.shape)):
        if(face_detection.is_face(img,lT,roi,bias)):
            color_detections.append(roi.mean_color)
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
    #print(detections)
    #print(color_detections)

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
    #plt.suptitle("Bias= "+str(bias)+", ROI= ("+str(roi.w)+", "+str(roi.h)+")")
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
    #img_gmm[:] = (255,255,255)
    plt.title("Model Fit")
    plt.xlim(0, img.shape[1])
    plt.ylim(0, img.shape[0])
    gmm = mixture.GaussianMixture(n_components=best_bic_c).fit(x)
    labels = gmm.predict(x)
    detections_ellipses = []
    for i in range(best_bic_c):
        mean = gmm.means_[i]
        covariance = gmm.covariances_[i]
        # singular value decomposition
        U, s, Vt = np.linalg.svd(covariance) 
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(3*s) # 3-sigma rule => 99% of all values
        #detections_ellipses.append(parse_file.ellipse(width,height,math.degrees(float(angle)),mean[0],mean[1]))
        window.add_artist(patches.Ellipse(mean, width, height, angle,color="r"))
        width, height = 2 * np.sqrt(1*s) 
        detections_ellipses.append(parse_file.ellipse(height,width,math.radians(float(angle))+math.pi/2,mean[0],mean[1]))
    #plt.scatter(x[:,0],x[:,1],c=labels)
    plt.gca().invert_yaxis()
    plt.imshow(img_gmm)

    # TODO: Maximize window

#    # construct bi-parted-graph
#    d_masks = [] # masks of detected ellipses
#    l_masks = [] # masks of ground-truth ellipses
#    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#    for d in detections_ellipses:
#        d_masks.append(get_ellipse_mask(gray_img,d))
#    for l in img_info.list_ellipse:
#        l_masks.append(get_ellipse_mask(gray_img,l))
#    all_mask = gray_img
#    all_mask[:] = (0)
#    for l in l_masks:
#        cv2.bitwise_or(all_mask,l,all_mask)
#        #cv2.imshow('detection_mask', d)
#        #cv2.waitKey(0)
#        #cv2.destroyAllWindows()
#    plt.subplot(223)
#    plt.imshow(all_mask)
#    plt.show()

#    bias_graph = 0.3
#    bi_graph = np.full((len(d_masks), len(l_masks)), 0.0)
#    for i,d in enumerate(d_masks):
#        for j,l in enumerate(l_masks):
#            union = cv2.bitwise_or(d,l)
#            intersection = cv2.bitwise_and(d,l)
#            #cv2.imshow('detection_mask', union)
#            union_count = cv2.countNonZero(union)
#            print("union_count: "+str(union_count))
#            #cv2.waitKey(0)
#            #cv2.destroyAllWindows()
#            #cv2.imshow('detection_mask', intersection)
#            intersection_count = cv2.countNonZero(intersection)
#            print("intersection_count: "+str(intersection_count))
#            #cv2.waitKey(0)
#            #cv2.destroyAllWindows()
#            bi_graph[i][j] = float(intersection_count)/float(union_count)
#    print(bi_graph)

    # get statistics
#    tp = 0
#    fp = 0
#    fn = 0
#    for d_i in range(bi_graph.shape[0]):
#        max_i = bi_graph[d_i].max()
#        print(max_i)
#        if(max_i == 0.0):
#            fp += 1
#        elif((max_i+bias_graph) < 0.5):
#            fp += 1
#        elif(bi_graph[:,bi_graph[d_i].argmax()].argmax() != d_i):
#            print(bi_graph[:,bi_graph[d_i].argmax()].argmax())
#            fp += 1
#        else:
#            tp += 1
#    for l_i in range(bi_graph.shape[1]):
#        if(bi_graph[:,l_i].max()+bias_graph<0.5):
#            fn += 1
#    print("tp: "+str(tp))
#    print("fp: "+str(fp))
#    print("fn: "+str(fn))

#    # COLOR CLUSTERING
#    y = np.array(color_detections)
#    y[:,0] -= y[:,0].min()
#    y[:,1] -= y[:,1].min()
#    y[:,0] /= y[:,0].max()
#    y[:,1] /= y[:,1].max()
#    y[:,0] *= 400
#    y[:,1] *= 450
#    print(y)
#    bic = []
#    components = range(1,10)
#    for c in components:
#        if(c<y.size):
#            gmm = mixture.GaussianMixture(n_components=c,covariance_type="full")
#            gmm.fit(y)
#            bic.append(gmm.bic(y))
#    bic = np.array(bic)
#    print("bic:")
#    print(bic)
#    best_bic_c = bic.argmin()+1
#    #print("best number of components: " + str(best_bic_c+1))
#    # GMM
#    window = plt.subplot(223)
#    img_gmm = img.copy()
#    plt.title("Model Fit")
#    plt.xlim(0, img.shape[1])
#    plt.ylim(0, img.shape[0])
#    print("best_bic: "+str(best_bic_c))
#    gmm = mixture.GaussianMixture(n_components=best_bic_c).fit(y)
#    labels = gmm.predict(y)
#    detections_ellipses = []
#    for i in range(best_bic_c):
#        mean = gmm.means_[i]
#        covariance = gmm.covariances_[i]
#        # singular value decomposition
#        U, s, Vt = np.linalg.svd(covariance) 
#        print(s)
#        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
#        width, height = 2 * np.sqrt(3*s) # 3-sigma rule => 99% of all values
#        #detections_ellipses.append(parse_file.ellipse(width,height,math.degrees(float(angle)),mean[0],mean[1]))
#        #window.add_artist(patches.Ellipse(mean, width, height, angle,color="r"))
#        width, height = 2 * np.sqrt(1*s) 
#        detections_ellipses.append(parse_file.ellipse(height,width,math.radians(float(angle))+math.pi/2,mean[0],mean[1]))
#    #plt.scatter(x[:,0],x[:,1],c=labels)
#    plt.scatter(x[:,0],x[:,1],c=labels)
#    plt.gca().invert_yaxis()
#    plt.imshow(img_gmm)

    fig = plt.gcf()
    fig.set_size_inches((12,10))
    fig.set_dpi(100)
    fig.savefig("resume.png")
    plt.show()


 
test()
# example command: python clustering.py ../dataset/FDDB_dataset/FDDB-folds/FDDB-fold-02-ellipseList.txt lT_1_280_RG 0.03 6
