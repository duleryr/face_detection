#!/usr/bin/env python

import sys
import numpy as np
import cv2
import time
import graphical_tools

# img_shape: [cols, rows]
def get_ellipse_mask(img_shape, e):
    mask = np.zeros(img_shape)
    #mask[:] = (0)
    # we have to extract all the ellipses
    cv2.ellipse(mask,(int(e.c_x),int(e.c_y)),(int(e.r_a), int(e.r_b)),
        int(e.theta),0,360,(255), -1)
    return mask

def get_rectangle_mask(img_shape, r):
    mask = np.zeros(img_shape)
    cv2.rectangle(mask,(r[0],r[1]),(r[0]+r[2],r[1]+r[3]),(255),-1)
    return mask

# compute score as according to the paper fddb
def compute_score(d, l):
    intersection = cv2.bitwise_and(d,l)
    union = cv2.bitwise_or(d,l)
    return intersection.sum()/union.sum()

# ellipses: the annotated ellipses for this image
# faces: the faces detected by the algorithm
def evaluate(img_info, faces):
    # construct masks for the annotated faces
    real_face_masks = []
    detected_face_masks = []
    for e in img_info.list_ellipse:
        real_face_masks.append(get_ellipse_mask(img_info.img_shape,e))
    for f in faces:
        detected_face_masks.append(get_rectangle_mask(img_info.img_shape,f))
    # DEBUG
    #for r in real_face_masks:
    #    graphical_tools.showImg("debug",r)
    #for d in detected_face_masks:
    #    graphical_tools.showImg("debug",d)
    score_mat = np.zeros([len(real_face_masks),len(detected_face_masks)])
    for i in range(len(real_face_masks)):
        for j in range(len(detected_face_masks)):
            score_mat[i,j] = compute_score(detected_face_masks[j],real_face_masks[i])
    # TODO: apply hungarian algorithm
    # TODO: showImg TP,FP,FN
    print(score_mat)
