import sys
import os
import numpy as np
from sklearn.metrics import auc
import parse_file
import cv2
from matplotlib import pyplot as plt
import pickle
import multiprocessing
import time
import graphical_tools
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import fddb_manager
import cnn
import tensorflow as tf
import roi

if __name__ == '__main__':
    """ Initialization : Choice of the files used for the training/testing """
    try:
        img_path = sys.argv[1]
        model_path = sys.argv[2]
        debug = False
    except IndexError as err:
        print("IndexError: {0}".format(err))
        print("\nAppel de la fonction :")
        print("python3 str1 str2, avec :")
        print("    str1 : image path")
        print("    str2 : model path")
        exit(1)

    # Hyper-parameters
    N = 31 # ROI-size
    CONF_THRESH = 0.99
    scale_factor = 0.95

    # Build the graph for the deep net
    y_pred, y_true, x_hold, optimizer,accuracy, cost, learning_rate, dropout, keep_prob, summary = cnn.construct_cnn(N)

    # Restore model
    saver = tf.train.Saver()

#    # groupRectangle test
#    p1 = (0,0)
#    p2 = (10,10)
#    p3 = (5,5)
#    r = []
#    r.append((p1[0],p1[1],p2[0],p2[1]))
#    r.append((p1[0],p1[1],p3[0],p3[1]))
#    g, w = cv2.groupRectangles(r,0)
#    print(g)

    # Train network
    with tf.Session() as sess:
        saver.restore(sess, model_path)
        color_img = cv2.imread(img_path)
        img = cv2.imread(img_path, 0)
        roi_eval = roi.ROI(N)
        inverse_scale = 1
        face_rect = []
        while(img.shape[0] >roi_eval.window_size and img.shape[1] > roi_eval.window_size):
            color_img_tmp = color_img.copy()
            mask = np.zeros(img.shape)
            mask[:] = 0
            while(roi_eval.c[0] != -1):
                batch = roi_eval.get_roi_content(img)
                out = sess.run(y_pred, feed_dict={x_hold: [batch], y_true:[[0,1]], dropout: False, keep_prob:1})
                #if(out[0][0]>=0.95):
                    #print(int(out[0][0]*255.0))
                    #mask[roi_eval.c[0],roi_eval.c[1]] = 255
                #mask[roi_eval.c[0],roi_eval.c[1]] = int(out[0][0]*255.0)
                #if(out[0][0]>0):
                #    print(out[0][0])
                if(out[0][0]>CONF_THRESH):
                    mask[roi_eval.c[0],roi_eval.c[1]] = out[0][0]
                    pt1 = (roi_eval.c[1]-roi_eval.half_window_size,roi_eval.c[0]-roi_eval.half_window_size)
                    pt2 = (roi_eval.c[1]+roi_eval.half_window_size,roi_eval.c[0]+roi_eval.half_window_size)
                    pt1_scaled = (int(pt1[0]*inverse_scale),int(pt1[1]*inverse_scale))
                    pt2_scaled = (int(pt2[0]*inverse_scale),int(pt2[1]*inverse_scale))
                    width = pt2_scaled[0]-pt1_scaled[0]
                    height = pt2_scaled[1]-pt1_scaled[1]
                    #face_rect.append((pt1_scaled[0], pt1_scaled[1], width, height))
                    face_rect.append((pt1_scaled[0], pt1_scaled[1], pt2_scaled[0], pt2_scaled[1]))
                    #print(pt1_scaled)
                    #print(pt2_scaled)
                    #cv2.rectangle(color_img_tmp,pt1,pt2,(255,0,0),2)
                    #cv2.rectangle(color_img_tmp,pt1_scaled,pt2_scaled,(255,0,0),2)
                roi_eval.next_step(img.shape)
            roi_eval.reset_pos()

            mask[:] /= np.max(mask[:])
            #graphical_tools.showImg("original image", color_img_tmp)
            #graphical_tools.showImg("detected faces", mask)
            img = graphical_tools.resize_img(img,scale_factor)
            #color_img = graphical_tools.resize_img(color_img,scale_factor)
            inverse_scale /= scale_factor
        grouped_faces, weights_list = cv2.groupRectangles(face_rect,4)
        print(len(grouped_faces))
        print(len(face_rect))
        #graphical_tools.showFaces(color_img,face_rect)
        graphical_tools.showFaces(color_img,grouped_faces)


        sess.close()
