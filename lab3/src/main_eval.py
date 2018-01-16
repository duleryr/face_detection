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

    # Build the graph for the deep net
    y_pred, y_true, x_hold, optimizer,accuracy, cost, learning_rate, dropout, keep_prob, summary = cnn.construct_cnn(N)

    # Restore model
    saver = tf.train.Saver()

    # Train network
    with tf.Session() as sess:
        saver.restore(sess, model_path)
        img = cv2.imread(img_path, 0)
        #img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        mask = np.zeros(img.shape)
        #mask = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        mask[:] = 0
        roi_eval = roi.ROI(N)
        while(roi_eval.c[0] != -1):
            batch = roi_eval.get_roi_content(img)
            out = sess.run(y_pred, feed_dict={x_hold: [batch], y_true:[[0,1]], dropout: False, keep_prob:1})
            #if(out[0][0]>=0.95):
                #print(int(out[0][0]*255.0))
                #mask[roi_eval.c[0],roi_eval.c[1]] = 255
            #mask[roi_eval.c[0],roi_eval.c[1]] = int(out[0][0]*255.0)
            #if(out[0][0]>0):
            #    print(out[0][0])
            mask[roi_eval.c[0],roi_eval.c[1]] = out[0][0]

            roi_eval.next_step(img.shape)
            print(roi_eval.c)
        print(np.sum(mask))
        print(mask)
        mask[:] /= np.max(mask[:])
        sess.close()
        graphical_tools.showImg("original image", img)
        graphical_tools.showImg("detected faces", mask)
