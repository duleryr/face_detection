import sys
import os
import numpy as np
import parse_file
import cv2
from matplotlib import pyplot as plt
import pickle
import multiprocessing
import time
import graphical_tools
from sklearn.metrics import precision_recall_curve
import fddb_manager
import cnn
import tensorflow as tf
import roi

if __name__ == '__main__':
    """ Initialization : Choice of the files used for the training/testing """
    try:
        model_path = sys.argv[1]
        debug = False
    except IndexError as err:
        print("IndexError: {0}".format(err))
        print("\nAppel de la fonction :")
        print("python3 str1, avec :")
        print("    str1 : model path")
        exit(1)

    # Hyper-parameters
    N = 31 # ROI-size
    CONF_THRESH = 0.99

    # Load FDDB data
    fddb = fddb_manager.Manager()
    fddb.set_train_folders([1])
    fddb.set_test_folders([10])
    fddb.set_fddb_dir("../dataset")
    fddb.load_img_descriptors()
    fddb.set_window_size(N)

    # Build the graph for the deep net
    y_pred, y_true, x_hold, optimizer,accuracy, cost, learning_rate, dropout, keep_prob, summary = cnn.construct_cnn(N)

    # Restore model
    saver = tf.train.Saver()

    scores = []
    labels = []
    # Evaluate network
    with tf.Session() as sess:
        saver.restore(sess, model_path)

        batch, full_batch = fddb.next_batch_test(1000)
        while(full_batch):
        #for i in range(0,1000):
            out, true_face_tmp = sess.run([y_pred, y_true], feed_dict={x_hold: batch[0], y_true: batch[1], dropout: False, keep_prob: 1})
            batch, full_batch = fddb.next_batch_test(1000)
            scores.append(out[:,0])
            labels.append(true_face_tmp[:,0])
            if(not full_batch):
                break
        sess.close()
    # Compute TPR, FPR
    scores = np.concatenate(scores)
    labels = np.concatenate(labels)
    scores_labels = zip(scores,labels)
    scores, labels = zip(*sorted(scores_labels))
    # manually checking TP,FP
    tp = 0
    fp = 0
    for i in range(len(scores)):
        if(scores[i]>0.99):
            if(labels[i]):
                tp += 1
            else:
                fp += 1
    print("tp: "+str(tp)+", fp :"+str(fp))

    graphical_tools.plot_roc_curve(labels,scores)
