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
import random

if __name__ == '__main__':
    """ Initialization : Choice of the files used for the training/testing """
    try:
        #file_number_used = sys.argv[1]
        #if (len(file_number_used) == 1):
        #    file_number_used = "0" + file_number_used
        debug = False
    except IndexError as err:
        print("IndexError: {0}".format(err))
        print("\nAppel de la fonction :")
        print("python3 int1 int2 str3, avec :")
        print("    int1 : numéro du fichier ellipseList à utiliser, entre 1 et 10")
        print("    int2 : nombre d'images à utiliser pour les tests de détection")
        exit(1)

    # Hyper-parameters
    N = 11 # ROI-size

    # Load FDDB data
    fddb = fddb_manager.Manager()
    fddb.set_train_folders([1,2])
    fddb.set_test_folders([3])
    fddb.set_fddb_dir("../dataset")
    fddb.load_img_descriptors()
    fddb.set_window_size(N)
    
    # Build the graph for the deep net
    y_pred, y_true, x_hold, optimizer,accuracy, summary = cnn.construct_cnn(N)

    # Debug and save
    writer = tf.summary.FileWriter("/tmp/fddb")
    saver = tf.train.Saver()

    # Train network
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        writer.add_graph(sess.graph)
        for i in range(10000):
          batch = fddb.next_batch_train(100)
          if i % 100 == 0:
              train_accuracy = sess.run(accuracy, feed_dict={
                  x_hold: batch[0], y_true: batch[1]})
              print('step %d, training accuracy %g' % (i, train_accuracy))
              print("number of face-batches: "+str(np.sum(batch[1],axis=0)))
              s = sess.run(summary, feed_dict={x_hold: batch[0], y_true: batch[1]})
              writer.add_summary(s, i)
          sess.run(optimizer, feed_dict={x_hold: batch[0], y_true: batch[1]})

        # Save model
        #save_path = saver.save(sess, "./cnn_model")
        #print("Model saved in file: %s" % save_path)


    # Evaluate network
