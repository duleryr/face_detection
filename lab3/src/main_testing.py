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
    
    #batch = fddb.next_batch_train(10)
    # Construct CNN
    #cnn.construct_cnn(batch[0],batch[1],N)

    # Create the model
    #x = tf.placeholder(tf.float32, [None, 11,3])
    #print(x)

    ## Define loss and optimizer
    #ground_truth = tf.placeholder(tf.float32, [None,2])
    #print(ground_truth)

    # Build the graph for the deep net
    y_conv, x_hold,y_hold,keep_prob,summary = cnn.construct_cnn(N)
    with tf.name_scope("xEntropy"):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_hold,
                                                                logits=y_conv)
        cross_entropy = tf.reduce_mean(cross_entropy)
    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_hold, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)

    writer = tf.summary.FileWriter("/tmp/fddb")
    # Train network

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      writer.add_graph(sess.graph)
      for i in range(1000):
        #print("hmm")
        batch = fddb.next_batch_train(100)
        #print(len(batch[1][0]))
        #print(batch[0].shape)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x_hold: batch[0], y_hold: batch[1], keep_prob: 0.5})
            print('step %d, training accuracy %g' % (i, train_accuracy))
            s = sess.run(summary, feed_dict={x_hold: batch[0], y_hold: batch[1], keep_prob: 0.5})
            writer.add_summary(s, i)
        train_step.run(feed_dict={x_hold: batch[0], y_hold: batch[1], keep_prob: 0.5})

    # Evaluate network
