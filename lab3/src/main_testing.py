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
import fddb_crop
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

    N = 31
    # Load FDDB data
    fddb = fddb_crop.Manager()
    fddb.load_images()

    # Build the graph for the deep net
    y_pred, y_true, x_hold, optimizer,accuracy, cost, learning_rate, dropout, keep_prob, summary = cnn.construct_cnn(N)

    # Debug and save
    writer = tf.summary.FileWriter("/tmp/fddb")
    saver = tf.train.Saver()
    
    old_c = 100.0
    old_eval_c = 100.0
    start_l_rate = 1e-4
    l_rate = start_l_rate
    desired_cost = 1.0
    avg_acc = 0
    avg_cost = 0
    batch_size = 100
    nb_batches = 1763
    nb_epochs = 10
    # Train network
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer.add_graph(sess.graph)
        for e in range(nb_epochs):
            for i in range(nb_batches):
                batch, full_batch = fddb.next_batch_train(batch_size)
                if(not full_batch):
                  break
                train_accuracy, c, l = sess.run([accuracy,cost,learning_rate], feed_dict={
                    x_hold: batch[0], y_true: batch[1], dropout: False, learning_rate: l_rate, keep_prob: 1})
                avg_acc += train_accuracy
                avg_cost += c
                opti,c =  sess.run([optimizer,cost], feed_dict={x_hold: batch[0], y_true: batch[1], dropout: True, learning_rate: l_rate, keep_prob: 0.5})
                s = sess.run(summary, feed_dict={
                    x_hold: batch[0], y_true: batch[1], dropout: False, learning_rate: l_rate, keep_prob: 1})
                writer.add_summary(s, i+e*nb_batches)

            avg_acc /= nb_batches
            avg_cost /= nb_batches
            print("Epoch: "+str(e))
            print("accuracy: "+str(avg_acc))
            print("cost: "+str(avg_cost))
            avg_acc = 0
            avg_cost = 0
            fddb.reset_img_counter()

        # Save model
        save_path = saver.save(sess, "./tmp_model/cnn_model")
        print("Model saved in file: %s" % save_path)
