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
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

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
    #N = 99
    # Load train data
    print("load train data")
    fddb = fddb_crop.Manager()
    fddb.load_images()
    # Load test data
    print("load test data")
    fddb_test = fddb_manager.Manager()
    fddb_test.set_train_folders([1])
    fddb_test.set_test_folders([10])
    fddb_test.set_fddb_dir("../dataset")
    fddb_test.load_img_descriptors()
    fddb_test.set_window_size(N)

    # Build the graph for the deep net
    y_pred, y_true, x_hold, optimizer,accuracy, cost, learning_rate, dropout, keep_prob, summary = cnn.construct_cnn(N)

    # Debug and save
    writer = tf.summary.FileWriter("/tmp/fddb")
    saver = tf.train.Saver()
    
    old_c = 100.0
    old_eval_c = 100.0
    start_l_rate = 1e-1
    l_rate = start_l_rate
    desired_cost = 1.0
    avg_acc = 0
    avg_cost = 0
    batch_size = 100
    #nb_batches = 1763
    nb_batches = 130
    nb_epochs = 20

    # plot training and test accuracy
    train_acc_vec = []
    test_acc_vec = []
    test_auc_vec = []
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

            # test the cnn
            test_batch, test_full_batch = fddb_test.next_batch_test(batch_size)
            scores = []
            labels = []
            avg_test_acc = 0
            nb_test_batches = 1
            while(full_batch):
                out, true_face_tmp, test_acc = sess.run([y_pred, y_true,accuracy], feed_dict={x_hold: batch[0], y_true: batch[1], dropout: False, keep_prob: 1})
                avg_test_acc += test_acc
                batch, full_batch = fddb_test.next_batch_test(batch_size)
                scores.append(out[:,0])
                labels.append(true_face_tmp[:,0])
                if(not full_batch):
                    break
                nb_test_batches += 1
            scores = np.concatenate(scores)
            labels = np.concatenate(labels)
            fpr,tpr, _ = roc_curve(labels,scores)
            test_auc = auc(fpr,tpr)
            avg_test_acc /= nb_test_batches
            fddb_test.reset_img_counter()

            # Display information
            avg_acc /= nb_batches
            avg_cost /= nb_batches
            print("Epoch: "+str(e))
            print("train_accuracy: "+str(avg_acc))
            print("test_accuracy: "+str(avg_test_acc))
            print("test_auc: "+str(test_auc))
            print("cost: "+str(avg_cost))
            train_acc_vec.append(avg_acc)
            test_acc_vec.append(avg_test_acc)
            test_auc_vec.append(test_auc)
            avg_acc = 0
            avg_cost = 0
            fddb.reset_img_counter()

        train_plot = plt.plot(train_acc_vec, label = "Training accuracy")
        test_plot = plt.plot(test_acc_vec, label = "Test accuracy")
        test_auc_plot = plt.plot(test_auc_vec, label = "Test AUC")
        #train_legend = plt.legend([train_plot,test_plot])
        train_legend = plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.ylim([0.0, 1.05])
        plt.show()
        # Save model
        save_path = saver.save(sess, "./tmp_model/cnn_model")
        print("Model saved in file: %s" % save_path)
