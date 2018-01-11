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
    N = 31 # ROI-size, should be odd

    # Load FDDB data
    fddb = fddb_manager.Manager()
    fddb.set_train_folders([1,2,3,4,5,6,7,8,9,10])
    fddb.set_test_folders([3])
    fddb.set_fddb_dir("../dataset")
    fddb.load_img_descriptors()
    # WIDER
    fddb.get_WIDER_img_info("../dataset/wider_face_split/wider_face_train_bbx_gt.txt")
    fddb.set_window_size(N)
    print(len(fddb.train_img_info_vec))

#    nb = 15226
#    img = cv2.imread(fddb.train_img_info_vec[nb].img_path)
#    print(fddb.train_img_info_vec[nb].img_path)
#    print(img.shape)
#    mask = graphical_tools.calc_mask(img,fddb.train_img_info_vec[nb])
#    graphical_tools.showImg("img",img)
#    #graphical_tools.showImg("mask",mask)
    
    # Build the graph for the deep net
    y_pred, y_true, x_hold, optimizer,accuracy, summary, cost, learning_rate, dropout = cnn.construct_cnn(N)

    # Debug and save
    writer = tf.summary.FileWriter("/tmp/fddb")
    saver = tf.train.Saver()
    
    old_c = 100.0
    old_eval_c = 100.0
    start_l_rate = 1e-4
    l_rate = start_l_rate
    # Train network
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer.add_graph(sess.graph)
        for i in range(1000000):
            #print("step: "+str(i))
            batch, full_batch = fddb.next_batch_train(1000)
            #batch, full_batch = fddb.next_balanced_batch_train(1000)
            if(not full_batch):
              break
            if (i % 100 == 0) and (i != 0):
                train_accuracy, c, l = sess.run([accuracy,cost,learning_rate], feed_dict={
                    x_hold: batch[0], y_true: batch[1], dropout: False, learning_rate: l_rate})
                print('step %d, training accuracy %g' % (i, train_accuracy))
                print("cost: "+str(c))
                #print("learning_rate: "+str(l))
                #print("number of face-batches: "+str(np.sum(batch[1],axis=0)))
                s = sess.run(summary, feed_dict={x_hold: batch[0], y_true: batch[1], dropout: False, learning_rate: l_rate})
                writer.add_summary(s, i)
                # We stop if we can't further optimize the network
                if(c>old_eval_c):
                    break
                old_eval_c = c
            else:
                # Adaptive learning rate
                c = sess.run(cost, feed_dict={x_hold: batch[0], y_true: batch[1], dropout: False, learning_rate: l_rate})
                while(c>old_c):
                    #print(l_rate)
                    l_rate /= 10
                    if(l_rate < 1e-30):
                        #print("c: "+str(c)+", old_c: "+str(old_c))
                        #print("BREAK: "+str(l_rate))
                        break
                    c = sess.run(cost, feed_dict={x_hold: batch[0], y_true: batch[1], dropout: False, learning_rate: l_rate})
                opti,c =  sess.run([optimizer,cost], feed_dict={x_hold: batch[0], y_true: batch[1], dropout: False, learning_rate: l_rate})
                l_rate = start_l_rate

        # Save model
        save_path = saver.save(sess, "./tmp_model/cnn_model")
        print("Model saved in file: %s" % save_path)


   # Evaluate network
