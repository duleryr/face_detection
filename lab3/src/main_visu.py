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

from tf_cnnvis import *
from scipy.misc import imread, imresize

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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

        #batch, full_batch = fddb.next_batch_test(500)
        batch, full_batch = fddb.next_batch_test(1)
        batch[0] = [cv2.imread("sample_images/images.jpg", 0)]
        #while(full_batch):
        #for i in range(0,1000):
            # out, true_face_tmp = sess.run([y_pred, y_true], feed_dict={x_hold: batch[0], y_true: batch[1], dropout: False, keep_prob: 1})
            # batch, full_batch = fddb.next_batch_test(1000)
            # scores.append(out[:,0])
            # labels.append(true_face_tmp[:,0])
            # if(not full_batch):
                # break
        #sess.close()
    # Compute TPR, FPR
    # scores = np.concatenate(scores)
    # labels = np.concatenate(labels)
    # scores_labels = zip(scores,labels)
    # scores, labels = zip(*sorted(scores_labels))
    #manually checking TP,FP
    # tp = 0
    # fp = 0
    # for i in range(len(scores)):
        # if(scores[i]>0.99):
            # if(labels[i]):
                # tp += 1
            # else:
                # fp += 1
    # print("tp: "+str(tp)+", fp :"+str(fp))
    
    #graphical_tools.plot_roc_curve(labels,scores)
    
    #mean = np.load("./img_mean.npy").transpose((1, 2, 0)) # load mean image of imagenet dataset
    
    # reading sample image
    #im = np.expand_dims(imresize(imresize(imread(os.path.join("sample_images", "images.jpg")), (256, 256)) - mean, 
    #                             (224, 224)), axis = 0)
    
    #im = np.expand_dims(imread(os.path.join("sample_images", "images.jpg")), axis = 0)
    
    #X = tf.placeholder(tf.float32, shape = [None, 33, 33, 3]) # placeholder for input images

    # open a session and initialize graph variables
    # CAVEAT: trained alexnet weights have been set as initialization values in the graph nodes.
    #         For this reason visualization can be performed just after initialization
    
    sess = tf.Session(graph=tf.get_default_graph())
    sess.run(tf.global_variables_initializer()) 
    
    # activation visualization
    layers = ['r', 'p', 'c']
    
    start = time.time()
    with sess.as_default():
    # with sess_graph_path = None, the default Session will be used for visualization.
        is_success = activation_visualization(sess_graph_path = None, value_feed_dict = {x_hold: batch[0], y_true: batch[1], dropout: False, keep_prob: 1}, 
                                              layers=layers, path_logdir=os.path.join("Log","projet"), 
                                              path_outdir=os.path.join("Output","projet"))
    start = time.time() - start
    print("Total Time = %f" % (start))
    
    #deepdream visualization
    
    start = time.time()
    with sess.as_default():
    # with sess_graph_path = None, the default Session will be used for visualization.
        is_success = deepdream_visualization(sess_graph_path = None, value_feed_dict = {x_hold: batch[0], y_true: batch[1], dropout: False, keep_prob: 1}, 
                                              classes=[1, 2, 3], layer='Conv1/Conv2D', path_logdir=os.path.join("Log","projet"), 
                                              path_outdir=os.path.join("Output","projet"))
    start = time.time() - start
    print("Total Time = %f" % (start))
    
    
    # deconv visualization
    # layers = ['r', 'p', 'c']
    
    # start = time.time()
    # with sess.as_default():
        # is_success = deconv_visualization(sess_graph_path = None, value_feed_dict = {x_hold: batch[0], y_true: batch[1], dropout: False, keep_prob: 1}, 
                                          # layers=layers, path_logdir=os.path.join("Log","projet"), 
                                          # path_outdir=os.path.join("Output","projet"))
    # start = time.time() - start
    # print("Total Time = %f" % (start))
    
    #close the session and release variables
    sess.close()
