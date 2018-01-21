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
    # Load test data
    fddb_test = fddb_manager.Manager()
    #fddb_test.set_train_folders([1,2,3,4,5,6,7,8,9])
    fddb_test.get_WIDER_img_info("../dataset/wider_face_split/wider_face_train_bbx_gt.txt")
    #fddb_test.set_test_folders([10])
    #fddb_test.set_fddb_dir("../dataset")
    #fddb_test.load_img_descriptors()
    print("before")
    fddb_test.set_window_size(N)
    #fddb_test.extract_faces()
    fddb_test.create_non_faces()
    #fddb_test.go_through_pyramid_face(fddb_test.train_img_info_vec[0])
    #batch, full_batch = fddb_test.next_test_batch_scale(10)
    #while(full_batch):
    #    for b in batch[0]:
    #        graphical_tools.showImg("b",b)
    #    batch, full_batch = fddb_test.next_test_batch_scale(10)
