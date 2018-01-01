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

    # Load FDDB data
    fddb = fddb_manager.Manager()
    fddb.set_train_folders([1,2])
    fddb.set_test_folders([3])
    fddb.set_fddb_dir("../dataset")
    fddb.load_img_descriptors()
    print(len(fddb.train_img_info_vec))
    print(len(fddb.test_img_info_vec))
    print(fddb.train_img_info_vec[0].img_path)
    fddb.set_window_size(101)

    fddb.next_batch_aux(5,fddb.train_img_info_vec,fddb.train_img_counter,fddb.train_batch_counter,fddb.train_roi)
    
    # Construct CNN

    # Train network

    # Evaluate network
