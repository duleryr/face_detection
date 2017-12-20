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
import single_image_evaluation
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

if __name__ == '__main__':
    """ Initialization : Choice of the files used for the training/testing """
    try:
        file_number_used = sys.argv[1]
        if (len(file_number_used) == 1):
            file_number_used = "0" + file_number_used
        nb_images_testing = int(sys.argv[2])
        min_neighbours = int(sys.argv[3])
        debug = False
    except IndexError as err:
        print("IndexError: {0}".format(err))
        print("\nAppel de la fonction :")
        print("python3 int1 int2 str3, avec :")
        print("    int1 : numéro du fichier ellipseList à utiliser, entre 1 et 10")
        print("    int2 : nombre d'images à utiliser pour les tests de détection")
        exit(1)

    face_cascade = cv2.CascadeClassifier("../haarcascades/haarcascade_frontalface_default.xml")
    # get the image information
    fd = open("../dataset/FDDB_dataset/FDDB-folds/FDDB-fold-" + file_number_used + "-ellipseList.txt")
    img_info_vec = []
    for k in range(nb_images_testing):
        img_info_vec.append(parse_file.get_img_info(fd))
    fd.close()

    """ ----------------- Phase 1 : Face detection ------------------- """
    if debug:
        print("Face detection...")
    first_scale_factor = 1.05
    last_scale_factor = 2.0
    scale_factor_step = 0.05
    first_min_neighbours = 0
    last_min_neighbours = 11
    min_neighbours_step = 1
    X = np.arange(1.05,2.0,0.05)
    Y = np.arange(0,11,1)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X**2 + Y**2)
    Z = np.sin(R)
    print(type(Z))
    print(X.shape)
    print(Y.shape)
    plt_handle_vec = []
    precision_x = []
    recall_x = []
    precision_y = []
    recall_y = []
    for min_neighbours in range(0,11,1):
        for scale_factor in np.arange(1.05,2.0,0.05):
            print("scale_factor: "+str(scale_factor)+", min_neighbours: "+str(min_neighbours))
            y_true = []
            levelWeights_all=[]
            counter = 0
            FN = 0
            for img_info in img_info_vec:
                counter += 1
                #print("img nb: "+str(counter))
                # viola-jones works with grayscale images
                gray_img = cv2.imread(img_info.img_path,0) # 0 = IMREAD_GRAYSCALE
                detected_faces = face_cascade.detectMultiScale(gray_img,scale_factor,min_neighbours)
                img_info.img_shape = gray_img.shape
                y_true_tmp = single_image_evaluation.evaluate(img_info,detected_faces)
                y_true.append(y_true_tmp)
                FN += len(img_info.list_ellipse)

            y_true = np.concatenate(y_true)
            TP = y_true.sum()
            FN = FN-TP
            FP = len(y_true)-TP
            P = 0
            R = 0
            if(not (TP==0)):
                P = float(TP)/float(TP+FP)
                R = float(TP)/float(TP+FN)
            precision_y.append(P)
            recall_y.append(R)
        precision_x.append(precision_y)
        recall_x.append(recall_y)
        precision_y = []
        recall_y = []

    recall_x = np.matrix(recall_x)
    precision_x = np.matrix(precision_x)
    print(precision_x.shape)
    print(recall_x.shape)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.text2D(0.05, 0.95, "Recall", transform=ax.transAxes)
    ax.set_ylabel("minNeighbours")
    ax.set_xlabel("scaleFactor")
    surf_recall = ax.plot_surface(X, Y, recall_x, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    fig.colorbar(surf_recall, shrink=0.5, aspect=5)

    plt.show()
    fig.set_size_inches((16, 10), forward=False)
    #fig.savefig("figures/recall_scale.png",dpi=600)

    plt.gcf().clear()
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.text2D(0.05, 0.95, "Precision", transform=ax.transAxes)
    ax.set_ylabel("minNeighbours")
    ax.set_xlabel("scaleFactor")
    surf_precision = ax.plot_surface(X, Y, precision_x, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    fig.colorbar(surf_precision, shrink=0.5, aspect=5)

    plt.show()
    fig.set_size_inches((16, 10), forward=False)
    #fig.savefig("figures/precision_scale.png",dpi=600)
