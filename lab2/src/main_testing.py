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

if __name__ == '__main__':
    """ Initialization : Choice of the files used for the training/testing """
    try:
        file_number_used = sys.argv[1]
        if (len(file_number_used) == 1):
            file_number_used = "0" + file_number_used
        nb_images_testing = int(sys.argv[2])
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

    # DEBUG
    #for img_info in img_info_vec:
    #    graphical_tools.showImg("debug",cv2.imread(img_info.img_path))

    """ ----------------- Phase 1 : Face detection ------------------- """
    if debug:
        print("Face detection...")

    scale_factor=1.2 # later we will run through these parameters
    min_neighbours=3

    for img_info in img_info_vec:
        gray_img = cv2.imread(img_info.img_path,0)
        graphical_tools.showImg("grayscale image", gray_img)
        faces = face_cascade.detectMultiScale(gray_img,scale_factor,min_neighbours)
        graphical_tools.showFaces(gray_img,faces)

    
    """ ---------- Performance evaluation of the detection ----------- """
    """ Plot of ROC curves """
#    # Parameters : Way to compare ROC curves : Area under the curve, FNR*FPR, FNR+FPR
#    
#    area_under_curve = auc(fpr_vec, tpr_vec)
#    plt.plot(fpr_vec, tpr_vec)
#    if debug:
#        print("  tpr : " + str(tpr_vec))
#        print("  fpr : " + str(fpr_vec))
#        print("  Area under curve : " + str(area_under_curve))
#        plt.show()
#    else:
#        destination = os.getcwd()+"/../ROC_curves/ROC_" + parameters[1] + "_" + parameters[2] + "_" + str(file_number_used) + "_" + str(nb_images_testing) + "_" + str(lookup_table_color_mode) + ".png"
#        plt.xlabel("FPR")
#        plt.ylabel("TPR")
#        plt.title("AUC = "+str(area_under_curve))
#        plt.savefig(destination)
#        print(destination + " : AUC = " + str(area_under_curve))
