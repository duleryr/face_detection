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

    scale_factor=1.1 # later we will run through these parameters

    plt_handle_vec = []
    for scale_factor in np.arange(1.01,1.06,0.01):
        for min_neighbours in range(min_neighbours,min_neighbours+1,1):
            print("scale_factor: "+str(scale_factor)+", min_neighbours: "+str(min_neighbours))
            y_true = []
            levelWeights_all=[]
            counter = 0
            FN = 0
            for img_info in img_info_vec:
                counter += 1
                print("img nb: "+str(counter))
                # viola-jones works with grayscale images
                gray_img = cv2.imread(img_info.img_path,0) # 0 = IMREAD_GRAYSCALE
                [detected_faces,rejectLevels,levelWeights] = face_cascade.detectMultiScale3(gray_img,scale_factor,min_neighbours,outputRejectLevels=True)
                img_info.img_shape = gray_img.shape
                y_true_tmp = single_image_evaluation.evaluate(img_info,detected_faces)
                y_true.append(y_true_tmp)
                FN += len(img_info.list_ellipse)
                if(len(levelWeights)>0):
                    levelWeights = np.concatenate(levelWeights)
                    levelWeights_all.append(levelWeights)

            y_true = np.concatenate(y_true)
            levelWeights_all = np.concatenate(levelWeights_all)
            FN = FN-y_true.sum()
            weight_tp_fp_vec = zip(levelWeights_all,y_true)
            levelWeights_all,y_true = zip(*sorted(weight_tp_fp_vec))
#            precision,recall, thresholds = precision_recall_curve(y_true,levelWeights_all,1)
            precision = []
            recall = []
            TP = 0
            FP = 0
            P = 0
            R = 0
            y_true = np.array(y_true)
#            print(y_true)
#            print(levelWeights_all)
            new_TP = [y_true.sum()]
            new_FP = [len(y_true)-y_true.sum()]
            new_FN = [FN]
            # ============= STRATEGIE GROUPE 2 ================
            # ici on fait augmenter le seuil
#            print("TP : "+str(new_TP[0])+", FP: "+str(new_FP[0])+", FN: "+str(new_FN[0]))
            for i in range(len(y_true)):
                # DEBUG
                if(y_true[i]==1):
                    new_TP.append(new_TP[i]-1)
                    new_FN.append(new_FN[i]+1)
                    new_FP.append(new_FP[i])
                else:
                    new_FP.append(new_FP[i]-1)
                    new_TP.append(new_TP[i])
                    new_FN.append(new_FN[i])
                # DEBUG
#                print("STEP "+str(i))
#                print("TP : "+str(new_TP[i])+", FP: "+str(new_FP[i])+", FN: "+str(new_FN[i]))
#                P = float(new_TP[i])/float(new_TP[i]+new_FP[i])
#                R = float(new_TP[i])/float(new_TP[i]+new_FN[i])
#                print("Precision: "+str(P)+", Recall: "+str(R))
#                print("")
            # compute precision, recall
            precision.append(0)
            recall.append(1)
            for i in range(len(new_TP)):
                if((new_TP[i]+new_FP[i])!=0):
                    P = float(new_TP[i])/float(new_TP[i]+new_FP[i])
                    R = float(new_TP[i])/float(new_TP[i]+new_FN[i])
                    precision.append(P)
                    recall.append(R)
            precision.append(1)
            recall.append(0)
#            print(precision)
#            print(recall)
            #print(auc(recall,precision))
                
            # ========== OLD STRATEGY ===============
#            for i in range(len(y_true)-1,0,-1):
#                if(y_true[i]==1):
#                    TP += 1
#                    #FN -= 1
#                else:
#                    FP += 1
#                print("TP: "+str(TP)+", FP: "+str(FP)+", FN: "+str(FN))
#                P = float(TP)/float(TP+FP)
#                R = float(TP)/float(FN)
#                precision.append(P)
#                recall.append(R)
#            print(precision)
#            print(recall)

#    plt.gcf().clear()
#    plt.plot(recall,precision)
            plt.step(recall, precision, where='post', label="scaleFactor: "+str(scale_factor)+", AUC: "+str(np.around(auc(recall,precision),decimals=5)))
#    plt.title("scaleFactor: "+str(scale_factor)+", minNeighbours: "+str(min_neighbours))
    plt.title("minNeighbours: "+str(min_neighbours))
    plt.axis([0,1.1,0,1.1])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    fig = plt.gcf()
    fig.set_size_inches((16, 10), forward=False)
    fig.savefig("figures/folder_"+str(file_number_used)+"_minN_"+str(min_neighbours)+".png",dpi=600)
    #plt.show()
