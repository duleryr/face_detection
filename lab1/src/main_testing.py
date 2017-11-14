import sys
import os
import numpy as np
from sklearn.metrics import auc
import calc_histo
import lookup_table
import parse_file
import cv2
from matplotlib import pyplot as plt
from face_detection import get_statistics_one_image
import pickle
import graphical_tools
import multiprocessing
import time

if __name__ == '__main__':
    """ Initialization : Choice of the files used for the training/testing """
    try:
        file_number_used = sys.argv[1]
        if (len(file_number_used) == 1):
            file_number_used = "0" + file_number_used
        nb_images_testing = int(sys.argv[2])
        lookup_table_file = sys.argv[3]
        debug = False
    except IndexError as err:
        print("IndexError: {0}".format(err))
        print("\nAppel de la fonction :")
        print("python3 int1 int2 str3, avec :")
        print("    int1 : numéro du fichier ellipseList à utiliser, entre 1 et 10")
        print("    int2 : nombre d'images à utiliser pour les tests de détection")
        print("    str3 : la lookup table à utiliser")
        exit(1)
    fd = open("../dataset/FDDB_dataset/FDDB-folds/FDDB-fold-" + file_number_used + "-ellipseList.txt")
    
    parameters = lookup_table_file.split("_")
    n_quantification = int(parameters[3])
    lookup_table_color_mode = parameters[4]
    
    """ --- Phase 1 : Training : Skin pixels detection with color ---  """
    """ Loading the lookup table """
    # Parameters : RGB or Chrominance, Quantification N, choice of the training Data 
    
    lookup_table_data = lookup_table.LookupTable(lookup_table_color_mode, n_quantification)
    try:
        lookup_table_fd = open(lookup_table_file, "rb")
        lookup_table_data = pickle.load(lookup_table_fd)
        lookup_table_fd.close() 
    except IndexError as err:
        print("IndexError: {0}".format(err))
        exit(1)
    
    if debug:
        lookup_table_data.plot("lookup table")
        graphical_tools.plot_3d_color_histogram(lookup_table_data, n_quantification)
    
    """ ----------------- Phase 2 : Face detection ------------------- """
    """ Use of sliding window : ROI """
    # Parameters : Width/Height of ROI, Scanning pattern, Decision algorithm of detection, Use of gaussian mask
    
    if debug:
        print("Face detection...")
    
    bias_vec = np.arange(-0.5, 0.501, 0.05)
    
    # Per-image operations
    tpr_vec, fpr_vec, tp_vec, fp_vec, tn_vec, fn_vec = ([0]*len(bias_vec) for i in range(6))
    
    #Parallel processing
    img_info_vec = []
    for k in range(nb_images_testing):
        img_info_vec.append(parse_file.get_img_info(fd))
    
    fd.close()
    results = multiprocessing.Array('i', 4*len(bias_vec), lock=False)
    
    def parallel_img(pid):
        #if (debug):
        print(" "+str(pid)+"-th image")
        img_info_local = img_info_vec[pid]
        img = cv2.imread(img_info_local.img_path)
        #print("  bias = ", end = "", flush = True)
        for i,bias in enumerate(bias_vec):
            #print(bias, end = ", ", flush = True)
            s = get_statistics_one_image(lookup_table_data,img,img_info_local,bias,4,4,9,9, n_quantification)
            results[i*4+0] += s.tp
            results[i*4+1] += s.fp
            results[i*4+2] += s.tn
            results[i*4+3] += s.fn
        #print("done.")
        return 0
    
    p = multiprocessing.Pool(multiprocessing.cpu_count())
    #p = multiprocessing.Pool(4)
    
    start_time = time.time()
    p.map(parallel_img, range(nb_images_testing))
    end_time = time.time()
    #if debug:
    #print("Parallel time=", end_time - start_time)
    #for i in results:
    #    print(i)
    
    #transfer results to stats arrays
    for i in range(len(bias_vec)):
        tp_vec[i] = results[i*4+0]
        fp_vec[i] = results[i*4+1]
        tn_vec[i] = results[i*4+2]
        fn_vec[i] = results[i*4+3]
    
    # Serialized processing
    #start_time = time.time()
    #for k in range(nb_images_testing):
    #    print(" "+str(k)+"-th image")
    #
    #    img_info = parse_file.get_img_info(fd)
    #    img = cv2.imread(img_info.img_path)
    #
    #    print("  bias = ", end = "", flush = True)
    #    for i,bias in enumerate(bias_vec):
    #        print(bias, end = ", ", flush = True)
    #        s = get_statistics_one_image(lookup_table_data,img,img_info,bias,4,4,9,9, n_quantification)
    #        tp_vec[i] += s.tp
    #        fp_vec[i] += s.fp
    #        tn_vec[i] += s.tn
    #        fn_vec[i] += s.fn
    #    print("done.")
    #end_time = time.time()
    #print("Serial time=", end_time - start_time)
    
    # calculating the tpr, fpr
    for i in range(len(bias_vec)):
        tpr_vec[i] = float(tp_vec[i])/float(tp_vec[i]+fn_vec[i])
        fpr_vec[i] = float(fp_vec[i])/float(fp_vec[i]+tn_vec[i])
    
    """ ---------- Performance evaluation of the detection ----------- """
    """ Plot of ROC curves """
    # Parameters : Way to compare ROC curves : Area under the curve, FNR*FPR, FNR+FPR
    
    area_under_curve = auc(fpr_vec, tpr_vec)
    plt.plot(fpr_vec, tpr_vec)
    if debug:
        print("  tpr : " + str(tpr_vec))
        print("  fpr : " + str(fpr_vec))
        print("  Area under curve : " + str(area_under_curve))
        plt.show()
    else:
        destination = os.getcwd()+"/../ROC_curves/ROC_" + parameters[1] + "_" + parameters[2] + "_" + str(file_number_used) + "_" + str(nb_images_testing) + "_" + str(lookup_table_color_mode) + ".png"
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("AUC = "+str(area_under_curve))
        plt.savefig(destination)
        print(destination + " : AUC = " + str(area_under_curve))
    
    """ --------------- Phase 3 : Face localisation ------------------ """
    """ Expectation Maximization, K-means """
    # Parameters : Number of faces in the image, Range of position/size/orientation, clustering regions sizes, distance for non-maximal suppression
