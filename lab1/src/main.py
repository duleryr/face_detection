import sys
import numpy as np
from sklearn.metrics import auc
import lookup_table
import parse_file
import cv2
from matplotlib import pyplot as plt
from face_detection import get_statistics_one_image
import pickle
import graphical_tools

""" Initialization : Choice of the files used for the training/testing """
try:
    file_used = sys.argv[1]
    fd = open("../dataset/FDDB_dataset/FDDB-folds/FDDB-fold-" + file_used + "-ellipseList.txt")
    nb_images_training = int(sys.argv[2])
    nb_images_testing = int(sys.argv[3])
    charge_lookup_table = int(sys.argv[4])
    n_quantification = int(sys.argv[5])
    if (charge_lookup_table):
        lookup_table_file = sys.argv[6]

except IndexError as err:
    print("IndexError: {0}".format(err))
    print("\nAppel de la fonction :")
    print("python3 int1 int2 int3 int4, avec :")
    print("    int1 : numéro du fichier ellipseList à utiliser, entre 1 et 10")
    print("    int2 : nombre d'images à utiliser pour l'entrainement")
    print("    int3 : nombre d'images à utiliser pour les tests de détection")
    print("    int4 : 0 to rebuild the lookup table, 1 to charge it from file \"lookup_table\" ")
    print("    int5 : the color sample rate, f.ex.1, 8")
    print("    str6 : if loading the lookup table (int4 == 1), the lookup table file to use")
    exit(1)

""" --- Phase 1 : Training : Skin pixels detection with color ---  """
""" Construction of the lookup table """
# Parameters : RGB or Chrominance, Quantification N, choice of the training Data 

lookup_table_data = []
if (charge_lookup_table==0):
    print("Construction of the lookup table...", end = " ", flush = True)
    print("")
    lookup_table_data = lookup_table.construct_lookup_table(fd, nb_images_training, n_quantification)
    print("done.")
    print("Saving lookup table in file lT ...", end = " ", flush = True)
    print("")
    try:
        lookup_table_fd = open("lT_" + file_used + "_" + str(nb_images_training), "wb")
        pickle.dump(lookup_table_data, lookup_table_fd)
        lookup_table_fd.close() 
        print("done.")
    except IndexError as err:
        print("IndexError: {0}".format(err))
        exit(1)
if (charge_lookup_table==1):
    try:
        lookup_table_fd = open(lookup_table_file, "rb")
        lookup_table_data = pickle.load(lookup_table_fd)
        lookup_table_fd.close() 
    except IndexError as err:
        print("IndexError: {0}".format(err))
        exit(1)

lookup_table.plot_array(lookup_table_data,"lookup table")
graphical_tools.plot_3d_color_histogram(lookup_table_data, n_quantification)

""" ----------------- Phase 2 : Face detection ------------------- """
""" Use of sliding window : ROI """
# Parameters : Width/Height of ROI, Scanning pattern, Decision algorithm of detection, Use of gaussian mask

print("Face detection...")

bias_vec = np.arange(-0.5, 0.6, 0.05)

# per-image operations
tpr_vec, fpr_vec, tp_vec, fp_vec, tn_vec, fn_vec = ([0]*len(bias_vec) for i in range(6))

for k in range(nb_images_testing):
    print(" "+str(k)+"-th image")

    img_info = parse_file.get_img_info(fd)
    img = cv2.imread(img_info.img_path)

    print("  bias = ", end = "", flush = True)
    for i,bias in enumerate(bias_vec):
        print(bias, end = ", ", flush = True)
        s = get_statistics_one_image(lookup_table_data,img,img_info,bias,30,30,61,61, n_quantification)
        tp_vec[i] += s.tp
        fp_vec[i] += s.fp
        tn_vec[i] += s.tn
        fn_vec[i] += s.fn
    print("done.")

# calculating the tpr, fpr
for i in range(len(bias_vec)):
    tpr_vec[i] = float(tp_vec[i])/float(tp_vec[i]+fn_vec[i])
    fpr_vec[i] = float(fp_vec[i])/float(fp_vec[i]+tn_vec[i])

print("  tpr : " + str(tpr_vec))
print("  fpr : " + str(fpr_vec))
area_under_curve = auc(fpr_vec, tpr_vec)
print("  Area under curve : " + str(area_under_curve))
plt.plot(fpr_vec, tpr_vec)
plt.show()

""" ---------- Performance evaluation of the detection ----------- """
""" Plot of ROC curves """
# Parameters : Way to compare ROC curves : Area under the curve, FNR*FPR, FNR+FPR

""" --------------- Phase 3 : Face localisation ------------------ """
""" Expectation Maximization, K-means """
# Parameters : Number of faces in the image, Range of position/size/orientation, clustering regions sizes, distance for non-maximal suppression

fd.close()
