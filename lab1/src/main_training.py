import sys
import numpy as np
import calc_histo
import lookup_table
import parse_file
from matplotlib import pyplot as plt
import pickle
import graphical_tools

# Number of images for each ellipse file FDDB, got from the script img_counter.sh
nb_img_by_fold = [0, 290, 285, 274, 302, 298, 302, 279, 276, 259, 280]

""" Initialization : Choice of the parameters """
try:
    first_file_used = sys.argv[1]
    last_file_used = sys.argv[2]
    n_quantification = int(sys.argv[3])
    color_mode = str(sys.argv[4])

    lookup_table_color_mode = calc_histo.Color.RGB
    if(color_mode == "RG"):
        lookup_table_color_mode = calc_histo.Color.RG
    else:
        color_mode = "RGB"

except IndexError as err:
    print("IndexError: {0}".format(err))
    print("\nAppel de la fonction :")
    print("python3 main_training.py int1 int2 int3 str4, avec :")
    print("    int1...int2 : intervalle des fichiers ellipseList à utiliser, entre 1 et 10")
    print("    int3 : the color sample rate, f.ex.1, 8")
    print("    str4 : the color mode, \"RGB\" or \"RG\"")
    exit(1)

""" --- Phase 1 : Training : Skin pixels detection with color ---  """
""" Construction of the lookup table """
# Parameters : RGB or Chrominance, Quantification N, choice of the training Data 

print("Construction of the lookup table...")
lookup_table_data = lookup_table.LookupTable(lookup_table_color_mode, n_quantification)

for file_nb in range(int(first_file_used), int(last_file_used)+1):

    if file_nb != 10:
        filename_nb = "0" + str(file_nb)
    else:
        filename_nb = str(file_nb)

    fd = open("../dataset/FDDB_dataset/FDDB-folds/FDDB-fold-" + filename_nb + "-ellipseList.txt")
    lookup_table_data.calc_fold_histograms(fd, nb_img_by_fold[file_nb])
    fd.close()

lookup_table_data.construct_lookup_table()

print("Saving lookup table in file ...", end = " ", flush = True)
try:
    lookup_table_fd = open("../lookup_tables/lT_" + first_file_used + "_" + last_file_used + "_" + str(n_quantification) + "_" + color_mode, "wb")
    pickle.dump(lookup_table_data, lookup_table_fd)
    lookup_table_fd.close() 
    print("done.")
except IndexError as err:
    print("IndexError: {0}".format(err))
    exit(1)

print("done.")

lookup_table_data.plot("lookup table")
graphical_tools.plot_3d_color_histogram(lookup_table_data, n_quantification)

