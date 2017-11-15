import sys
import lookup_table
import pickle
import graphical_tools

if __name__ == '__main__':
    try:
        lookup_table_file = sys.argv[1]
    except IndexError as err:
        print("IndexError: {0}".format(err))
        exit(1)

    parameters = lookup_table_file.split("_")
    n_quantification = int(parameters[3])
    lookup_table_color_mode = parameters[4]
    lookup_table_data = lookup_table.LookupTable(lookup_table_color_mode, n_quantification)
    try:
        lookup_table_fd = open(lookup_table_file, "rb")
        lookup_table_data = pickle.load(lookup_table_fd)
        lookup_table_fd.close() 
    except IndexError as err:
        print("IndexError: {0}".format(err))
        exit(1)
    
    lookup_table_data.plot("lookup table")
    graphical_tools.plot_3d_color_histogram(lookup_table_data, n_quantification)

