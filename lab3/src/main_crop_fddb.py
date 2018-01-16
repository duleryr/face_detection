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
import cnn
import tensorflow as tf
import random
import fddb_crop

# Hyper-parameters

# Load FDDB data
fddb_crop = fddb_crop.Manager()
fddb_crop.set_folders([1,5])
fddb_crop.set_fddb_dir("../dataset")
fddb_crop.load_img_descriptors()
#fddb_crop.crop_images_noise("../crops/positives","../crops/negatives")
fddb_crop.crop_images("../crops/positives_small","../crops/negatives_small")

# load images
#fddb_crop.load_images()
