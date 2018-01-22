#   import facenet libraires
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import tensorflow as tf
import os
import align.detect_face

import fddb_manager
import graphical_tools

#  import other libraries
import cv2
import matplotlib.pyplot as plt

#   setup facenet parameters
gpu_memory_fraction = 1.0
minsize = 50 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor

#   Start code from facenet/src/compare.py
print('Creating networks and loading parameters')
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=gpu_memory_fraction)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,
        log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = align.detect_face.create_mtcnn(
            sess, None)
#   end code from facenet/src/compare.py

    N = 31
    # Load FDDB data
    fddb = fddb_manager.Manager()
    fddb.set_train_folders([1])
    fddb.set_test_folders([10])
    fddb.set_fddb_dir("../dataset")
    fddb.load_img_descriptors()
    fddb.set_window_size(N)

    batch = fddb.next_batch_faceDetect(fddb.test_img_info_vec, 1)

    TP = 0
    FP = 0
    FN = 0

    # For each testing image
    for i in range(len(batch[0])):

        #   run detect_face from the facenet library
        bounding_boxes, _ = align.detect_face.detect_face(batch[0][i], minsize, pnet,
                rnet, onet, threshold, factor)

        #   for each box
        for (x1, y1, x2, y2, acc) in bounding_boxes:
            w = x2-x1
            h = y2-y1
            #   plot the box using cv2
            cv2.rectangle(batch[0][i],(int(x1),int(y1)),(int(x1+w),
                int(y1+h)),(255,0,0),2)
            bb_center = [int((y1+y2)/2), int((x1+x2)/2)]
            print(batch[1][i][bb_center[0]][bb_center[1]][0])
            if batch[1][i][bb_center[0]][bb_center[1]][0] == 255:
                TP += 1
            else:
                plt.figure()
                plt.imshow(batch[0][i])
                plt.show()
                FP += 1
            print ('Accuracy score', acc)
        #   save a new file with the boxed face
        #cv2.imsave('faceBoxed'+i, img)
        #   show the boxed face

        #plt.figure()
        #plt.imshow(batch[0][i])
        #plt.show()
    print(TP)
    print(FP)
