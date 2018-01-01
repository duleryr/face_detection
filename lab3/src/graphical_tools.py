#!/usr/bin/env python

import cv2
#import pyqtgraph.examples
import numpy as np
#import pyqtgraph.opengl as gl
#from pyqtgraph.Qt import QtCore, QtGui
import sys
import roi

#define _ITERATOR_DEBUG_LEVEL=0

def showImg(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(100)
    cv2.destroyAllWindows()

def showROI(img, roi):
    img_copy = img
    cv2.rectangle(img_copy,(roi.c[0]-roi.half_window_size,roi.c[1]-roi.half_window_size),
        (roi.c[0]+roi.half_window_size,roi.c[1]+roi.half_window_size),(255,0,0),2)
    cv2.imshow("ROI", img_copy)
    cv2.waitKey(100)
    del(img_copy)
    del(img)
    cv2.destroyAllWindows()

def showFaces(img, faces):
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.namedWindow("face_detections", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("face_detections", img.shape[1], img.shape[0])
    if(img.shape[0]>2000 or img.shape[1]>2000):
        print(img.shape)
        cv2.resizeWindow("face_detections", 800, 600)
    cv2.imshow("face_detections", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def save_snapshot(file_name,img):
    cv2.imwrite(file_name,img)

def showTPFPFN(img_info, true_detections, false_detections):
    img = cv2.imread(img_info.img_path)
    # TP
    for(x,y,w,h) in true_detections:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    # FP
    for(x,y,w,h) in false_detections:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
    # FN
    for e in img_info.list_ellipse:
        print(int(e.c_x))
        pt1 = (int(e.c_x-e.r_b),int(e.c_y-e.r_a))
        pt2 = (int(e.c_x+e.r_b),int(e.c_y+e.r_a))
        cv2.rectangle(img,pt1,pt2,(0,0,255),2)
        #cv2.ellipse(img,(int(e.c_x),int(e.c_y)),(int(e.r_a), int(e.r_b)),
        #    int(e.theta),0,360,(0,0,255), 2)
    cv2.namedWindow("face_detections", cv2.WINDOW_NORMAL)
    #cv2.resizeWindow("face_detections", img.shape[1], img.shape[0])
    cv2.resizeWindow("face_detections", 1000, 1000)
    if(img.shape[0]>2000 or img.shape[1]>2000):
        print(img.shape)
        cv2.resizeWindow("face_detections", 800, 600)
    #save_snapshot('result.jpg',img)
    cv2.imshow("face_detections", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#def plot_3d_color_histogram(lookup_table, n_quantification):
#    #pyqtgraph.examples.run()
#    color_value = int(256/n_quantification)
#    app = QtGui.QApplication([])
#    w = gl.GLViewWidget()
#    w.opts['distance'] = 50
#    w.show()
#    w.setWindowTitle('pyqtgraph example: GLScatterPlotItem')
#    
#    #g = gl.GLGridItem()
#    #w.addItem(g)
#
#    pos = np.empty((lookup_table.table.size, 3))
#    size = np.empty((lookup_table.table.size))
#    color = np.empty((lookup_table.table.size, 4))
#    index = 0
#    if(lookup_table.mode == calc_histo.Color.RGB):
#        for i in range(lookup_table.table.shape[0]):
#            for j in range(lookup_table.table.shape[1]):
#                for k in range(lookup_table.table.shape[2]):
#                    pos[index] = (i,j,k)
#                    color[index] = (i*n_quantification/256,
#                        j*n_quantification/256,k*n_quantification/256,1.0)
#                    size[index] = lookup_table.table[i][j][k]
#                    index += 1
#    elif(lookup_table.mode == calc_histo.Color.RG):
#        for i in range(lookup_table.table.shape[0]):
#            for j in range(lookup_table.table.shape[1]):
#                pos[index] = (i,j,0)
#                color[index] = (i*n_quantification/256,
#                        j*n_quantification/256,0,1.0)
#                size[index] = lookup_table.table[i][j]
#                index += 1
#    sp1 = gl.GLScatterPlotItem(pos=pos, size=size, color=color, pxMode=False)
#    sp1.translate(-128/n_quantification,-128/n_quantification,-128/n_quantification)
#    w.addItem(sp1)
#
#    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
#        QtGui.QApplication.instance().exec_()
