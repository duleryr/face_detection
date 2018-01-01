import parse_file
import roi
import graphical_tools
import numpy as np
import cv2

nb_images_per_folder = [290,285,274,302,298,302,279,276,259,280]

class Manager:
    def __init__(self):
        self.train_folders_vec = []
        self.test_folders_vec = []
        self.train_img_info_vec = []
        self.test_img_info_vec = []
        self.fddb_dir = "../dataset/"
        self.train_img_counter = 0
        self.test_img_counter = 0
        self.train_batch_counter = 0
        self.test_batch_counter = 0
        self.window_size = 11
        self.train_roi = roi.ROI(self.window_size)
        self.test_roi = roi.ROI(self.window_size)
    
    def set_window_size(self, window_size_input):
        self.window_size = window_size_input
        self.train_roi = roi.ROI(self.window_size)
        self.test_roi = roi.ROI(self.window_size)

    def set_train_folders(self, train_folders_vec_input):
        self.train_folders_vec = train_folders_vec_input

    def set_test_folders(self, test_folders_vec_input):
        self.test_folders_vec = test_folders_vec_input

    def set_fddb_dir(self, fddb_dir_input):
        self.fddb_dir = fddb_dir_input


    def load_img_descriptors_aux(self,folders_vec,img_info_vec):
        for t in folders_vec:
            ellipse_path = "."
            if(t<10):
                ellipse_path = self.fddb_dir+"/FDDB_dataset/FDDB-folds/FDDB-fold-0"+str(t)+"-ellipseList.txt"
            else:
                ellipse_path = self.fddb_dir+"/FDDB_dataset/FDDB-folds/FDDB-fold-"+str(t)+"-ellipseList.txt"
            fd = open(ellipse_path)
            for i in range(nb_images_per_folder[t-1]):
                img_info_vec.append(parse_file.get_img_info(fd))
            fd.close()

    def load_img_descriptors(self):
        self.load_img_descriptors_aux(self.train_folders_vec,self.train_img_info_vec)
        self.load_img_descriptors_aux(self.test_folders_vec,self.test_img_info_vec)

    # img_counter and batch_counter help us to keep track of where we stopped 
    # last time we called this function
    def next_batch_aux(self, batch_size, img_info_vec, img_counter, batch_counter, roi):
        batch = []
        if(img_counter>len(img_info_vec)):
            return batch
        img = cv2.imread(img_info_vec[img_counter].img_path)
        graphical_tools.showImg("test",img)
        for i in range(batch_size):
            # move the roi forward
            batch.append(roi.get_roi_content(img))
            roi.next_step(img.shape)
            if(roi.c[0] == -1): # roi is out of bounds
                roi.reset_pos()
                img_counter += 1 # pass on to the next image
                if(img_counter>len(img_info_vec)):
                    return batch
                img = cv2.imread(img_info_vec[img_counter].img_path)
        return batch

    def next_batch_train(self, batch_size):
        return self.next_batch_aux(batch_size,self.train_img_info_vec,self.train_img_counter,self.train_batch_counter,self.train_roi)

    def next_batch_test(self, batch_size):
        return self.next_batch_aux(batch_size,self.test_img_info_vec,self.test_img_counter,self.test_batch_counter,self.test_roi)
