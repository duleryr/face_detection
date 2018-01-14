import os
import parse_file
import roi
import graphical_tools
import numpy as np
import cv2
import random
import wider_loader

nb_images_per_folder = [290,285,274,302,298,302,279,276,259,280]
FACE_PATH = "/home/felix/Documents/cours3A/reconnaissance_formes/pattern_recognition/lab3/crops/positives/"
NON_FACE_PATH = "/home/felix/Documents/cours3A/reconnaissance_formes/pattern_recognition/lab3/crops/negatives/"
#FACE_PATH = "/home/felix/Documents/cours3A/reconnaissance_formes/pattern_recognition/lab3/crops/positives_face_finder/"
#NON_FACE_PATH = "/home/felix/Documents/cours3A/reconnaissance_formes/pattern_recognition/lab3/crops/negatives_face_finder/"
#IN_SIZE = (32,32)  #face finder 
IN_SIZE = (31,31)
POS = [1,0]
NEG = [0,1]

class Manager:
    def __init__(self):
        self.folders_vec = []
        self.img_info_vec = []
        self.fddb_dir = "../dataset/"
        # img_counter_vec holds the image counters for the train and the test set
        self.img_counter = 0
        self.window_size = 32
        self.roi = roi.ROI(self.window_size)
        self.batch = []
    
    def set_window_size(self, window_size_input):
        self.window_size = window_size_input
        self.roi = roi.ROI(self.window_size)

    def get_WIDER_img_info(self, bbx_file_name):
        wider_vec = wider_loader.get_all_img_info(bbx_file_name)
        self.img_info_vec.extend(wider_vec)

    def set_folders(self, folders_vec_input):
        self.folders_vec = folders_vec_input

    def set_fddb_dir(self, fddb_dir_input):
        self.fddb_dir = fddb_dir_input


    def load_img_descriptors(self):
        for t in self.folders_vec:
            ellipse_path = "."
            if(t<10):
                ellipse_path = self.fddb_dir+"/FDDB_dataset/FDDB-folds/FDDB-fold-0"+str(t)+"-ellipseList.txt"
            else:
                ellipse_path = self.fddb_dir+"/FDDB_dataset/FDDB-folds/FDDB-fold-"+str(t)+"-ellipseList.txt"
            fd = open(ellipse_path)
            for i in range(nb_images_per_folder[t-1]):
                self.img_info_vec.append(parse_file.get_img_info(fd))
            fd.close()

    # last time we called this function
    def crop_images(self, positive_path, negative_path):
        #counter = 0
        crop_counter = 0
        for img_info in self.img_info_vec:
            print(img_info.img_path)
            img = cv2.imread(img_info.img_path,0)
            ground_truth = graphical_tools.calc_mask(img,img_info)
            while(self.roi.c[0] != -1): # roi is out of bounds
                is_in_face = ground_truth[self.roi.c[0],self.roi.c[1]]>0 # BW
                file_name = ""
                if is_in_face:
                    file_name = positive_path+"/"+str(crop_counter)+".png"
                else:
                    file_name = negative_path+"/"+str(crop_counter)+".png"
                cv2.imwrite(file_name,self.roi.get_roi_content(img))
                self.roi.next_step(img.shape)
                crop_counter += 1
            self.roi.reset_pos()
            #counter += 1
            #if counter > 0:
            #    break
    def load_images(self):
        batch = []
        #Load faces (positive samples)
        for n in os.listdir(FACE_PATH):
            name = FACE_PATH+n
            img_path = name
            img = cv2.imread(img_path,0)
            t_img = cv2.resize(img,IN_SIZE)
            batch.append((t_img, POS))
            batch.append((cv2.flip(t_img,1),POS)) #Use the horizontal mirror image
        print(len(batch))
        nbatch = []
        #Load faces (positive samples)
        for n in os.listdir(NON_FACE_PATH):
            name = NON_FACE_PATH+n
            img_path = name
            img = cv2.imread(img_path,0)
            t_img = cv2.resize(img,IN_SIZE)
            nbatch.append((t_img, NEG))
            #nbatch.append((cv2.flip(t_img,1),NEG)) #Use the horizontal mirror image
        print(len(nbatch))
        batch.extend(nbatch)
        random.shuffle(batch)
        print(len(batch))
        reshaped_batch = [[],[]]
        for b in batch:
            reshaped_batch[0].append(b[0])
            reshaped_batch[1].append(b[1])
        print(len(reshaped_batch))
        self.batch = reshaped_batch

    def reset_img_counter(self):
        self.img_counter = 0

    def next_batch_train(self, n):
        batch =[[],[]]
        full_batch = True
        for i in range(self.img_counter, self.img_counter+n):
            if(i > len(self.batch[0])):
                full_batch = False
                break # no data left
            batch[0].append(self.batch[0][i])
            batch[1].append(self.batch[1][i])
        self.img_counter += n
        return batch, full_batch
            
