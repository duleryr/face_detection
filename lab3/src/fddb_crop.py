import os
import parse_file
import roi
import graphical_tools
import numpy as np
import cv2
import random
import wider_loader
from scipy import ndimage
from sklearn.decomposition import PCA

nb_images_per_folder = [290,285,274,302,298,302,279,276,259,280]
#FACE_PATH = "/home/emily/Documents/Ensimag/3A/Pattern_Recognition/pattern_recognition/lab3/crops/positives/"
#NON_FACE_PATH = "/home/emily/Documents/Ensimag/3A/Pattern_Recognition/pattern_recognition/lab3/crops/negatives/"
#FACE_PATH = "/home/felix/Documents/cours3A/reconnaissance_formes/second_git/pattern_recognition/lab3/crops/positives/"
#NON_FACE_PATH = "/home/felix/Documents/cours3A/reconnaissance_formes/second_git/pattern_recognition/lab3/crops/negatives/"
FACE_PATH_TRAIN = "/home/felix/Documents/cours3A/reconnaissance_formes/second_git/pattern_recognition/lab3/crops/faces_train/"
NON_FACE_PATH_TRAIN = "/home/felix/Documents/cours3A/reconnaissance_formes/second_git/pattern_recognition/lab3/crops/non_faces_train/"
FACE_PATH_TEST = "/home/felix/Documents/cours3A/reconnaissance_formes/second_git/pattern_recognition/lab3/crops/faces_test/"
NON_FACE_PATH_TEST = "/home/felix/Documents/cours3A/reconnaissance_formes/second_git/pattern_recognition/lab3/crops/non_faces_test/"
IN_SIZE = (31,31)  
POS = [1,0]
NEG = [0,1]

class Manager:
    def __init__(self):
        self.folders_vec = []
        self.img_info_vec = []
        self.fddb_dir = "../dataset/"
        # img_counter_vec holds the image counters for the train and the test set
        self.train_img_counter = 0
        self.test_img_counter = 0
        self.window_size = 31
        self.roi = roi.ROI(self.window_size)
        self.batch_vec = [[],[]]
    
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
            #print(img_info.img_path)
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
        print(str(crop_counter)+" crops created")
            #counter += 1
            #if counter > 0:
            #    break
    
    def noise(self, im):
        row,col = im.shape
        mean = 0
        sigma = 0.4*im.std()
        gauss = np.random.normal(mean,sigma,(row,col))
        output = im + gauss
        output = np.clip(output,0,255)
        output = output.astype(np.uint8)
        return output

    def crop_images_noise(self, positive_path, negative_path):
        counter = 0
        crop_counter = 0
        for img_info in self.img_info_vec:
            #print(img_info.img_path)
            img = cv2.imread(img_info.img_path,0)
            counter += 1
            #img = self.noise(img)
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
        print(counter)

    def load_images_aux(self, batch_nb, FACE_PATH, NON_FACE_PATH):
        print(FACE_PATH)
        print(NON_FACE_PATH)
        batch_tmp = []
        #Load faces (positive samples)
        for n in os.listdir(FACE_PATH):
            name = FACE_PATH+n
            img_path = name
            #print(img_path)
            img = cv2.imread(img_path,0)
            t_img = cv2.resize(img,IN_SIZE)
            #original image
            batch_tmp.append((t_img, POS))
            #horizontal flipped image
            batch_tmp.append((cv2.flip(t_img,1),POS)) #Use the horizontal mirror image
            #rotated image (-90,0,90)
            #angle = np.random.randint(-1,1,1)*90
            #batch.append(ndimage.rotate(t_img,angle,reshape=False))
            #TODO alterate intensities
        print(len(batch_tmp))
        nbatch = []
        #Load faces (positive samples)
        for n in os.listdir(NON_FACE_PATH):
            name = NON_FACE_PATH+n
            img_path = name
            img = cv2.imread(img_path,0)
            t_img = cv2.resize(img,IN_SIZE)
            nbatch.append((t_img, NEG))
            nbatch.append((cv2.flip(t_img,1),NEG)) #Use the horizontal mirror image
        print(len(nbatch))
        batch_tmp.extend(nbatch)
        random.shuffle(batch_tmp)
        print(len(batch_tmp))
        reshaped_batch = [[],[]]
        for b in batch_tmp:
            reshaped_batch[0].append(b[0])
            reshaped_batch[1].append(b[1])
        print(len(reshaped_batch))
        self.batch_vec[batch_nb] = reshaped_batch

    def load_images(self):
        self.load_images_aux(0,FACE_PATH_TRAIN,NON_FACE_PATH_TRAIN)
        print(len(self.batch_vec[0][0]))
        self.load_images_aux(1,FACE_PATH_TEST,NON_FACE_PATH_TEST)

    def reset_train_img_counter(self):
        self.train_img_counter = 0

    def reset_test_img_counter(self):
        self.test_img_counter = 0

    def next_random_batch_train(self, n):
        batch =[[],[]]
        full_batch = True
        batch_len = len(self.batch[0])
        rand = random.randint(0,batch_len)
        for i in range(rand, rand+n):
            batch[0].append(self.batch[0][i%batch_len])
            batch[1].append(self.batch[1][i%batch_len])
        return batch, full_batch

    def next_batch_aux(self, n, batch_nb):
        batch =[[],[]]
        full_batch = True
        for i in range(self.test_img_counter, self.test_img_counter+n):
            if(i >= len(self.batch_vec[1][0])):
                full_batch = False
                break # no data left
            batch[0].append(self.batch_vec[batch_nb][0][i])
            batch[1].append(self.batch_vec[batch_nb][1][i])
        self.test_img_counter += n
        return batch, full_batch

    def next_batch_train(self, n):
        return self.next_batch_aux(n, 0)

    def next_batch_test(self, n):
        return self.next_batch_aux(n, 1)
