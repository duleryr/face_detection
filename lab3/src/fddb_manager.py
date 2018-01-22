import parse_file
import roi
import graphical_tools
import numpy as np
import cv2
import random
import wider_loader
import os

nb_images_per_folder = [290,285,274,302,298,302,279,276,259,280]
FACE_PATH = "../crops/faces/"
NON_FACE_PATH = "../crops/non_faces/"
IN_SIZE = (31,31)  

class Manager:
    def __init__(self):
        self.train_folders_vec = []
        self.test_folders_vec = []
        self.train_img_info_vec = []
        self.test_img_info_vec = []
        self.fddb_dir = "../dataset/"
        # img_counter_vec holds the image counters for the train and the test set
        self.img_counter_vec = [0, 0]
        self.train_batch_counter = 0
        self.test_batch_counter = 0
        self.window_size = 11
        self.train_roi = roi.ROI(self.window_size)
        self.test_roi = roi.ROI(self.window_size)
        self.scale_factor = 0.8
        self.test_scale_level = 0
    
    def set_window_size(self, window_size_input):
        self.window_size = window_size_input
        self.train_roi = roi.ROI(self.window_size)
        self.test_roi = roi.ROI(self.window_size)

    def get_WIDER_img_info(self, bbx_file_name):
        print("get_WIDER_img_info")
        wider_vec = wider_loader.get_all_img_info(bbx_file_name)
        self.train_img_info_vec.extend(wider_vec)

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

    # this function assures that 50% of the output are true face batches and 
    # 50% are false batches. It does so by first building two seperate batches
    # and shuffling them afterwards
    def next_balanced_batch_aux(self, batch_size, img_info_vec, img_counter_vec_index, roi):
        batch = [[],[]]
        batch_true = [[],[]]
        batch_false = [[],[]]
        half_batch_size = int(batch_size/2)
        false_counter = 0
        true_counter = 0
        total_counter = 0
        if(self.img_counter_vec[img_counter_vec_index]>len(img_info_vec)):
            return batch
        img = cv2.imread(img_info_vec[self.img_counter_vec[img_counter_vec_index]].img_path,0)
        ground_truth = graphical_tools.calc_mask(img,img_info_vec[self.img_counter_vec[img_counter_vec_index]])
        #graphical_tools.showImg("test",img)
        #graphical_tools.showImg("vérité terrain",ground_truth)

        # Fetch true and false samples
        while((false_counter<half_batch_size) or (true_counter<half_batch_size)):
            sub_image = roi.get_roi_content(img)
            total_counter += 1
            #is_in_face = (ground_truth[roi.c[0]-roi.half_window_size,roi.c[1]-roi.half_window_size]>0 and
            #            ground_truth[roi.c[0]-roi.half_window_size,roi.c[1]+roi.half_window_size]>0 and
            #            ground_truth[roi.c[0]+roi.half_window_size,roi.c[1]-roi.half_window_size]>0 and
            #            ground_truth[roi.c[0]+roi.half_window_size,roi.c[1]+roi.half_window_size]>0)
            is_in_face = (ground_truth[roi.c[0],roi.c[1]]>0)

            if(is_in_face and (true_counter<half_batch_size)):
                batch_true[0].append(sub_image)
                batch_true[1].append([float(is_in_face), float(not is_in_face)])
                true_counter += 1
            if((not is_in_face) and (false_counter<half_batch_size)):
                batch_false[0].append(sub_image)
                batch_false[1].append([float(is_in_face), float(not is_in_face)])
                false_counter += 1
                
            # move the roi forward
            roi.next_step(img.shape)
            if(roi.c[0] == -1): # roi is out of bounds
                roi.reset_pos()
                self.img_counter_vec[img_counter_vec_index] += 1
                #print(self.img_counter_vec[img_counter_vec_index])
                if(self.img_counter_vec[img_counter_vec_index]>=len(img_info_vec)):
                    print("NO more images left")
                    print("img_counter: "+str(self.img_counter_vec[img_counter_vec_index]))
                    return batch, False
                img = cv2.imread(img_info_vec[self.img_counter_vec[img_counter_vec_index]].img_path,0)
                #graphical_tools.showImg("test",img)
                ground_truth = graphical_tools.calc_mask(img,img_info_vec[self.img_counter_vec[img_counter_vec_index]])

        # Insert them in the final batch in random order
        random_order = np.zeros(batch_size)
        random_order[0:half_batch_size] = 1
        random.shuffle(random_order)
        false_counter = 0
        true_counter = 0
        for i in range(batch_size):
            if(random_order[i]):
                batch[0].append(batch_true[0][true_counter])
                #graphical_tools.showImg("batch",batch_true[0][true_counter])
                batch[1].append(batch_true[1][true_counter])
                true_counter += 1
            else:
                batch[0].append(batch_false[0][false_counter])
                batch[1].append(batch_false[1][false_counter])
                false_counter += 1

        batch[0] = np.concatenate(batch[0])
        batch[0] = np.asarray(batch[0],dtype=np.float32)
        return batch, True

    # img_counter and batch_counter help us to keep track of where we stopped 
    # last time we called this function
    def next_batch_aux(self, batch_size, img_info_vec, img_counter_vec_index, roi):
        batch = [[],[]]
        if(self.img_counter_vec[img_counter_vec_index]>len(img_info_vec)):
            return batch
        #print(img_info_vec[self.img_counter_vec[img_counter_vec_index]].img_path)
        img = cv2.imread(img_info_vec[self.img_counter_vec[img_counter_vec_index]].img_path,0)
        ground_truth = graphical_tools.calc_mask(img,img_info_vec[self.img_counter_vec[img_counter_vec_index]])
        for i in range(batch_size):
            # move the roi forward
            batch[0].append(roi.get_roi_content(img))
            #graphical_tools.showImg('roi', batch[0][0])
            is_in_face = ground_truth[roi.c[0],roi.c[1]]>0 # BW
            batch[1].append([float(is_in_face), float(not is_in_face)])
            roi.next_step(img.shape)
            if(roi.c[0] == -1): # roi is out of bounds
                roi.reset_pos()
                self.img_counter_vec[img_counter_vec_index] += 1
                #print(self.img_counter_vec[img_counter_vec_index])
                if(self.img_counter_vec[img_counter_vec_index]>=len(img_info_vec)):
                    #print("NOOO more images left")
                    return batch, False
                img = cv2.imread(img_info_vec[self.img_counter_vec[img_counter_vec_index]].img_path,0)
                #graphical_tools.showImg("test",img)
                ground_truth = graphical_tools.calc_mask(img,img_info_vec[self.img_counter_vec[img_counter_vec_index]])
        #batch[0] = np.concatenate(batch[0])
        batch[0] = np.asarray(batch[0],dtype=np.float32)
        return batch, True

    def next_batch_train(self, batch_size):
        return self.next_batch_aux(batch_size,self.train_img_info_vec,0,self.train_roi)

    def next_batch_test(self, batch_size):
        return self.next_batch_aux(batch_size,self.test_img_info_vec,1,self.test_roi)

    def next_balanced_batch_train(self, batch_size):
        return self.next_balanced_batch_aux(batch_size,self.train_img_info_vec,0,self.train_roi)

    def next_balanced_batch_test(self, batch_size):
        return self.next_balanced_batch_aux(batch_size,self.test_img_info_vec,1,self.test_roi)

    def reset_img_counter(self):
        self.img_counter_vec[0] = 0
        self.img_counter_vec[1] = 0
        self.train_roi.reset_pos()
        self.test_roi.reset_pos()
    
    # returns batch_size batches with the appropriate labels
    def next_test_batch_scale(self, batch_size):
        batch = [[],[]]
        # scale_levels contains the scale_level for every batch, so that we can reconstruct the bounding boxes
        full_batch = False
        while (len(batch[0])<batch_size):
            if(self.img_counter_vec[1]>=len(self.test_img_info_vec)):
                return batch, False
            img_info = self.test_img_info_vec[self.img_counter_vec[1]]
            img = cv2.imread(img_info.img_path,0)
            ground_truth = graphical_tools.calc_mask(img,img_info)
            # resize img to the last scale level
            for i in range(self.test_scale_level):
                img = graphical_tools.resize_img(img,self.scale_factor)
                ground_truth = graphical_tools.resize_img(ground_truth,self.scale_factor)
            # parse through the leftover scales
            while(img.shape[0] >self.test_roi.window_size and img.shape[1] > self.test_roi.window_size):
                #print(self.test_scale_level)
                # parse through the leftover roi positions
                while (self.test_roi.c[0]!=-1):
                    sub_image = self.test_roi.get_roi_content(img)
                    # rgb
                    #is_in_face = ground_truth[self.test_roi.c[0],self.test_roi.c[1]][0]>0
                    # bw
                    is_in_face = (ground_truth[self.test_roi.c[0],self.test_roi.c[1]]>0)
                    # DEBUG: check if we got the ground_truth right
                    #if(is_in_face):
                    #    cv2.rectangle(img,(self.test_roi.c[1]-self.test_roi.half_window_size,self.test_roi.c[0]-self.test_roi.half_window_size),
                    #        (self.test_roi.c[1]+self.test_roi.half_window_size,self.test_roi.c[0]+self.test_roi.half_window_size),(255,0,0),2)
                    batch[0].append(sub_image)
                    batch[1].append((is_in_face, not is_in_face))
                    self.test_roi.next_step(img.shape)
                    if(len(batch[0])>=batch_size):
                        full_batch = True
                        break

                if(full_batch):
                    break
                # prepare next iteration
                img = graphical_tools.resize_img(img,self.scale_factor)
                ground_truth = graphical_tools.resize_img(ground_truth,self.scale_factor)
                self.test_roi.reset_pos()
                self.test_scale_level += 1

                #graphical_tools.showImg("test_scale", img)
                #graphical_tools.showImg("test_scale", ground_truth)
            if(full_batch):
                break
            self.img_counter_vec[1] += 1
            self.test_roi.reset_pos()
            self.test_scale_level = 0
        batch[0] = np.asarray(batch[0],dtype=np.float32)
        return batch, full_batch


    # the function goes through the entire image pyramid
    def go_through_pyramid_face(self,img_info):
        img = cv2.imread(img_info.img_path,0)
        ground_truth = graphical_tools.calc_mask(img,img_info)
        while(img.shape[0] >self.train_roi.window_size and img.shape[1] > self.train_roi.window_size):
            new_size = (int(img.shape[0]*self.scale_factor),int(img.shape[1]*self.scale_factor))
            img = cv2.resize(img,(new_size[1],new_size[0]))
            ground_truth = cv2.resize(ground_truth,(new_size[1],new_size[0]))
            graphical_tools.showImg("pyrDown",img)
            graphical_tools.showImg("pyrDown",ground_truth)
        

    def extract_faces(self):
        face_counter = 0
        print(face_counter)
        for img_info in self.train_img_info_vec:
            img = cv2.imread(img_info.img_path)
            print(img_info.img_path)
            ellipse_counter = 0
            for f in img_info.list_ellipse:
                #print("ellipse_counter: "+str(ellipse_counter))
                #print(f.c_x)
                #print(f.c_y)
                #print(f.r_a)
                #print(f.r_b)
                pt1 = (int(f.c_x-f.r_b),int(f.c_y-f.r_a))
                pt2 = (int(f.c_x+f.r_b),int(f.c_y+f.r_a))
                sub_image = img[max(0,pt1[1]):min(img.shape[0],pt2[1]),max(0,pt1[0]):min(img.shape[1],pt2[0])]
                #print(sub_image.shape)
                #print(pt1)
                #print(pt2)
                #print(max(0,pt1[1]))
                #print(min(img.shape[0],pt2[1]))
                #print(max(0,pt1[0]))
                #print(min(img.shape[1],pt2[0]))
                if((sub_image.shape[0] != 0) and (sub_image.shape[1] != 0)):
                    cv2.imwrite(FACE_PATH+str(face_counter)+".png",cv2.resize(sub_image,IN_SIZE))
                    face_counter += 1
                ellipse_counter += 1

    def create_non_faces(self):
        # get number of small faces
        nb_faces = len(os.listdir(FACE_PATH))
        # get a random non-face roi from each image
        self.train_roi.reset_pos()
        non_face_counter = 0
        for img_info in self.train_img_info_vec:
            print(img_info.img_path)
            img = cv2.imread(img_info.img_path)
            ground_truth = graphical_tools.calc_mask(img,img_info)
            #if((img.shape[0] == 683) and (img.shape[1] == 1024)):
            #    graphical_tools.showImg("img", img)
            print(str(img.shape) +", "+str(len(img_info.list_ellipse)))
            ellipse_counter = 0
            for f in range(len(img_info.list_ellipse)):
                print(ellipse_counter)
                found_patch = False
                random_counter = 0
                while (not found_patch) and (random_counter <  10):
                    self.train_roi.reset_pos()
                    random_step = random.randint(0,int(img.shape[0]*img.shape[1]/(self.train_roi.window_size*self.train_roi.window_size/2)))
                    random_counter += 1
                    for i in range(random_step):
                        self.train_roi.next_step(img.shape)
                        if(self.train_roi.c[0] == -1):
                            self.train_roi.reset_pos()
                    found_patch = bool(ground_truth[self.train_roi.c[0],self.train_roi.c[1]][0]==0)
                    if found_patch:
                        #graphical_tools.showImg("roi",self.train_roi.get_roi_content(img))
                        file_write_check = cv2.imwrite(NON_FACE_PATH+str(non_face_counter)+".png",self.train_roi.get_roi_content(img))
                        non_face_counter += 1
                ellipse_counter += 1

    def next_batch_faceDetect(self, img_info_vec, img_counter_vec_index):
        batch = [[],[]]
        img = cv2.imread(img_info_vec[0].img_path)
        ground_truth = graphical_tools.calc_mask(img,img_info_vec[0])
        batch[0].append(img)
        batch[1].append(ground_truth)
        for i in range(nb_images_per_folder[self.test_folders_vec[0]-1]):
            img = cv2.imread(img_info_vec[i].img_path)
            ground_truth = graphical_tools.calc_mask(img,img_info_vec[i])
            batch[0].append(img)
            batch[1].append(ground_truth)
        return batch
