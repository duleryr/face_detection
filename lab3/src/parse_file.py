#!/usr/bin/env python
import math

class ellipse:
    def __init__(self, r_a, r_b, theta, c_x, c_y):
        self.c_x = float(c_x);
        self.c_y = float(c_y);
        self.theta = math.degrees(float(theta)); #the information in the FDDB is in radians
        self.r_a = float(r_a);
        self.r_b = float(r_b);

class img_information:
    def __init__(self, img_path, nb_faces, list_ellipse):
        self.img_path = img_path
        self.nb_faces = nb_faces
        self.img_shape = (0,0) # for the construction of the masks
        self.list_ellipse = list_ellipse

def get_nb_images(file_name):
    nb_images = 0
    with open(file_name, "r") as fh:
        for line in fh:
            nb_images += 1
    return nb_images
        
#get the ellipse information for all the faces in the image
def get_img_info(fh):
    img_path = "../dataset/" + fh.readline()
    nb_faces = int(fh.readline())
    
    list_ellipse = []
    for i in range(0,nb_faces):
        e_str_tmp = str.split(fh.readline())
        e_tmp = ellipse(e_str_tmp[0],e_str_tmp[1],e_str_tmp[2],
            e_str_tmp[3],e_str_tmp[4])
        list_ellipse.append(e_tmp)
    return img_information(img_path.rstrip()+".jpg", int(nb_faces), list_ellipse)

def get_img_info_wider(fh, folder_nb):
    current = ""
    while (current[:len(folder_nb)+1] != folder_nb + "/"):
        current = fh.readline()
    img_path = "../dataset/WIDER_train/images/" + current
    nb_faces = int(fh.readline())
    
    list_ellipse = []
    nb_occlused = 0
    for i in range(0,nb_faces):
        e_str_tmp = str.split(fh.readline())
        #if (e_str_tmp[8] != "2"): # skipping heavy occlustion
        e_tmp = ellipse(e_str_tmp[2], e_str_tmp[3], 0,
                        float(e_str_tmp[0]) + float(e_str_tmp[2])/2,
                        float(e_str_tmp[1]) + float(e_str_tmp[3])/2)
        list_ellipse.append(e_tmp)
        #else:
        #    nb_occlused += 1
    return img_information(img_path.rstrip(), int(nb_faces) - nb_occlused, list_ellipse)
