import numpy as np
import cv2

class ROI:
    def __init__(self,window_size):
        self.window_size = window_size
        self.half_window_size = int(window_size/2)
        # the center of the roi
        self.c = np.array([self.half_window_size,self.half_window_size])
        # stride in x and y
        self.stride = np.array([1,1])

    def next_step(self, img_dim):
        img_dim = np.array(img_dim[0:2])
        outside_img = np.greater(self.c+np.array([self.half_window_size,self.half_window_size])+self.stride,img_dim)
        if(np.all(outside_img)):
            self.c[0] = -1
            return
        if(outside_img[0]):
            self.c[1] += self.stride[1]
            self.c[0] = self.half_window_size
        else:
            self.c[0] += self.stride[0]

    def get_roi_content(self, img):
        return img[(self.c[1]-self.half_window_size):(self.c[1]+self.half_window_size),(self.c[0]-self.half_window_size):(self.c[0]+self.half_window_size)]

    def reset_pos(self):
        self.c = np.array([self.half_window_size,self.half_window_size])
