#!/usr/bin/env python

import sys
import calc_histo
import cv2
import parse_file
import os

repository = sys.argv[1]
for filename in os.listdir(repository):
    if filename.endswith("ellipseList.txt"):
        print(filename)
        # descriptor_file = open(repository + "/" + filename)
        # img_info = parse_file.get_img_info(descriptor_file)
        # 
        # img_global = cv2.imread(img_info.img_path)
        # 
        # masked_img = calc_histo.calc_mask(img_global)
        # hist = calc_hist(img_global, masked_img)

