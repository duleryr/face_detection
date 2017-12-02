#!/usr/bin/env python

import sys
import numpy as np
import cv2
import time

# ellipses: the annotated ellipses for this image
# faces: the faces detected by the algorithm
def evaluate(ellipses, faces):
    print(len(ellipses))
