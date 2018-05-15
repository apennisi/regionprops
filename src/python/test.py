# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 00:24:57 2018

@author: aferust
"""

import numpy as np
import pylab as pl
import cv2

from scipy import ndimage as ndi
import time
import regionprops as rp

imrgb = cv2.imread('p1.jpg'); pl.figure(); pl.imshow(imrgb)
img_gray = cv2.cvtColor(imrgb, cv2.COLOR_BGR2GRAY)

r = imrgb[:,:,0]
binary = (50 < r) & (r < 165)
im_filled = ndi.binary_fill_holes(binary).astype(np.uint8) # You may want to use opencv methods to fill holes

start_time = time.time()

regions = rp.regionprops(im_filled, img_gray)
"""
regions is a python list containing dictionaries of props. props can be accessed such that 
regions[0]["Area"]

available prop names can be seen in regionprops_.pyx
"""
print("--- %s seconds ---" % (time.time() - start_time))
print(regions[0]["Area"])






