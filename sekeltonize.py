# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 10:23:51 2019

@author: VanBoven
"""


import geopandas as gpd
import shapely
from shapely.geometry import Polygon, MultiPolygon
import numpy as np
import cv2
from PIL import Image

from skimage.morphology import skeletonize

def skeletonize_cv2(image):
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    done = False
    size = np.size(image)
    skel = np.zeros(image.shape,np.uint8)
     
    while( not done):
        eroded = cv2.erode(image,element)
        temp = cv2.dilate(eroded,element)
        temp = cv2.subtract(image,temp)
        skel = cv2.bitwise_or(skel,temp)
        image = eroded.copy()
     
        zeros = size - cv2.countNonZero(image)
        if zeros==size:
            done = True
    image[image > 0] = 255
    skel[skel > 0] = 255
    return image, skel

Image.MAX_IMAGE_PIXELS = 30000000000   

image = cv2.imread(r'D:\700 Georeferencing\Guido Cortvriendt\temp_results/c03_termote-Guido Cortvriendt-201906181831_clipped.jpg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
dilate = cv2.di

image[image > 0] = 255

# perform skeletonization
skeleton = skeletonize(image)

img, skel = skeletonize_cv2(image)

img = image[skeleton == True]

img = np.ma.array(image, mask = (skeleton == False))
img = img*255
cv2.imwrite(r'D:\700 Georeferencing\Guido Cortvriendt\temp_results/190618_skeleton.jpg', skel)


