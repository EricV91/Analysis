# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 21:52:38 2019

@author: VanBoven
"""


import os

import pandas as pd

import time
import cv2
import sklearn
import numpy as np
import numpy.ma as ma
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing

from sklearn.datasets import make_blobs

from PIL import Image

#import rasterio
import gdal
from osgeo import gdalnumeric
from osgeo import ogr, osr
from fiona.crs import from_epsg
import fiona
import geopandas as gpd
import rasterio 
from rasterio.features import shapes
from shapely.geometry import mapping, Polygon, shape, Point
from rasterio.mask import mask

from functools import partial
from shapely.ops import transform
import pyproj
    
import logging

from skimage.util.shape import view_as_blocks

#initiate log file
#timestr = time.strftime("%Y%m%d-%H%M%S")
#for handler in logging.root.handlers[:]:
#    logging.root.removeHandler(handler)
#logging.basicConfig(filename = r"F:\700 Georeferencing\AZ74 georeferencing\Log/" + str(timestr) + "_log_file.log",level=logging.DEBUG)

cover_lab = cv2.cvtColor(np.array([[[165,159,148]]]).astype(np.uint8), cv2.COLOR_BGR2LAB)
cover_init = np.array(cover_lab[0,0,1:3], dtype = np.uint8)
background_init = cv2.cvtColor(np.array([[[120,125,130]]]).astype(np.uint8), cv2.COLOR_BGR2LAB) # as sampled from tif file
background_init = np.array(background_init[0,0,1:3])
green_lab = cv2.cvtColor(np.array([[[87,116,89]]]).astype(np.uint8), cv2.COLOR_BGR2LAB)
green_init = np.array(green_lab[0,0,1:3])

#create init input for clustering algorithm
kmeans_init = np.array([background_init, green_init, cover_init])

min_plant_size = 36

 
def isvalid(geom):
    try:
        Polygon(geom)
        return 1
    except:
        return 0

def reshape_img(x):
    x = x[0:3,:,:]
    y = np.zeros((x.shape[1], x.shape[2], x.shape[0]), dtype = np.uint8)
    y[:,:,0] = x[2,:,:]
    y[:,:,1] = x[1,:,:]    
    y[:,:,2] = x[0,:,:]
    y= y[1:-1, 1:-1]
    return y

def cluster(img, kmeans_init, min_plant_size, transform_mat):    
    kernel = np.ones((5,5), dtype='uint8')
    #convert to CieLAB colorspace
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    #get a and b bands
    a = np.array(img_lab[:,:,1])
    b2 = np.array(img_lab[:,:,2])
    
    #flatten the bands for kmeans clustering algorithm
    a_flat = a.flatten()
    b2_flat = b2.flatten()
    
    #stack bands to create final input for clustering algorithm
    Classificatie_Lab = np.column_stack((a_flat, b2_flat))
    
    #perform kmeans clustering
    kmeans = KMeans(init = kmeans_init, n_jobs = -1, max_iter = 25, n_clusters = 3, verbose = 0)
    #kmeans = KMeans(n_jobs = -1, max_iter = 25, n_clusters = 3, verbose = 0)
    kmeans.fit(Classificatie_Lab)
    y_kmeans = kmeans.predict(Classificatie_Lab)
    unique, counts = np.unique(y_kmeans, return_counts=True)
    if (counts[1] > -1): #min_plant_size): #& (counts[1] < 1500):
        centres = kmeans.cluster_centers_
        get_green = np.argmax(centres[:,1] - centres[:,0])
        
        y_kmeans[y_kmeans == get_green] = 1
        y_kmeans[y_kmeans != get_green] = 0

        #convert binary output back to 8bit image                
        kmeans_img = y_kmeans                
        kmeans_img = kmeans_img.reshape(img.shape[0:2]).astype(np.uint8)
        binary_img = kmeans_img * 250
        closing = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
    else:
        closing = np.zeros((img.shape), dtype = np.uint8)
    
    
    closing = cv2.dilate(closing, kernel)
    contours, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    area_list = []
    for cnt in contours:
        area_list.append(cv2.contourArea(cnt))
    try:
        area_arr = np.asarray(area_list)
        get_cnt = np.argmax(area_arr)
        cnt = contours[get_cnt]
        area = area_arr.max()
        geom = rasterio.transform.xy(transform = transform_mat, rows = list(cnt[:,0,1]), cols = list(cnt[:,0,0]), offset='ul')
        
        x_list = geom[0]
        y_list = geom[1]
        coords_list = []
        for i in range(len(x_list)):
            x = x_list[i]
            y = y_list[i]
            coords_list.append([x, y])
    except:
        area = 0 
        coords_list = []
    #shape_list.append(coords_list)
    return area, coords_list

    #poly = Polygon(coords_list)    

#img_size corresponding with plant spacing
img_size = 0.22

#input points
gdf_points = gpd.read_file(r"F:\700 Georeferencing\Hendrik de Heer georeferencing\Results/merged_points.shp")
gdf_points = gdf_points.iloc[15000:25000]

points = gdf_points.copy()

#input time series
img_path = r'F:\700 Georeferencing\Hendrik de Heer georeferencing\clipped_imagery'
filenames = os.listdir(img_path)

out_path = r'F:\700 Georeferencing\Hendrik de Heer georeferencing\Results'

gdf_points = gdf_points.to_crs({'init': 'epsg:28992'})

#create square buffer around points
gdf_points.geometry = gdf_points.geometry.buffer(distance = img_size, cap_style = 1)

#convert square geometries back to wgs84
gdf_points = gdf_points.to_crs({'init': 'epsg:4326'})

#convert geometries to json features
gdf_points['geoms'] = gdf_points.geometry.values # list of shapely geometries
gdf_points['mapped_geom'] = gdf_points.geoms.apply(lambda x:[mapping(x)])

gdf_points = gdf_points.drop(['geoms'], axis=1)    

for filename in filenames:
    if filename.endswith('.tif'):
        shape_list = []
        area_list = []
            
        #mask image with shapes
        with rasterio.open(os.path.join(img_path, filename)) as src:
            transform_mat = src.transform
            gdf_points['out_img/out_trans'] = gdf_points.mapped_geom.apply(lambda x:mask(src, x, crop=True))
        
        #get output images
        gdf_points['output'] = gdf_points['out_img/out_trans'].apply(lambda x:x[0])
        #gdf_points = gdf_points.drop(['out_img/out_trans'], axis=1) 
        
        #reshape to opencv img format
        gdf_points.output = gdf_points.output.apply(lambda x: reshape_img(x))
        #log_list = list(range(0,100000, 500))
        #tic = time.time()
        #tac = 0
        for it in range(len(gdf_points)):
            #if it in log_list:
                #toc = time.time()
                #tec = toc-tic-tac
                #tac = toc-tic
                #logging.info("Processing of 500 points took " + str(tec) + ' seconds.')
            img = gdf_points.output.iloc[it]
            transform_mat = gdf_points['out_img/out_trans'].iloc[it][1]
            area, coords_list = cluster(img, kmeans_init, min_plant_size, transform_mat)
            shape_list.append(coords_list)
            area_list.append(area)            
            
        #points[str(filename[-19:-11])] = area_list
        df = pd.DataFrame({'area': area_list})
        df['shape'] = shape_list
        df['isvalid'] = df['shape'].apply(lambda x: isvalid(x))
        df = df[df.isvalid == 1]
        points['shape'] = shape_list
        points['isvalid'] = points['shape'].apply(lambda x: isvalid(x))
        points = points[points.isvalid == 1]
        
        gdf_poly = gpd.GeoDataFrame(df.index, geometry = [Polygon(shape1) for shape1 in df['shape']], crs = {'init': 'epsg:4326'}) 
        gdf_poly.columns = ['ind', 'geometry']
        gdf_poly.to_file(os.path.join(out_path, 'area_'+ str(filename[:-4]) + '.shp'))
        
        calc_area = gdf_poly.to_crs({'init': 'epsg:28992'})
        points[str(filename[:-4])] = np.asarray(calc_area.geometry.area)
        points = points.drop(['shape', 'isvalid'], axis = 1)
        #gdf_points['area_'+str(filename[-19:-11])] = gdf_points['output'].apply(lambda x: cluster(x, kmeans_init, min_plant_size, transform_mat, shape_list))
        print(filename + ' finished')


points['Growth1'] = points[str(filenames[3][:-4])] - points[str(filenames[2][:-4])]
points['Growth2'] = points[str(filenames[4][:-4])] - points[str(filenames[3][:-4])]
points['Total_growth'] = points[str(filenames[-1][:-4])] - points[str(filenames[2][:-4])]
points.to_file(os.path.join(out_path, 'Hendrik_area.shp'))

                
