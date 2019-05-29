# -*- coding: utf-8 -*-
"""
Created on Wed May 22 11:03:45 2019

@author: VanBoven
"""

import os
import numpy as np

import cv2

import geopandas as gpd
import rasterio 
from shapely.geometry import mapping
from rasterio.mask import mask

def reshape_img(x):
    x = x[0:3,:,:]
    y = np.zeros((x.shape[1], x.shape[2], x.shape[0]), dtype = np.uint8)
    y[:,:,0] = x[2,:,:]
    y[:,:,1] = x[1,:,:]    
    y[:,:,2] = x[0,:,:]
    y= y[1:-1, 1:-1]
    return y
    
#define variables
output_path = r'E:\400 Data analysis\410 Plant count\Training_data\mammoet/' 
img_path = r'E:\VanBovenDrive\VanBoven MT\Archive\c01_verdonk\Mammoet\20190521\1139\Orthomosaic/c01_verdonk-Mammoet-201905211139_clipped.tif'
shape_file = r'E:\400 Data analysis\410 Plant count\test_results\c01_verdonk-Mammoet-201905211139_clipped_points2.shp'
#img size in m radius from centroid
img_size = 0.18
it = list(range(0,300000, 8))

def points2train_img(img_size, img_path, shape_file, output_path, it):
    #read shp into gdf and convert to projected crs of NL   
    gdf_points = gpd.read_file(shape_file)
    gdf_points = gdf_points.to_crs({'init': 'epsg:28992'})
    
    #create square buffer around points
    gdf_points.geometry = gdf_points.geometry.buffer(distance = img_size, cap_style = 3)
    
    #convert square geometries back to wgs84
    gdf_points = gdf_points.to_crs({'init': 'epsg:4326'})
    
    #convert geometries to json features
    gdf_points['geoms'] = gdf_points.geometry.values # list of shapely geometries
    gdf_points['mapped_geom'] = gdf_points.geoms.apply(lambda x:[mapping(x)])
    
    #mask image with shapes
    with rasterio.open(img_path) as src:
         gdf_points['out_img/out_trans'] = gdf_points.mapped_geom.apply(lambda x:mask(src, x, crop=True))

    #get output images
    gdf_points['output'] = gdf_points['out_img/out_trans'].apply(lambda x:x[0])
    #reshape to opencv img format
    gdf_points.output = gdf_points.output.apply(lambda x: reshape_img(x))
    
    #write each img to disk
    for i, img in enumerate(gdf_points.output):
        if i in it:
            cv2.imwrite(output_path + str(os.path.basename(shape_file))[:-4]+str(i)+'.jpg',img)
    return
