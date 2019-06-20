# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 07:41:52 2019

@author: VanBoven
"""

import os
import numpy as np

import cv2

import geopandas as gpd
import rasterio 
from shapely.geometry import mapping
from rasterio.mask import mask

    
def remove_outliers_DEM(x):
    x = x[0,:,:]
    x[x < -5] = np.nan
    x[x > 15] = np.nan
    x_masked = np.ma.masked_invalid(x) 
    return x_masked
    
#define variables
output_path = r'F:\700 Georeferencing\Hendrik de Heer georeferencing\testing/' 
img_path = r'F:\700 Georeferencing\Hendrik de Heer georeferencing\DEM/20190603_modified.tif'
shape_file = r'F:\700 Georeferencing\Hendrik de Heer georeferencing\Results/merged_points.shp'
#img size in m radius from centroid
img_size = 0.50
it = list(range(0,300000, 1))

def extract_DEM_values(img_size, img_path, shape_file, output_path, it):
    #read shp into gdf and convert to projected crs of NL   
    gdf_points = gpd.read_file(shape_file)
    point_set = gdf_points.copy()
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
    gdf_points['output'] = gdf_points.output.apply(lambda x:remove_outliers_DEM(x))
    
    #gdf_points['height'] = gdf_points.output.apply(lambda x: x.max()-x.min())
    gdf_points['height'] = gdf_points.output.apply(lambda x: (x.mean+2.5*x.std())- (x.mean()-2.5*x.std()))
    
    point_set['height'] = gdf_points.height

    point_set.to_file(os.path.join(output_path, 'points_height.shp'))
    
extract_DEM_values(img_size, img_path, shape_file, output_path, it)