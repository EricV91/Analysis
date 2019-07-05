# -*- coding: utf-8 -*-
"""
Created on Fri May 24 10:31:20 2019

@author: VanBoven
"""

#from tensorflow.keras import models

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

from functools import partial
from shapely.ops import transform
import pyproj
    
#from skimage.util.shape import view_as_blocks

#resize input images
def resize(x):
    new_shape = (50,50,3)
    x_resized = np.array(Image.fromarray(x).resize((50,50)))
    #X_train_new = scipy.misc.imresize(x, new_shape)
    x_rescaled = x_resized/255
    return x_rescaled

def points_in_array(x):
    for point in x:
        point_list = point[0]
        return cx, cy


def transform_geometry(geometry):
    project = partial(
    pyproj.transform,
    pyproj.Proj(init='epsg:4326'), # source coordinate system
    pyproj.Proj(init='epsg:28992')) # destination coordinate system
    geometry = transform(project, geometry)  # apply projection
    return geometry

def write_plants2shp(img_path, out_path, df):
    #get transform
    src = rasterio.open(img_path)
    #convert centroids to coords and contours to shape in lat, lon
    df['coords'] = df.centroid.apply(lambda x:rasterio.transform.xy(transform = src.transform, rows = x[0], cols = x[1], offset='ul'))
    df['geom'] = df.contours.apply(lambda x:rasterio.transform.xy(transform = src.transform, rows = list(x[:,0,1]), cols = list(x[:,0,0]), offset='ul'))
    #convert df to gdf
    #for polygon, first reformat into lists of coordinate pairs
    shape_list = []
    for geom in df.geom:
        x_list = geom[0]
        y_list = geom[1]
        coords_list = []
        for i in range(len(x_list)):
            x = x_list[i]
            y = y_list[i]
            coords_list.append([x, y])
        shape_list.append(coords_list)
    df['geom2'] = shape_list

    #create points
    gdf_point = gpd.GeoDataFrame(df, geometry = [Point(x, y) for x, y in df.coords], crs = {'init': 'epsg:4326'})
    gdf_point = gdf_point.drop(['contours', 'moment', 'cx', 'cy', 'bbox', 'output', 'input',
       'centroid', 'coords', 'geom', 'geom2'], axis=1)    
    #create polygons
    gdf_poly = gpd.GeoDataFrame(df, geometry = [Polygon(shape) for shape in df.geom2], crs = {'init': 'epsg:4326'}) 
    gdf_poly = gdf_poly.drop(['contours', 'moment', 'cx', 'cy', 'bbox', 'output', 'input',
       'centroid', 'coords', 'geom', 'geom2'], axis=1)
    
    calc_area = gdf_poly.to_crs({'init': 'epsg:28992'})
    gdf_point['area'] = np.asarray(calc_area.geometry.area)
    
    gdf_point.to_file(os.path.join(out_path, (os.path.basename(img_path)[-16:-4] + '_points2.shp')))
    gdf_poly.to_file(os.path.join(out_path, (os.path.basename(img_path)[-16:-4] + '_poly2.shp')))
    return

def multi2single(gpdf):
    gpdf = gpdf.drop(['area'], axis = 1)
    gpdf.geometry = gpdf.buffer(0)
    gpdf_dissolved = gpdf.dissolve('prediction')
    gpdf_singlepoly = gpdf_dissolved[gpdf_dissolved.geometry.type == 'Polygon']
    gpdf_multipoly = gpdf_dissolved[gpdf_dissolved.geometry.type == 'MultiPolygon']

    for i, row in gpdf_multipoly.iterrows():
        Series_geometries = pd.Series(row.geometry)
        df = pd.concat([gpd.GeoDataFrame(row, crs=gpdf_multipoly.crs).T]*len(Series_geometries), ignore_index=True)
        df['geometry']  = Series_geometries
        gpdf_singlepoly = pd.concat([gpdf_singlepoly, df])

    gpdf_singlepoly.reset_index(inplace=True, drop=True)
    return gpdf_singlepoly

def get_scaler(img_path, kmeans_init):
    #get raster
    ds = gdal.Open(img_path)
    band = ds.GetRasterBand(1)
    xsize = band.XSize
    ysize = band.YSize
    #define size of img blocks
    x_block_size = 512  
    y_block_size = 512
    #create iterator to process blocks of imagery one by one. #get 10% subset 
    it = list(range(0,1000, 5))
    #initiate nparray
    subset = np.array((), dtype = np.uint8)
    subset = subset.reshape(0,2)
    
    #iterate through img using blocks
    blocks = 0
    for y in range(0, ysize, y_block_size):
        if y > 0:
            y = y 
        if y + y_block_size < ysize:
            rows = y_block_size
        else:
            rows = ysize - y
        for x in range(0, xsize, x_block_size):
            if x > 0:
                x = x 
            blocks += 1
            #if statement for subset
            if blocks in it:
                if x + x_block_size < xsize:
                    cols = x_block_size
                else:
                    cols = xsize - x
                #read bands as array
                r = np.array(ds.GetRasterBand(1).ReadAsArray(x, y, cols, rows), dtype = np.uint(8))
                g = np.array(ds.GetRasterBand(2).ReadAsArray(x, y, cols, rows), dtype = np.uint(8))
                b = np.array(ds.GetRasterBand(3).ReadAsArray(x, y, cols, rows), dtype = np.uint(8))
                img = np.zeros([b.shape[0],b.shape[1],3], np.uint8)
                img[:,:,0] = b
                img[:,:,1] = g
                img[:,:,2] = r
                if img.mean() != 0:             
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                    img = img[:,:,1:3]                
                    flattened_arr = img.reshape(-1, img.shape[-1])
                    subset = np.append(flattened_arr, subset, axis = 0)

    scaler = preprocessing.MinMaxScaler(feature_range = (0,255))
    #get a and b band of subset
    a = subset[:,0]
    b = subset[:,1]
    subset = None
    
    #stack bands to create final input for clustering algorithm
    Classificatie_Lab = np.column_stack((a, b))
    a = None
    b = None

    scaler.fit(Classificatie_Lab)

    return scaler 

def fit_kmeans_on_subset(img_path, kmeans_init):
    #get raster
    ds = gdal.Open(img_path)
    band = ds.GetRasterBand(1)
    xsize = band.XSize
    ysize = band.YSize
    #define size of img blocks
    x_block_size = 512  
    y_block_size = 512
    #create iterator to process blocks of imagery one by one. #get 10% subset 
    it = list(range(0,10000, 8))
    #initiate nparray
    subset = np.array((), dtype = np.uint8)
    subset = subset.reshape(0,2)
    
    #iterate through img using blocks
    blocks = 0
    for y in range(0, ysize, y_block_size):
        if y > 0:
            y = y 
        if y + y_block_size < ysize:
            rows = y_block_size
        else:
            rows = ysize - y
        for x in range(0, xsize, x_block_size):
            if x > 0:
                x = x 
            blocks += 1
            #if statement for subset
            if blocks in it:
                if x + x_block_size < xsize:
                    cols = x_block_size
                else:
                    cols = xsize - x
                #read bands as array
                r = np.array(ds.GetRasterBand(1).ReadAsArray(x, y, cols, rows), dtype = np.uint(8))
                g = np.array(ds.GetRasterBand(2).ReadAsArray(x, y, cols, rows), dtype = np.uint(8))
                b = np.array(ds.GetRasterBand(3).ReadAsArray(x, y, cols, rows), dtype = np.uint(8))
                img = np.zeros([b.shape[0],b.shape[1],3], np.uint8)
                img[:,:,0] = b
                img[:,:,1] = g
                img[:,:,2] = r
                if img.mean() != 0:             
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                    img = img[:,:,1:3]                
                    flattened_arr = img.reshape(-1, img.shape[-1])
                    subset = np.append(flattened_arr, subset, axis = 0)

    #get a and b band of subset
    a = subset[:,0]
    b = subset[:,1]
    subset = None
    
    #stack bands to create final input for clustering algorithm
    Classificatie_Lab = np.column_stack((a, b))
    a = None
    b = None
    
    tic = time.time()
    #perform kmeans clustering
    kmeans = KMeans(init = kmeans_init, n_jobs = -1, max_iter = 50, n_clusters = 3, verbose = 0, precompute_distances = False)
    kmeans.fit(Classificatie_Lab)
    toc = time.time()

    print('Processing took ' + str(toc-tic) + ' seconds')
    return kmeans


#set initial cluster centres based on sampling on images
cover_lab = cv2.cvtColor(np.array([[[165,159,148]]]).astype(np.uint8), cv2.COLOR_BGR2LAB)
cover_init = np.array(cover_lab[0,0,1:3], dtype = np.uint8)
background_init = cv2.cvtColor(np.array([[[120,125,130]]]).astype(np.uint8), cv2.COLOR_BGR2LAB) # as sampled from tif file
background_init = np.array(background_init[0,0,1:3])
green_lab = cv2.cvtColor(np.array([[[87,116,89]]]).astype(np.uint8), cv2.COLOR_BGR2LAB)
green_init = np.array(green_lab[0,0,1:3])

green_init = np.array([0, 0])
#create init input for clustering algorithm
kmeans_init = np.array([background_init, green_init, cover_init])

kmeans_dark_init = kmeans_init.copy() 
kmeans_dark_init[1,:] = 0
kmeans_dark_init = np.append(kmeans_dark_init, np.array([50,50]).reshape(1,-1), axis = 0)
kmeans_dark_init = np.append(kmeans_dark_init, np.array([65,65]).reshape(1,-1), axis = 0)


#load model for object classification
#model = models.load_model(r'C:\Users\VanBoven\Documents\GitHub\DataAnalysis/Broccoli_model1.h5')

# set limits to object size
#min_plant_size = 16
#max_plant_size = 1600

#use small block size to cluster based on colors in local neighbourhood
x_block_size = 512
y_block_size = 512

#input img_path
img_path = r'C:\Users\ericv\Desktop\Technodag/c08_biobrass-C49-201906041643.tif'
#output file directory
out_path = r'C:\Users\ericv\Desktop\Technodag'

#create iterator to process blocks of imagery one by one. 
it = list(range(0,200000, 1))

#True if you want to process the image and generate shapefile output
process_full_image = True
#True if you want to fit the cluster centres iteratively to every image block, false if you fit in once to a subset of the entire ortho
iterative_fit = True
#True to run the classification model and add the result to the shapefile output
run_classification = False

#kmeans, scaler = get_cluster_centroids(img_path, kmeans_init)
#scaler = get_scaler(img_path, kmeans_init)
#kmeans_init = scaler.transform(kmeans_init)
                    
# Function to read the raster as arrays for the chosen block size.
def cluster_objects(x_block_size, y_block_size, it, img_path, out_path, kmeans_init, iterative_fit, run_classification):    
    #fit kmeans to random subset of entire image if False
    if iterative_fit == False:
        kmeans = fit_kmeans_on_subset(img_path, kmeans_init)
    #time process
    tic = time.time()

    #srcArray = gdalnumeric.LoadFile(raster)
    ds = gdal.Open(img_path)
    band = ds.GetRasterBand(1)
    xsize = band.XSize
    ysize = band.YSize
    
    #create template for img mask resulting from clustering algorithm 
    template = np.zeros([ysize, xsize], np.uint8)

    #define kernel for morhpological closing operation
    kernel = np.ones((3,3), dtype='uint8')
    
    #iterate through img using blocks
    blocks = 0
    for y in range(0, ysize, y_block_size):
        if y > 0:
            y = y - 30 # use -30 pixels overlap to prevent "lines at the edges of blocks in object detection"
        if y + y_block_size < ysize:
            rows = y_block_size
        else:
            rows = ysize - y
        for x in range(0, xsize, x_block_size):
            if x > 0:
                x = x - 30
            blocks += 1
            #if statement for subset
            if blocks in it:
                if x + x_block_size < xsize:
                    cols = x_block_size
                else:
                    cols = xsize - x
                #read bands as array
                r = np.array(ds.GetRasterBand(1).ReadAsArray(x, y, cols, rows), dtype = np.uint(8))
                g = np.array(ds.GetRasterBand(2).ReadAsArray(x, y, cols, rows), dtype = np.uint(8))
                b = np.array(ds.GetRasterBand(3).ReadAsArray(x, y, cols, rows), dtype = np.uint(8))
                img = np.zeros([b.shape[0],b.shape[1],3], np.uint8)
                img[:,:,0] = b
                img[:,:,1] = g
                img[:,:,2] = r
                
                #check if block of img has values
                if img.mean() < 255: #uitzoeken of dit altijd zo is en dan evt. aanpassen in Data analysis script
                    #create img mask
                    gray = np.ma.masked_equal(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 255)
                    #img = np.ma.masked_equal(img, [255,255,255])
                    mask = gray.mask                    
                    mask_flat = mask.flatten()
                                     
                    #convert to CieLAB colorspace
                    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                    
                    #l = img_lab[:,:,0]
                    #cv2.imwrite(r'C:\Users\ericv\Desktop\Technodag/llllll_' + str(blocks)+'.jpg', l)
                    #img_lab[:,:,0] = 100
                    #cv2.imwrite(r'C:\Users\ericv\Desktop\Technodag/AB_'+ str(blocks)+'.jpg', img_lab)
                    
                    l = np.array(img_lab[:,:,0])
                    
                    #get a and b bands
                    a = np.array(img_lab[:,:,1])
                    b2 = np.array(img_lab[:,:,2])
                    
                    #flatten the bands for kmeans clustering algorithm
                    a_flat = a.flatten()
                    b2_flat = b2.flatten()
                    
                    #stack bands to create final input for clustering algorithm
                    #Classificatie_Lab = np.column_stack((a_flat, b2_flat))
                    
                    #scale values between 0-1
                    #scaler = preprocessing.MinMaxScaler()
                    #Classificatie_Lab = scaler.fit_transform(Classificatie_Lab)
                    #scaled_green = Classificatie_Lab[:,1] - Classificatie_Lab[:,0]                  
                    #value = scaled_green.mean()+1*scaled_green.std()
                    #kmeans_init = scaler.transform(kmeans_init)                                       
                    #idx = (np.abs(scaled_green - value)).argmin()
                    #kmeans_init[1,:] = Classificatie_Lab[idx,:]
                    
                    #idx_background = (np.abs(scaled_green - 0)).argmin()
                    #kmeans_init[0,:] = Classificatie_Lab[idx_background,:]

                    #scaled_cover = Classificatie_Lab[:,0] - Classificatie_Lab[:,1]   
                    #value = scaled_cover.mean()+1*scaled_cover.std()                                
                    #idx_cover = (np.abs(scaled_cover - value)).argmin()
                    #kmeans_init[2,:] = Classificatie_Lab[idx_cover,:]
                    
                    #fit kmeans to data distribution of block if True
                    """
                    if iterative_fit == True:
                        kmeans = KMeans(init = kmeans_init, n_jobs = -1, max_iter = 25, n_clusters = 3, verbose = 0)
                        #kmeans = KMeans(n_jobs = -1, max_iter = 25, n_clusters = 3, verbose = 0)
                        if len(mask_flat) == len(Classificatie_Lab):
                            kmeans.fit(Classificatie_Lab[mask_flat == False])
                        else:
                            kmeans.fit(Classificatie_Lab)
                    
                    #cluster image block
                    if len(mask_flat) == len(Classificatie_Lab):
                        y_kmeans = kmeans.predict(Classificatie_Lab[mask_flat==False])
                        kmeans_result = np.zeros((a_flat.shape))
                        kmeans_result[mask_flat == False] = y_kmeans
                        y_kmeans = kmeans_result.copy()
                    else:
                        y_kmeans = kmeans.predict(Classificatie_Lab)
                    unique, counts = np.unique(y_kmeans, return_counts=True)
                    
                    #green cluster has by default value 1
                    #get_green = 1
                    #image blocks with a lot
                    #if np.count_nonzero(img) < (512*512*3) - 25000:                        
                    #kmeans_init = kmeans.cluster_centers_
                        #get cluster centres
                        #centres = kmeans.cluster_centers_
                    #print(centres)
                        #get_green = np.argmax(centres[:,1] - centres[:,0])
                        #not_green = np.argmin(counts)
                        #not_green2 = np.argmax(counts)                    

                    #print(counts.min())
                    #print(np.argmin(counts))
                    #get_green = np.argmin(counts)
                                       
                    #Get cluster centres
                    centres = kmeans.cluster_centers_
                    #print(centres)
                    #get index of the greenest cluster centre
                    get_green = np.argmax(centres[:,1] - centres[:,0])
                    
                    #create binary output, green is one the rest is zero
                    y_kmeans[y_kmeans == get_green] = 1
                    y_kmeans[y_kmeans != get_green] = 0

                    #convert binary output back to 8bit image                
                    kmeans_img = y_kmeans                
                    kmeans_img = kmeans_img.reshape(img.shape[0:2]).astype(np.uint8)
                    binary_img = kmeans_img * 50
                    
                    #close detected shapes
                    closing = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
                    """
                    #get dark lettuce
                    l_flat = l.flatten()
                    Classificatie_Lab = np.column_stack((l_flat, l_flat))
                    if iterative_fit == True:
                        kmeans = KMeans(init = kmeans_dark_init, n_jobs = -1, max_iter = 25, n_clusters = 5, verbose = 0)
                        #kmeans = KMeans(n_jobs = -1, max_iter = 25, n_clusters = 3, verbose = 0)
                        if len(mask_flat) == len(Classificatie_Lab):
                            kmeans.fit(Classificatie_Lab[mask_flat == False])
                        else:
                            kmeans.fit(Classificatie_Lab)
                    
                    #cluster image block
                    if len(mask_flat) == len(Classificatie_Lab):
                        y_kmeans = kmeans.predict(Classificatie_Lab[mask_flat==False])
                        kmeans_result = np.zeros((a_flat.shape))
                        kmeans_result[mask_flat == False] = y_kmeans
                        y_kmeans = kmeans_result.copy()
                    else:
                        y_kmeans = kmeans.predict(Classificatie_Lab)
                        
                    get_dark = 1
                    y_kmeans[y_kmeans == get_dark] = 1
                    y_kmeans[y_kmeans != get_dark] = 0
                    
                    kmeans_img = y_kmeans                
                    kmeans_img = kmeans_img.reshape(img.shape[0:2]).astype(np.uint8)
                    binary_img = kmeans_img * 50
                    erode = cv2.erode(binary_img, kernel, iterations=1)
                    
                    #element = np.ones((7,2), dtype='uint8')
                    #closing = cv2.erode(closing, element)
                    #closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel)

                    #closing = binary_img
                    
                    #write img block result back on original sized image template         
                    template[y:y+rows, x:x+cols] = template[y:y+rows, x:x+cols] + erode
                    #print('processing of block ' + str(blocks) + ' finished')
                    
    template[template > 0] = 255
    #write of file as jpg img to store results                 
    cv2.imwrite(os.path.join(out_path, (os.path.basename(img_path)[:-4] + '.jpg')),template)     
    toc = time.time()
    print("processing of blocks took "+ str(toc - tic)+" seconds")
    
    return template
 
def contours2shp(template, process_full_image, model, out_path):
    """
    #read entire img to memory 
    ds = gdal.Open(img_path)
    band = ds.GetRasterBand(1)
    xsize = band.XSize
    ysize = band.YSize

    print('Start with classification of objects')
    #initiate output img
    output = np.zeros([ysize,xsize,3], np.uint8)
    r = np.array(ds.GetRasterBand(1).ReadAsArray()).astype(np.uint(8))
    output[:,:,0] = r
    r = None
    g = np.array(ds.GetRasterBand(2).ReadAsArray()).astype(np.uint(8))
    output[:,:,1] = g
    g = None
    b = np.array(ds.GetRasterBand(3).ReadAsArray()).astype(np.uint(8))
    output[:,:,2] = b
    b = None
    
    #result_img = np.zeros((template.shape[0],template.shape[1]),dtype=np.uint8)
    """
    #Get contours of features
    contours, hierarchy = cv2.findContours(template, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #create df with relevant data
    df = pd.DataFrame({'contours': contours})
    df['area'] = df.contours.apply(lambda x:cv2.contourArea(x)) 
    #filter features, area has to be > 0
    df = df[(df['area'] > 9)]# & (df['area'] < 1600)]
    df['moment'] = df.contours.apply(lambda x:cv2.moments(x))        
    df['centroid'] = df.moment.apply(lambda x:(int(x['m01']/x['m00']),int(x['m10']/x['m00'])))
    df['cx'] = df.moment.apply(lambda x:int(x['m10']/x['m00']))
    df['cy'] = df.moment.apply(lambda x:int(x['m01']/x['m00']))
    df['bbox'] = df.contours.apply(lambda x:cv2.boundingRect(x))
    if run_classification == True:
        #create input images for model
        df['output'] = df.bbox.apply(lambda x:output[x[1]-5: x[1]+x[3]+5, x[0]-5:x[0]+x[2]+5])
        df = df[df.output.apply(lambda x:x.shape[0]*x.shape[1]) > 0]
        #resize data to create input tensor for model
        df['input'] = df.output.apply(lambda x:resize(x))
        
        #remove img from memory
        output = None
        
        #resize images
        #df.input.apply(lambda x:x.resize(50,50,3, refcheck=False))       
        model_input = np.asarray(list(df.input.iloc[:]))
        
        #predict
        tic = time.time()
        try:
            prediction = model.predict(model_input)
            
            #get prediction result
            pred_final = prediction.argmax(axis=1)
            #add to df
            df['prediction'] = pred_final
        except:
            print('no prediction')
    
    #df['prediction'] = 1
        toc = time.time()
        print('classification of '+str(len(df))+' objects took '+str(toc - tic) + ' seconds')
    if run_classification == False:
        df['output'] = np.nan
        df['input'] = np.nan
    #write output features to shapefile
    write_plants2shp(img_path, out_path ,df)

template = cluster_objects(x_block_size, y_block_size, it, img_path, out_path, kmeans_init, iterative_fit, run_classification)
if process_full_image == True:
    contours2shp(template, process_full_image, model, out_path)






