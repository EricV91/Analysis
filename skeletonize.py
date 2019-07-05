# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 10:23:51 2019

@author: VanBoven
"""


import geopandas as gpd
import shapely
from shapely.geometry import Polygon, MultiPolygon, Point, LineString
import numpy as np
import cv2
from PIL import Image
import os
import math

from skimage.transform import (hough_line, hough_line_peaks, probabilistic_hough_line)
from skimage.morphology import skeletonize

os.chdir(r'C:\Users\VanBoven\Documents\GitHub\ACT_georeferencing')
#from scripts.line_operations import calculate_line_angles

def calculate_line_angles(lines):
    """Calculates the angle of every line in a list of lines and adds it to a new list.
    Parameters
    ----------
        lines(list): list of lines consisting of coordinate tupples e.g. [((1, 5)(7, 6)), ((7, 8)(2, 8))]
    Returns
    -------
        angle_list(list): list of all the lines in the input list, unit is degrees.
    """
    angle_list = []
    for line in lines:
        p0, p1 = line
        angle = np.rad2deg(np.arctan2(p0[1] - p1[1], p0[0] - p1[0]))
        angle_list += [angle]
    return angle_list


def filter_direction(lines, angles, tolerance=3):
    """Filters a list of lines so only lines in the dominant direction remain.
    Parameters
    ----------
        lines(list): list of lines consisting of coordinate tupples e.g. [((1, 5)(7, 6)), ((7, 8)(2, 8))]
        angles(list): list of all the lines in the input list, unit is degrees.
        tolerance(int/float): the upper and lower boundary of direction derivation from the median angle
    Returns
    -------
        lines_in_main_direction(list): list of all the lines in the input list that are within a given
                                       tolerance of the median direction
    """
    lines_in_main_direction = []
    median_angle = np.median(angles)
    if median_angle < (360-tolerance) and median_angle > (0+tolerance):
        max_angle = median_angle + tolerance
        min_angle = median_angle - tolerance
    elif median_angle > (360-tolerance):
        max_angle = abs(360 - (median_angle + tolerance))
        min_angle = median_angle - tolerance
    elif median_angle < (0+tolerance):
        max_angle = median_angle + tolerance
        min_angle = 360 - (tolerance-median_angle)
    for i in range(len(lines)):
        if angles[i] > min_angle and angles[i] < max_angle:
            lines_in_main_direction += [lines[i]]
    return lines_in_main_direction

def lines_to_coords(lines, reference_array, bounds):
    """Converts coordinates of lines in local coordinate system to geographic coordinates from a reference.
    Parameters
    ----------
        lines(list): list of lines consisting of coordinate tupples e.g. [((1, 5)(7, 6)), ((7, 8)(2, 8))]
        reference_array(numpy masked array): array from which the lines were derived
        bounds(tuple): tuple of the most northern, southern, eastern and western coordinate
                        of the reference array (w, s, e, n)
    Returns
    -------
        coord_lines(list): list of all the lines in the input list, converted to degrees.
    """
    # Unpack the most norther, southern, eastern and western coordinate of the raster boundaries
    coord_north, coord_south, coord_east, coord_west  = bounds[3], bounds[1], bounds[2], bounds[0]

    # Calculate the X and Y pixel sizes
    width_pixel_size, height_pixel_size = find_raster_resolution(reference_array, bounds)

    # Intitialise a list of lines which will be filled in with the coordinates of the lines
    coord_lines = []
    for line in lines:
        # Unpack the line tuples
        ((x0, y0), (x1, y1)) = line
        # Calculate the coordinates in crs that correspond to the array coorindates
        lat_coord_start = coord_north - (height_pixel_size * y0)
        lon_coord_start = coord_west + (width_pixel_size * x0)
        lat_coord_end = coord_north - (height_pixel_size * y1)
        lon_coord_end = coord_west + (width_pixel_size * x1)
        # Add the new coordinate tuples to a list
        coord_lines += [((lon_coord_start, lat_coord_start), (lon_coord_end, lat_coord_end))]
    return coord_lines

def lines_to_gdf(lines, crs='epsg:28992'):
    """Converts coordinate tuples to geodataframe.
    Parameters
    ----------
        lines(list): list of lines consisting of coordinate tupples e.g. [((1, 5)(7, 6)), ((7, 8)(2, 8))]
        crs(str): EPSG code of the coordinate system of the lines
    Returns
    -------
        gdf(Geopandas GeoDataFrame): gdf with geometries of lines
    """
    line_geometry = []
    for line in lines:
        p0, p1 = line
        line_geometry += [LineString([Point(p1[0], p1[1]), Point(p0[0], p0[1])])]
        gdf = gpd.GeoDataFrame(geometry=line_geometry)
        gdf.crs = {'init':crs}
return gdf

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

image[image > 0] = 255

# perform skeletonization
skeleton = skeletonize(image)

img, skel = skeletonize_cv2(image)

img = image[skeleton == True]

img = np.ma.array(image, mask = (skeleton == False))
img = img*255
cv2.imwrite(r'D:\700 Georeferencing\Guido Cortvriendt\temp_results/190609_skeleton2.jpg', skel)

lines = cv2.HoughLines(skel, 1, np.pi / 180, 500, None, 0, 0)
if lines is not None:
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 10000*(-b)), int(y0 + 10000*(a)))
        pt2 = (int(x0 - 10000*(-b)), int(y0 - 10000*(a)))
        cv2.line(template, pt1, pt2, (255), 3, cv2.LINE_AA)


linesP = cv2.HoughLinesP(skel, 1, np.pi / 180, 5, None, 100, 10)
lines = probabilistic_hough_line(skel, threshold = 5, line_length = 10, line_gap = 10)
template = np.zeros((image.shape[0], image.shape[1]), dtype = np.uint8)


angle = calculate_line_angles(lines)
# Draw the lines
if linesP is not None:
    for i in range(0, len(linesP)):
        l = linesP[i][0]
        cv2.line(template, (l[0], l[1]), (l[2], l[3]), (255), 3, cv2.LINE_AA)

# Draw the lines
if lines is not None:
    for i in range(0, len(lines)):
        (x1, y1), (x2, y2) = lines[i]
        cv2.line(template, (x1, y1), (x2, y2), (255), 3, cv2.LINE_AA)

cv2.imwrite(r'D:\700 Georeferencing\Guido Cortvriendt\temp_results/190609_lines2.jpg', template)



if linesP is not None:
    for i in range(0, len(linesP)):
        l = linesP[i][0]
        cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)


