# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 13:43:42 2019

@author: VanBoven
"""

from sklearn.neighbors import NearestNeighbors

from shapely.geometry import mapping, Polygon, shape, MultiPoint, Point

from matplotlib import pyplot as plt 

def cluster_points(x):
    index = x[0]
    index2 = x[1]
    centroid = X[index]
    centroid2 = X[index2]
    new_centroid = ((centroid[0]+centroid2[0])/2, (centroid[1]+centroid2[1])/2)
    area = gpdf.loc[index,'area']
    area2 = gpdf.loc[index2, 'area']
    if area >= area2:
        return True, centroid
    else:
        return False, centroid2

#merge close points
gdf = gpd.read_file(r'F:\700 Georeferencing\Hendrik de Heer georeferencing\Results/Count_0513.shp')
gpdf = gdf.to_crs({'init': 'epsg:28992'})
for i in range(3):
    coord = gpdf.geometry.centroid
    coord = coord.apply(lambda x:x.coords.xy)
    X = np.array(list(coord.apply(lambda x:tuple((x[0][0],x[1][0])))))
    
    knn = NearestNeighbors(algorithm='auto', leaf_size=50, n_neighbors=2, p=2,radius=300.0).fit(X)
    
    distances, indices = knn.kneighbors(X)
    distances = distances[:,1]
    indices = indices[:,1]
    
    df = pd.DataFrame()
    df['distance'] = distances
    df['indices'] = list(zip(df.index, indices))
    df2 = df[df.distance < 0.18]
    df2['cluster_points'] = df2.indices.apply(lambda x:cluster_points(x))
    df2['store'] = df2.cluster_points.apply(lambda x: x[0])
    df2['new_centroid'] = df2.cluster_points.apply(lambda x:x[1])
    drop = df2[df2.store == False]
    df2 = df2[df2.store == True]
    gpdf = gpdf.drop(drop.index, axis = 0)
    
    gpdf.geometry = gpdf.geometry.apply(lambda x: Point(x.centroid))
    
    for index in df2.index:
        new_centroid = df2.loc[index, 'new_centroid']    
        point = Point(new_centroid[0], new_centroid[1])
        gpdf.loc[index, 'geometry'] = point #MultiPoint([Point(new_centroid[0], new_centroid[1])])
    
gpdf = gpdf.to_crs({'init': 'epsg:4326'})
gpdf.to_file(r'F:\700 Georeferencing\Hendrik de Heer georeferencing\Results/merged_points.shp')
