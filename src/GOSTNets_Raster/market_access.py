import sys, os, importlib, time, copy
import rasterio

import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
import GOSTnets as gn
import skimage.graph as graph

from skimage.graph import _mcp
from rasterio.mask import mask
from rasterio import features
from shapely.geometry import box, Point, shape
from shapely.wkt import loads
from shapely.ops import cascaded_union
from scipy import sparse
from scipy.ndimage import generic_filter
from pandana.loaders import osm

speed_dict = {
   'residential': 20,  # kmph
   'primary': 40,
   'primary_link':35,
   'motorway':50,
   'motorway_link': 45,
   'trunk': 40,
   'trunk_link':35,
   'secondary': 30,
   'secondary_link':25,
   'tertiary':30,
   'tertiary_link': 25,
   'unclassified':20,
   'living_street':10,
   'service':10
}
'''prints the time along with the message'''
def tPrint(s):
    print("%s\t%s" % (time.strftime("%H:%M:%S"), s))

def get_speed(x, s_dict):
    try: 
        
        speed = s_dict[x]
    except:
        if type(x) == list:
            try:
                speed = s_dict[x[0]]
            except:
                speed = 5
        else:
            speed=5
    return(speed)

def get_nodes(b, tags):
    nodes = osm.node_query(b[1], b[0], b[3], b[2], tags=tags)
    nodes_geom = [Point(x) for x in zip(nodes['lon'], nodes['lat'])]
    nodes_df = gpd.GeoDataFrame(nodes[['amenity','lat','lon']], geometry=nodes_geom, crs={'init':'epgs:4326'})
    return(nodes_df)
    
def get_roads(b):
    sel_graph = ox.graph_from_bbox(b[3], b[1], b[2], b[0], retain_all=True)
    sel_roads = gn.edge_gdf_from_graph(sel_graph)
    sel_roads['speed'] = sel_roads['highway'].apply(lambda x: get_speed(x, speed_dict))
    return(sel_roads)

def get_mcp_dests(inH, destinations, makeset=True):
    ''' Get indices from inH for use in mcp.find_costs
    INPUT
        inH[rasterio] - object from which to extract geographic coordinates
        destinations[geopandas geodataframe] - point geodataframe of destinations
    RETURN
        [list of indices]
    '''
    if makeset:
        cities = list(set([inH.index(x.x, x.y) for x in destinations['geometry']]))
    else:
        cities = list([inH.index(x.x, x.y) for x in destinations['geometry']])
        
    cities = [x for x in cities if ((x[0] > 0) and (x[1] > 0) and 
                (x[0] <= inH.shape[0]) and (x[1] <= inH.shape[1]))]
    return(cities)
    
def name_mcp_dests(inH, destinations):
    ''' Somestimes multiple destinations fall within the same cell and duplicates are removed before processing in get_mcp_dests.
            This function is designed to created a column in destinations that can be merged with results from get_mcp_dests
        INPUT
            inH[rasterio] - object from which to extract geographic coordinates
            destinations[geopandas geodataframe] - point geodataframe of destinations
        RETURN
            destinations [geopandas] - returns with new column called MCP_DESTS_NAME        
    '''
    destinations['MCP_DESTS_NAME'] = ''
    for idx, row in destinations.iterrows():
        x = row['geometry']
        destinations.loc[idx, 'MCP_DESTS_NAME'] = "_".join([str(xx) for xx in inH.index(x.x, x.y)])    
    return(destinations)
    
    
def generate_network_raster(inH, sel_roads, min_speed=5, speed_col='speed', resolution=100):   
    ''' Create raster with network travel times from a road network
    
    INPUTS
        inH [rasterio object] - template raster used to define raster shape, resolution, crs, etc.
        sel_roads [geopandas dataframe] - road network to burn into raster
        [optional] min_speed [int] - minimum travel speed for areas without roads
        [optional] speed_col [string] - column in sel_roads that defines the speed in KM/h
        [optional] resolution [int] - resolution of the raster in metres
        
    RETURNS
        [numpy array]
    '''
    # create a copy of inH with value set to slowest walking speed
    distance_data = np.zeros(inH.shape)
    # burn the speeds into the distance_data using the road network 
    sel_roads = sel_roads.sort_values([speed_col])
    shapes = ((row['geometry'], row[speed_col]) for idx, row in sel_roads.iterrows())
    speed_image = features.rasterize(shapes, out_shape=inH.shape, transform=inH.transform, fill=min_speed)
    # convert to a version that claculates the seconds to cross each cell
    traversal_time = resolution / (speed_image * 1000 / (60 * 60)) # km/h --> m/s * resolution of image in metres
    return(traversal_time)
   
def calculate_travel_time(inH, mcp, destinations, out_raster = ''):
    ''' Calculate travel time raster
    
    INPUTS
        inH [rasterio object] - template raster used to identify locations of destinations
        mcp [skimage.graph.MCP_Geometric] - input graph
        destinations [geopandas df] - destinations for nearest calculations
        
    LINKS
        https://scikit-image.org/docs/0.7.0/api/skimage.graph.mcp.html#skimage.graph.mcp.MCP.find_costs
    '''
    # create skimage graph
    cities = get_mcp_dests(inH, destinations)
    costs, traceback = mcp.find_costs(cities)        
    if not out_raster == '':
        meta = inH.meta.copy()
        meta.update(dtype=costs.dtype)
        with rasterio.open(out_raster, 'w', **meta) as out:
            out.write_band(1, costs)
            
    return((costs, traceback))    
    
def get_all_amenities(bounds):
    amenities = ['toilets', 'washroom', 'restroom']
    toilets_tags = '"amenity"~"{}"'.format('|'.join(amenities))
    toilets = get_nodes(inH.bounds, toilets_tags)

    amenities = ['water_points', 'drinking_water', 'pumps', 'water_pumps', 'well']
    water_tags = '"amenity"~"{}"'.format('|'.join(amenities))
    water_points = get_nodes(inH.bounds, water_tags)
        
    amenities = ['supermarket', 'convenience', 'general', 'department_stores', 'wholesale', 'grocery', 'general']
    shp_tags = '"shop"~"{}"'.format('|'.join(amenities))
    shops = get_nodes(inH.bounds, shp_tags)
    
    
def generate_feature_vectors(network_r, mcp, inH, threshold, featIdx, verbose=True):
    ''' Generate individual market sheds for each feature in the input dataset
    
    INPUTS
        network_r [rasterio] - raster from which to grab index for calculations in MCP
        mcp [skimage.graph.MCP_Geometric] - input graph
        inH [geopandas data frame] - geopandas data frame from which to calculate features
        threshold [list of int] - travel treshold from which to calculate vectors in units of graph
        featIdx [string] - column name in inH to append to output marketshed dataset
        
    RETURNS
        [geopandas dataframe]
    '''
    n = inH.shape[0]
    feat_count = 0
    complete_shapes = []
    for idx, row in inH.iterrows():
        feat_count = feat_count + 1
        if verbose:
            tPrint(f"{feat_count} of {n}: {row[featIdx]}")
        cur_idx = network_r.index(row['geometry'].x, row['geometry'].y)
        if cur_idx[0] > 0 and cur_idx[1] > 0 and cur_idx[0] < network_r.shape[0] and cur_idx[1] < network_r.shape[1]:
            costs, traceback = mcp.find_costs([cur_idx])
            for thresh in threshold:
                within_time = ((costs < thresh) * 1).astype('int16')
                all_shapes = []
                for cShape, value in features.shapes(within_time, transform = network_r.transform):
                    if value == 1.0:
                        all_shapes.append([shape(cShape)])
                complete_shape = cascaded_union([x[0] for x in all_shapes])
                complete_shapes.append([complete_shape, thresh, row[featIdx]])
    final = gpd.GeoDataFrame(complete_shapes, columns=["geometry", "threshold", "IDX"], crs=network_r.crs)
    return(final)
    
def generate_market_sheds(inR, inH, out_file='', verbose=True, factor=1000, bandIdx=0):
    ''' identify pixel-level maps of market sheds based on travel time    
    INPUTS
        inR [rasterio] - raster from which to grab index for calculations in MCP
        inH [geopandas data frame] - geopandas data frame of destinations
        factor [int] - value by which to multiply raster 
        
    RETURNS
        [numpy array] - marketsheds by index
        
    NOTES:
        Incredible help from StackOverflow:
        https://stackoverflow.com/questions/62135639/mcp-geometrics-for-calculating-marketsheds
        https://gist.github.com/bpstewar/9c15fc0948e82aa9667f1b04fd2c0295
    '''
    xx = inR.read()[bandIdx,:,:] * factor
    orig_shape = xx.shape
    # In order to calculate the marketsheds, the input array needs to be NxN shape, 
    #   at the end, we will select out the original shape in order to write to file
    max_speed = xx.max()
    if xx.shape[0] < xx.shape[1]:
        extra_size = np.zeros([(xx.shape[1] - xx.shape[0]), xx.shape[1]]) + max_speed
        new_xx = np.vstack([xx, extra_size])
        
    if xx.shape[1] < xx.shape[0]:
        extra_size = np.zeros([(xx.shape[0] - xx.shape[1]), xx.shape[0]]) + max_speed
        new_xx = np.hstack([xx, extra_size])        
    mcp = graph.MCP_Geometric(new_xx)
    
    
    dests = get_mcp_dests(inR, inH)    
    costs, traceback = mcp.find_costs(dests)
    
    offsets = _mcp.make_offsets(2, True)
    offsets.append(np.array([0, 0]))
    offsets_arr = np.array(offsets)
    indices = np.indices(traceback.shape)
    offset_to_neighbor = offsets_arr[traceback]
    neighbor_index = indices - offset_to_neighbor.transpose((2, 0, 1))
    ids = np.arange(traceback.size).reshape(costs.shape)
    neighbor_ids = np.ravel_multi_index(
        tuple(neighbor_index), traceback.shape
    )
    g = sparse.coo_matrix((
        np.ones(traceback.size),
        (ids.flat, neighbor_ids.flat),
    ), shape=[traceback.size, traceback.size]).tocsr()
    n, components = sparse.csgraph.connected_components(g)
    basins = components.reshape(costs.shape)
    out_basins = basins[:orig_shape[0], :orig_shape[1]]
    if out_file != '':
        meta = inR.meta.copy()
        meta.update(dtype=out_basins.dtype)
        with rasterio.open(out_file, 'w', **meta) as out_raster:
            out_raster.write_band(1, out_basins)
    else:
        return(out_basins)

def generate_market_sheds_old(img, mcp, inH, out_file = '', verbose=True):
    ''' identify pixel-level maps of market sheds based on travel time
    
    INPUTS
        network_r [rasterio] - raster from which to grab index for calculations in MCP
        mcp [skimage.graph.MCP_Geometric] - input graph
        inH [geopandas data frame] - geopandas data frame from which to calculate features
        
    RETURNS
        [numpy array]
    '''
    dests_geom = [img.index(x.x, x.y) for x in inH['geometry']]
    all_c = []
    n = inH.shape[0]
    idx = 0
    for dest in dests_geom:
        idx += 1
        if dest[0] > 0 and dest[0] < img.shape[0] and dest[1] > 0 and dest[1] < img.shape[1]:
            if verbose:
                tPrint(f"{idx} of {n}")
            c1, trace = mcp.find_costs([dest])
            all_c.append(copy.deepcopy(c1))
        else:
            tPrint(f"{idx} of {n} cannot be processed")
    if verbose:
        tPrint("Finished calculating access")
    # Iterate through results to generate final marketshed
    output = np.zeros(all_c[0].shape)
    for idx in range(0, len(all_c)):
        cur_res = all_c[idx]
        if idx == 0:
            min_res = cur_res
        else:
            combo = np.dstack([min_res, cur_res])
            min_res = np.amin(combo, 2)
            cur_val = (min_res == cur_res).astype(np.byte)
            m_idx = np.where(cur_val == 1)
            output[m_idx] = idx
    '''
    res = np.dstack(all_c)
    res_min = np.amin(res, axis=2)
    output = np.zeros([res_min.shape[0], res_min.shape[1]])
    for idx in range(0, res.shape[2]):
        cur_data = res[:,:,idx]
        cur_val = (cur_data == res_min).astype(np.byte) * idx
        output = output + cur_val
    output = output.astype(np.byte)
    def get_min_axis(x):
        return(np.where(x == x.min()))
    res_min = np.apply_along_axis(get_min_axis, 2, res)
    '''        
    if verbose:
        tPrint("Finished calculating pixel-level marketsheds")
    if out_file != '':
        meta = img.meta.copy()
        output = output.astype(meta['dtype'])
        with rasterio.open(out_file, 'w', **meta) as outR:
            outR.write_band(1, output)
    return(output)