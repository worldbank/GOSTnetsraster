""" Simplify calculations of travel time over gridded friction surfaces using skimage.graph

TBD description

"""
import sys, os, importlib, time, copy
import rasterio

import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
import GOSTnets as gn
import skimage.graph as graph
import GOSTRocks.rasterMisc as rMisc

from skimage.graph import _mcp
from rasterio.mask import mask
from rasterio import features
from shapely.geometry import box, Point, shape
from shapely.wkt import loads
from shapely.ops import cascaded_union
from scipy import sparse
from scipy.ndimage import generic_filter
from pandana.loaders import osm
from numpy import inf      

from . import conversion_tables as speed_tables

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

def generate_roads_lc_friction(lc_file, sel_roads, lc_travel_table=None, min_lc_val=0.01, min_road_speed=0.01, speed_col='speed', resolution=100, out_file = ''):
    ''' Combine a landcover dataset and a road network dataset to create
        a friction surface. See generate_lc_friction and generate_road_friction for more details
        
        Returns
        dictionary of 'meta'[dictionary] and 'friction'[numpy array] - meta is metadata required to write rasterio
            friction is the resulting friction surface used to run in GOSTNets Raster
    '''
    if not lc_travel_table:
        lc_travel_table = speed_tables.copernicus_landcover
    lc_friction = generate_lc_friction(lc_file, lc_travel_table = lc_travel_table, min_val = min_lc_val, resolution = resolution)
    road_friction = generate_network_raster(lc_file, sel_roads, min_speed=min_road_speed, speed_col=speed_col, resolution=resolution)
    
    #Stack frictions and find minimum
    stacked_friction = np.dstack([road_friction, lc_friction[0,:,:]])    
    combo_friction = np.amin(stacked_friction, axis=2)
    combo_friction[combo_friction == inf] = 65000
    
    out_meta = lc_file.meta.copy()
    combo_friction = combo_friction.astype('uint16')
    out_meta['dtype'] = combo_friction.dtype
    
    if out_file != '':
        with rasterio.open(out_file, 'w', **out_meta) as outR:
            outR.write_band(1, combo_friction)
    return({'meta': out_meta, 'friction': combo_friction})
        

def generate_lc_friction(lc_file, lc_travel_table=None, min_val=0.01, resolution=100):
    ''' Convert a landcover dataset to a friction surface based on a table
    
    Inputs
        lc_file [rasterio] - landcover file
        [optional] lc_travel_table [dictionary] - dictionary of travel speeds per lc class in km/h; defaults
            to conversion_tables.copernicus_landcover
        [optional] min_val [float] - minimum speed for lc dataset
    
    Returns
        [numpy array] - friction surface describing travel speed per landcover class
    '''
    if not lc_travel_table:
        lc_travel_table = speed_tables.copernicus_landcover
    lc_data = lc_file.read()
    res = np.vectorize(lc_travel_table.get)(lc_data)
    res = res.astype(float)
    res = np.nan_to_num(res, nan=0.01)
    res = resolution / (res * 1000 / (60 * 60)) # km/h --> m/s * resolution of image in metres
    return(res)
    
def generate_road_friction(inH, sel_roads, min_speed=0.01, speed_col='speed', resolution=100):   
    ''' Create raster with network travel times from a road network that measures seconds to cross a cell
    
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
        featIdx [string] - column name in inH to append to output marketshed dataset. 'tempID' for default.
        
    RETURNS
        [geopandas dataframe]
    '''
    n = inH.shape[0]
    inH['tempID'] = inH.index
    #Create temporary index (tempID) in the original shape to be fed to 'featIdx.'
   
    feat_count = 0
    complete_shapes = []
   
    for idx, row in inH.iterrows():
    # The 1st iteration loop for each target point.
   
        feat_count = feat_count + 1
      
        if verbose:
            tPrint(f"{feat_count} of {n}: {row[featIdx]}")
            # This is just to genereate a progress message.
            
        cur_idx = network_r.index(row['geometry'].x, row['geometry'].y)  
        # Retrive a xy coordinate of the target point shape from 'geometry' column (at row idx).
        # And retrive the rc location at the target network raster that is corresponding to the xy cooridnate.

        if cur_idx[0] > 0 and cur_idx[1] > 0 and cur_idx[0] < network_r.shape[0] and cur_idx[1] < network_r.shape[1]:
            costs, traceback = mcp.find_costs([cur_idx])
            # Checking the validity of cur_idx (Row x Column) - they must be postive and within the shape of the target raster.

            for thresh in threshold:
            # The 2nd iteration loop for the threshold value in minutes (e.g. 60, 120, 180, 240) specified by 'threshold' variable.

                within_time = ((costs < thresh) * 1).astype('int16')
                within_time_mask = within_time > 0
                # Masking cells in the raster that are less than the selected threshold value by inserting 1 (int16).
                # The cells larger than the 'thresh' value are masked by 0 (int16).
                # within_time numpy array will be trasnformed into boolean type by 'within_time > 0'
                # to be used in the mask of 'features.shapes' below to exclude features (cells) with False (=0).

                all_shapes = []# Creating an empty list to store shapes.
                
                polyCount = 0
                for poly, value in features.shapes(within_time, mask = within_time_mask, transform = network_r.transform):
                # The 3rd iteration loop for retriving a pair of polygon and value for each feature found in the raster image.
                    polyCount += 1
                    all_shapes.append([shape(poly)])

                # Convert 'all_shapes' list to 'gs' GeoSeries.
                gs = gpd.GeoSeries()#Create an empty GeoSeries.
            
                for x in all_shapes:
                	gsTemp = gpd.GeoSeries(x)
                	gs = gs.append(gsTemp)
            
                gs = gs.reset_index(drop=True)#Reset the index of 'gs'
            
        
            	# Geometry validity check and correction
                for i, geom in enumerate(gs):
                    geomCheck = geom.is_valid
                    if geomCheck == False:
                        gs[i] = geom.buffer(0)
                    	# Conventional way to correct invalid geometry.
                    	# Not sure how this create difference b/w the non-corrected and the corrected.
            
            	# Geometry validity double-check (inform the result to the user)
                # This code block can be omitted if it's redundant.
                for i, geom in enumerate(gs):
                	print('Geometry No. {}: Validity = {} | THS = {} minutes | Poly Count = {}'.format(i, geom.is_valid, thresh, polyCount))

                union = gs.unary_union
                complete_shapes.append([union, thresh, row[featIdx]])
				
    final = gpd.GeoDataFrame(complete_shapes, columns=["geometry", "threshold", "IDX"], crs=network_r.crs)
    return(final)
    
def generate_market_sheds(inR, inH, out_file='', verbose=True, factor=1000, bandIdx=0, column_id=None, reclass=True):
    ''' identify pixel-level maps of market sheds based on travel time    
    INPUTS
        inR [rasterio] - raster from which to grab index for calculations in MCP
        inH [geopandas data frame] - geopandas data frame of destinations
        factor [int] - value by which to multiply raster 
        column_id [int] - column with unique identifiers in inH
        reclass [boolean] - if True, sheds will be remapped to their original index (or column_id value). If False (old default), code generates a new index for sheds based on the order of the array.
        
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
    xx[xx < 0] = max_speed # untraversable
    if xx.shape[0] < xx.shape[1]:
        extra_size = np.zeros([(xx.shape[1] - xx.shape[0]), xx.shape[1]]) + max_speed
        new_xx = np.vstack([xx, extra_size])
        
    if xx.shape[1] < xx.shape[0]:
        extra_size = np.zeros([xx.shape[0], (xx.shape[0] - xx.shape[1])]) + max_speed
        new_xx = np.hstack([xx, extra_size])        
    mcp = graph.MCP_Geometric(xx)
    
    dests = get_mcp_dests(inR, inH, makeset=False)
    if column_id:
        destinations_ids = list(inH[column_id])
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
    
    # get original index
    if reclass:
        dest_idx = []
        for dest_coords in dests:
            dest_id = neighbor_ids[dest_coords[0], dest_coords[1]]
            dest_idx.append(dest_id)
        
        basins_reclass = basins.copy()
        for i, dest in enumerate(dests):
            basins_value = basins[dest[0], dest[1]]
            if column_id:
                basins_reclass[basins==basins_value] = destinations_ids[i]
            else:
                basins_reclass[basins==basins_value] = i
            # print(f"Reclassify {basins_value} to {i}
        
        basins = basins_reclass.copy()
    
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
    
def summarize_travel_time_populations(popR, ttR, dests, mcp, zonalD, out_tt_file='', calc_thresh=True, calc_weighted=True, thresholds=[30,60,120,180,240]):
    ''' Summarize the population according to travel time within a set of thresholds
    
    Args:
        popR:          rasterio object describing the population data
        dests:         geopandas.GeoDataFrame describing the destinations for the travel time calculation
        ttR:           rasterio obejct describing the travel time raster
        mcp:           skimage.graph object created from ttR
        zonalD:        geopandas.GeoDataFrane of zones for summarizing population
        out_tt_file:   Optional; path to geotiff to output travle time results
        calc_thresh:   Optional; calculate population within travel time thresholds
        calc_weighted: Optional; calculate population weighted travel time
        thresholds:    Optional; travel times at which to summarize population
    Returns:
        A geopandas.GeoDataFrame with extra columns describing the population within traveltime thresholds
    '''
    # Check inputs
    if popR.crs != ttR.crs:
        raise(ValueError("Population and Travel time must have matching CRS and shape"))
    if popR.shape != ttR.shape:
        popD, profile = rMisc.standardizeInputRasters(popR, ttR)       
    else:
        popD = popR.read()           
    if popR.crs.to_epsg() != zonalD.crs.to_epsg():
        zonalD = zonalD.to_crs(popR.crs)
    if popR.crs.to_epsg() != dests.crs.to_epsg():
        dests = dests.to_crs(popR.crs)
    
    res = rMisc.zonalStats(zonalD, popR, minVal=0)
    res = pd.DataFrame(res, columns=['SUM','MIN','MAX','MEAN'])
    zonalD[f'total_pop'] = res['SUM']    
    
    # calculate travel time to destinations
    ttD, traceback = calculate_travel_time(ttR, mcp, dests)   
    if out_tt_file != '':
        ttD = ttD.astype(ttR.meta['dtype'])
        with rasterio.open(out_tt_file, 'w', **ttR.meta) as outR:
            outR.write_band(1, ttD)
    
    # Calculate population within thresholds    
    if calc_thresh:
        for thresh in thresholds:
            cur_ttD = ttD <= thresh
            cur_popD = popD * cur_ttD        
            with rMisc.create_rasterio_inmemory(popR.profile, cur_popD) as cur_popR:
                res = rMisc.zonalStats(zonalD, cur_popR, minVal=0)
                res = pd.DataFrame(res, columns=['SUM','MIN','MAX','MEAN'])
                zonalD[f'pop_{thresh}'] = res['SUM']
            
    # Calculate population weighted travel time
    if calc_weighted:
        # Calculate total population in each zone
        with rMisc.create_rasterio_inmemory(popR.profile, popD) as temp_popR:
            pop_res = rMisc.zonalStats(zonalD, temp_popR, minVal=0)
            pop_res = pd.DataFrame(pop_res, columns=['SUM','MIN','MAX','MEAN'])
            zonalD['total_pop'] = pop_res['SUM']
        # combine travel time and population
        tt_pop = ttD * popD
        tt_pop = np.nan_to_num(tt_pop)
        #return(tt_pop)
        with rMisc.create_rasterio_inmemory(popR.profile, tt_pop) as temp_ttPopR:
            pop_res = rMisc.zonalStats(zonalD, temp_ttPopR, minVal=0, maxVal=1000000000)
            pop_res = pd.DataFrame(pop_res, columns=['SUM','MIN','MAX','MEAN'])
            zonalD['tt_pop_w'] = pop_res['SUM']            
        zonalD['tt_pop_w'] = zonalD.apply(lambda x: x['tt_pop_w']/x['total_pop'], axis=1)
            
    return(zonalD)
        
        
        
        
        
        
