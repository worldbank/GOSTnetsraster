"""Simplify calculations of travel time over gridded friction surfaces using skimage.graph

TBD description

"""

import time
import copy
import rasterio

import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
import GOSTnets as gn
import skimage.graph as graph
import GOSTrocks.rasterMisc as rMisc

from skimage.graph import _mcp
from rasterio import features
from shapely.geometry import shape
from scipy import sparse
from numpy import inf

from . import conversion_tables as speed_tables

speed_dict = {
    "residential": 20,  # kmph
    "primary": 40,
    "primary_link": 35,
    "motorway": 50,
    "motorway_link": 45,
    "trunk": 40,
    "trunk_link": 35,
    "secondary": 30,
    "secondary_link": 25,
    "tertiary": 30,
    "tertiary_link": 25,
    "unclassified": 20,
    "living_street": 10,
    "service": 10,
}
"""prints the time along with the message"""


def tPrint(s):
    print("%s\t%s" % (time.strftime("%H:%M:%S"), s))


def get_speed(x, s_dict):
    try:
        return s_dict[x]
    except KeyError:
        if isinstance(x, list):
            for speed_key in x:
                if speed_key in s_dict:
                    return s_dict[speed_key]
        return 5


def get_nodes(b, tags):
    nodes = ox.geometries_from_bbox(b[3], b[1], b[2], b[0], tags=tags).reset_index()
    nodes["lat"] = nodes.geometry.y
    nodes["lon"] = nodes.geometry.x
    cols = [
        col for col in ["amenity", "shop", "lat", "lon", "geometry"] if col in nodes
    ]
    return gpd.GeoDataFrame(nodes[cols], geometry="geometry", crs="EPSG:4326")


def get_roads(b):
    sel_graph = ox.graph_from_bbox(b[3], b[1], b[2], b[0], retain_all=True)
    sel_roads = gn.edge_gdf_from_graph(sel_graph)
    sel_roads["speed"] = sel_roads["highway"].apply(lambda x: get_speed(x, speed_dict))
    return sel_roads


def get_mcp_dests(inH, destinations, makeset=True):
    """Get indices from inH for use in mcp.find_costs
    INPUT
        inH[rasterio] - object from which to extract geographic coordinates
        destinations[geopandas geodataframe] - point geodataframe of destinations
    RETURN
        [list of indices]
    """
    if makeset:
        cities = list(set([inH.index(x.x, x.y) for x in destinations["geometry"]]))
    else:
        cities = list([inH.index(x.x, x.y) for x in destinations["geometry"]])

    cities = [
        x
        for x in cities
        if (
            (x[0] > 0)
            and (x[1] > 0)
            and (x[0] <= inH.shape[0])
            and (x[1] <= inH.shape[1])
        )
    ]
    return cities


def name_mcp_dests(inH, destinations):
    """Sometimes multiple destinations fall within the same cell and duplicates are removed before processing in get_mcp_dests.
        This function is designed to create a column in destinations that can be merged with results from get_mcp_dests
    INPUT
        inH[rasterio] - object from which to extract geographic coordinates
        destinations[geopandas geodataframe] - point geodataframe of destinations
    RETURN
        destinations [geopandas] - returns with new column called MCP_DESTS_NAME
    """
    destinations["MCP_DESTS_NAME"] = ""
    for idx, row in destinations.iterrows():
        x = row["geometry"]
        destinations.loc[idx, "MCP_DESTS_NAME"] = "_".join(
            [str(xx) for xx in inH.index(x.x, x.y)]
        )
    return destinations


def generate_roads_lc_friction(
    lc_file,
    sel_roads,
    lc_travel_table=None,
    min_lc_val=0.01,
    min_road_speed=0.01,
    speed_col="speed",
    resolution=100,
    out_file="",
):
    """Combine a landcover dataset and a road network dataset to create
    a friction surface. See generate_lc_friction and generate_road_friction for more details

    Returns
    dictionary of 'meta'[dictionary] and 'friction'[numpy array] - meta is metadata required to write rasterio
        friction is the resulting friction surface used to run in GOSTNets Raster
    """
    if not lc_travel_table:
        lc_travel_table = speed_tables.copernicus_landcover
    lc_friction = generate_lc_friction(
        lc_file,
        lc_travel_table=lc_travel_table,
        min_val=min_lc_val,
        resolution=resolution,
    )
    road_friction = generate_road_friction(
        lc_file,
        sel_roads,
        no_road_speed=min_road_speed,
        speed_col=speed_col,
        resolution=resolution,
    )

    # Stack frictions and find minimum
    stacked_friction = np.dstack([road_friction, lc_friction[0, :, :]])
    combo_friction = np.amin(stacked_friction, axis=2)
    combo_friction[combo_friction == inf] = None
    combo_friction = combo_friction.astype("float")

    out_meta = lc_file.meta.copy()
    out_meta.update(dtype=combo_friction.dtype)

    if out_file != "":
        with rasterio.open(out_file, "w", **out_meta) as outR:
            outR.write_band(1, combo_friction)
    return {"meta": out_meta, "friction": combo_friction}


def generate_lc_friction(lc_file, lc_travel_table=None, min_val=0.01, resolution=100):
    """Convert a landcover dataset to a friction surface based on a table

    Inputs
        lc_file [rasterio] - landcover file
        [optional] lc_travel_table [dictionary] - dictionary of travel speeds per lc class in km/h; defaults
            to conversion_tables.copernicus_landcover
        [optional] min_val [float] - minimum speed for lc dataset

    Returns
        [numpy array] - friction surface describing travel speed per landcover class
    """
    if not lc_travel_table:
        lc_travel_table = speed_tables.copernicus_landcover
    lc_data = lc_file.read()
    res = np.vectorize(lc_travel_table.get)(lc_data)
    res = res.astype(float)
    res = np.nan_to_num(res, nan=min_val)
    res = 1 / ((res * 1000) / 60)  # km/h --> m/min --> minutes/m
    return res


def generate_road_friction(
    inH, sel_roads, no_road_speed=0.01, speed_col="speed", resolution=100
):
    """Create raster with network travel times from a road network that measures seconds to cross a cell

    INPUTS
        inH [rasterio object] - template raster used to define raster shape, resolution, crs, etc.
        sel_roads [geopandas dataframe] - road network to burn into raster
        [optional] no_road_speed [int] - minimum travel speed for areas without roads
        [optional] speed_col [string] - column in sel_roads that defines the speed in KM/h
        [optional] resolution [int] - resolution of the raster in metres

    RETURNS
        [numpy array]
    """
    # burn the speeds into the distance_data using the road network
    sel_roads = sel_roads.sort_values([speed_col])
    shapes = ((row["geometry"], row[speed_col]) for idx, row in sel_roads.iterrows())
    res = features.rasterize(
        shapes, out_shape=inH.shape, transform=inH.transform, fill=no_road_speed
    )
    # convert to a version that calculates the seconds to cross each cell
    res = 1 / ((res * 1000) / 60)  # km/h --> m/min --> minutes/m
    return res


def calculate_travel_time(inH, mcp, destinations, out_raster=""):
    """Calculate travel time from all cells to the set of destinations using an MCP graph

    INPUTS
        inH [rasterio] - raster from which to grab index for calculations in MCP
        mcp [skimage.graph.MCP_Geometric] - input graph
        destinations [geopandas data frame] - geopandas data frame of destinations
        out_raster [string] - optional path to write travel time raster
    RETURNS
        (costs [numpy array], traceback [numpy array])
    """

    # create skimage graph
    cities = get_mcp_dests(inH, destinations)
    costs, traceback = mcp.find_costs(cities)
    if not out_raster == "":
        meta = inH.meta.copy()
        meta.update(dtype=costs.dtype)
        with rasterio.open(out_raster, "w", **meta) as out:
            out.write_band(1, costs)

    return (costs, traceback)


def get_all_amenities(bounds):
    amenities = ["toilets", "washroom", "restroom"]
    toilets_tags = '"amenity"~"{}"'.format("|".join(amenities))
    toilets = get_nodes(bounds, toilets_tags)

    amenities = ["water_points", "drinking_water", "pumps", "water_pumps", "well"]
    water_tags = '"amenity"~"{}"'.format("|".join(amenities))
    water_points = get_nodes(bounds, water_tags)

    amenities = [
        "supermarket",
        "convenience",
        "general",
        "department_stores",
        "wholesale",
        "grocery",
        "general",
    ]
    shp_tags = '"shop"~"{}"'.format("|".join(amenities))
    shops = get_nodes(bounds, shp_tags)

    return {"toilets": toilets, "water_points": water_points, "shops": shops}


def generate_feature_vectors(
    network_r, mcp, inH, threshold, featIdx="tempID", verbose=True
):
    """Generate individual market sheds for each feature in the input dataset

    INPUTS
        network_r [rasterio] - raster from which to grab index for calculations in MCP
        mcp [skimage.graph.MCP_Geometric] - input graph
        inH [geopandas data frame] - geopandas data frame from which to calculate features
        threshold [list of int] - travel threshold from which to calculate vectors in units of graph
        featIdx [string] - column name in inH to append to output marketshed dataset. 'tempID' for default.

    RETURNS
        [geopandas dataframe]
    """
    n = inH.shape[0]
    inH["tempID"] = inH.index
    # Create temporary index (tempID) in the original shape to be fed to 'featIdx.'

    feat_count = 0
    complete_shapes = []

    for idx, row in inH.iterrows():
        feat_count = feat_count + 1
        if verbose:
            tPrint(f"{feat_count} of {n}: {row[featIdx]}")
        cur_idx = network_r.index(row["geometry"].x, row["geometry"].y)
        # Retrieve a xy coordinate of the target point shape from 'geometry' column (at row idx).
        if (
            cur_idx[0] > 0
            and cur_idx[1] > 0
            and cur_idx[0] < network_r.shape[0]
            and cur_idx[1] < network_r.shape[1]
        ):
            costs, traceback = mcp.find_costs([cur_idx])
            # Checking the validity of cur_idx (Row x Column) - they must be positive and within the shape of the target raster.

            for thresh in threshold:
                # The 2nd iteration loop for the threshold value in minutes (e.g. 60, 120, 180, 240) specified by 'threshold' variable.

                within_time = ((costs < thresh) * 1).astype("int16")
                within_time_mask = within_time > 0
                # Masking cells in the raster that are less than the selected threshold value by inserting 1 (int16).
                # The cells larger than the 'thresh' value are masked by 0 (int16).
                # within_time numpy array will be transformed into boolean type by 'within_time > 0'
                # to be used in the mask of 'features.shapes' below to exclude features (cells) with False (=0).

                all_shapes = []  # Creating an empty list to store shapes.

                polyCount = 0
                for poly, value in features.shapes(
                    within_time, mask=within_time_mask, transform=network_r.transform
                ):
                    # The 3rd iteration loop for retrieving a pair of polygon and value for each feature found in the raster image.
                    polyCount += 1
                    all_shapes.append(shape(poly))

                shape_df = pd.DataFrame([list(range(len(all_shapes))), all_shapes]).T
                shape_df.columns = ["ID", "geometry"]
                shape_df = gpd.GeoDataFrame(
                    shape_df, geometry="geometry", crs=network_r.crs
                )

                union = shape_df.unary_union
                complete_shapes.append([union, thresh, row[featIdx]])

    final = gpd.GeoDataFrame(
        complete_shapes, columns=["geometry", "threshold", "IDX"], crs=network_r.crs
    )
    return final


def generate_market_sheds(
    inR,
    inH,
    out_file="",
    verbose=True,
    factor=1000,
    bandIdx=0,
    column_id=None,
    reclass=True,
):
    """identify pixel-level maps of market sheds based on travel time
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
    """
    xx = inR.read()[bandIdx, :, :] * factor
    orig_shape = xx.shape
    # In order to calculate the marketsheds, the input array needs to be NxN shape,
    #   at the end, we will select out the original shape in order to write to file
    max_speed = xx.max()
    xx[xx < 0] = max_speed  # untraversable
    if xx.shape[0] < xx.shape[1]:
        extra_size = np.zeros([(xx.shape[1] - xx.shape[0]), xx.shape[1]]) + max_speed
        xx = np.vstack([xx, extra_size])

    if xx.shape[1] < xx.shape[0]:
        extra_size = np.zeros([xx.shape[0], (xx.shape[0] - xx.shape[1])]) + max_speed
        xx = np.hstack([xx, extra_size])
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
    neighbor_ids = np.ravel_multi_index(tuple(neighbor_index), traceback.shape)
    g = sparse.coo_matrix(
        (
            np.ones(traceback.size),
            (ids.flat, neighbor_ids.flat),
        ),
        shape=[traceback.size, traceback.size],
    ).tocsr()
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
                basins_reclass[basins == basins_value] = destinations_ids[i]
            else:
                basins_reclass[basins == basins_value] = i
            # print(f"Reclassify {basins_value} to {i}

        basins = basins_reclass.copy()

    out_basins = basins[: orig_shape[0], : orig_shape[1]]
    if out_file != "":
        meta = inR.meta.copy()
        meta.update(dtype=out_basins.dtype)
        with rasterio.open(out_file, "w", **meta) as out_raster:
            out_raster.write_band(1, out_basins)
    else:
        return out_basins


def generate_market_sheds_old(img, mcp, inH, out_file="", verbose=True):
    """identify pixel-level maps of market sheds based on travel time

    INPUTS
        network_r [rasterio] - raster from which to grab index for calculations in MCP
        mcp [skimage.graph.MCP_Geometric] - input graph
        inH [geopandas data frame] - geopandas data frame from which to calculate features

    RETURNS
        [numpy array]
    """
    dests_geom = [img.index(x.x, x.y) for x in inH["geometry"]]
    all_c = []
    n = inH.shape[0]
    idx = 0
    for dest in dests_geom:
        idx += 1
        if (
            dest[0] > 0
            and dest[0] < img.shape[0]
            and dest[1] > 0
            and dest[1] < img.shape[1]
        ):
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
    """
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
    """
    if verbose:
        tPrint("Finished calculating pixel-level marketsheds")
    if out_file != "":
        meta = img.meta.copy()
        output = output.astype(meta["dtype"])
        with rasterio.open(out_file, "w", **meta) as outR:
            outR.write_band(1, output)
    return output


def summarize_travel_time_populations(
    popR,
    ttR,
    dests,
    mcp,
    zonalD,
    out_tt_file="",
    calc_thresh=True,
    calc_weighted=True,
    thresholds=[30, 60, 120, 180, 240],
):
    """Summarize the population according to travel time within a set of thresholds

    Args:
        popR:          rasterio object describing the population data
        dests:         geopandas.GeoDataFrame describing the destinations for the travel time calculation
        ttR:           rasterio object describing the travel time raster
        mcp:           skimage.graph object created from ttR
        zonalD:        geopandas.GeoDataFrane of zones for summarizing population
        out_tt_file:   Optional; path to geotiff to output travel time results
        calc_thresh:   Optional; calculate population within travel time thresholds
        calc_weighted: Optional; calculate population weighted travel time
        thresholds:    Optional; travel times at which to summarize population
    Returns:
        A geopandas.GeoDataFrame with extra columns describing the population within traveltime thresholds
    """
    # Check inputs
    if popR.crs != ttR.crs:
        raise (
            ValueError("Population and Travel time must have matching CRS and shape")
        )
    if popR.shape != ttR.shape:
        popD, profile = rMisc.standardizeInputRasters(popR, ttR)
    else:
        popD = popR.read()
    if popR.crs.to_epsg() != zonalD.crs.to_epsg():
        zonalD = zonalD.to_crs(popR.crs)
    if popR.crs.to_epsg() != dests.crs.to_epsg():
        dests = dests.to_crs(popR.crs)

    res = rMisc.zonalStats(zonalD, popR, minVal=0)
    res = pd.DataFrame(res, columns=["SUM", "MIN", "MAX", "MEAN"])
    zonalD["total_pop"] = res["SUM"].values

    # calculate travel time to destinations
    ttD, traceback = calculate_travel_time(ttR, mcp, dests)
    if out_tt_file != "":
        ttD = ttD.astype(ttR.meta["dtype"])
        with rasterio.open(out_tt_file, "w", **ttR.meta) as outR:
            outR.write_band(1, ttD)

    # Calculate population within thresholds
    if calc_thresh:
        for thresh in thresholds:
            cur_ttD = ttD <= thresh
            cur_popD = popD * cur_ttD
            with rMisc.create_rasterio_inmemory(popR.profile, cur_popD) as cur_popR:
                res = rMisc.zonalStats(zonalD, cur_popR, minVal=0)
                res = pd.DataFrame(res, columns=["SUM", "MIN", "MAX", "MEAN"])
                zonalD[f"pop_{thresh}"] = res["SUM"].values

    # Calculate population weighted travel time
    if calc_weighted:
        # Calculate total population in each zone
        with rMisc.create_rasterio_inmemory(popR.profile, popD) as temp_popR:
            pop_res = rMisc.zonalStats(zonalD, temp_popR, minVal=0)
            pop_res = pd.DataFrame(pop_res, columns=["SUM", "MIN", "MAX", "MEAN"])
            zonalD["total_pop"] = pop_res["SUM"].values
        # combine travel time and population
        tt_pop = ttD * popD
        tt_pop = np.nan_to_num(tt_pop)
        # return(tt_pop)
        with rMisc.create_rasterio_inmemory(popR.profile, tt_pop) as temp_ttPopR:
            pop_res = rMisc.zonalStats(zonalD, temp_ttPopR, minVal=0, maxVal=1000000000)
            pop_res = pd.DataFrame(pop_res, columns=["SUM", "MIN", "MAX", "MEAN"])
            zonalD["tt_pop_w"] = pop_res["SUM"].values

        def get_weighted_tt_pop(zonalD):
            total_pop = zonalD.get("total_pop", 0)
            if total_pop:
                return zonalD.get("tt_pop_w", 0) / total_pop
            return 0

        zonalD["tt_pop_w"] = zonalD.apply(lambda x: get_weighted_tt_pop(x), axis=1)

    return zonalD


def calculate_gravity(
    inH,
    mcp,
    dests,
    gravity_col,
    outfile="",
    decayVals=[
        0.01,
        0.005,
        0.001,
        0.0007701635,  # Market access halves every 15 mins
        0.0003850818,  # Market access halves every 30 mins
        0.0001925409,  # Market access halves every 60 mins
        0.0000962704,  # Market access halves every 120 mins
        0.0000385082,  # Market access halves every 300 mins
        0.00001,
    ],
):
    """Using a friction surface, run a gravity model to evaluate access to all cities

    Parameters
    ----------
    inH : rasterio object
        rasterio object of friction surface
    mcp : skimage.graph.MCP_Geometric
        graph used to calculate travel times; must match shape of inH
    dests : geopandas dataframe
        Destination coordinates for gravity model, must be in same CRS as inH
    gravity_col : string
        column in dests that describes the attractiveness of the destination
    outfile : string, optional
        Path to save the gravity model output. Defaults to '' which writes nothing
    decayVals : list, optional
        List of decay values for market access. Defaults to a predefined list.

    Returns
    -------
    final_gravity : numpy array
        3D array of gravity model results, with the third dimension corresponding to decay values
    """

    ### ToDo: Add in a check to ensure that the CRS of inH and dests match
    def decayFunction(x, decay):
        return np.exp(-1 * decay * x)

    all_costs = []
    for idx, row in dests.iterrows():
        # for each destination, calculate travel time
        cur_costs, cur_traceback = calculate_travel_time(
            inH, mcp, gpd.GeoDataFrame([row], geometry="geometry", crs=dests.crs)
        )
        all_costs.append(cur_costs.copy())
    costs = np.dstack(all_costs)

    # Iterate through stack of calculations and calculate gravity
    final_gravity = None
    gravity_idx = 0
    for decay in decayVals:
        gravity_idx += 1
        cur_gravity = np.zeros([costs.shape[0], costs.shape[1]])
        for row in range(costs.shape[0]):
            for col in range(costs.shape[1]):
                cur_gravity[row, col] = np.sum(
                    decayFunction(costs[row, col, :], decay) * dests[gravity_col].values
                )
        if final_gravity is None:
            final_gravity = cur_gravity
        else:
            final_gravity = np.dstack([final_gravity, cur_gravity])

    if outfile != "":
        out_meta = inH.meta.copy()
        if final_gravity.ndim == 3:
            out_meta.update(dtype=final_gravity.dtype, count=final_gravity.shape[2])
            with rasterio.open(outfile, "w", **out_meta) as outR:
                for band_idx in range(final_gravity.shape[2]):
                    outR.write_band(band_idx + 1, final_gravity[:, :, band_idx])
        else:
            with rasterio.open(outfile, "w", **out_meta) as outR:
                outR.write_band(1, final_gravity)

    return {"costs": costs, "gravity": final_gravity}
