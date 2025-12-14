# market_access.py

## `get_mcp_dests(inH, destinations, makeset=True)`

**Description:**
Get indices from the raster for use in `mcp.find_costs`.

**Parameters:**

- `inH` (rasterio): Object from which to extract geographic coordinates.
- `destinations` (geopandas GeoDataFrame): Point dataset of destinations.
- `makeset` (bool): When true, deduplicate destinations sharing a cell.

**Returns:**

- list: Indices suitable for MCP processing.

---

## `name_mcp_dests(inH, destinations)`

**Description:**
Label destinations with a name matching MCP indices when duplicates are removed.

**Parameters:**

- `inH` (rasterio): Raster used to derive cell indices.
- `destinations` (geopandas GeoDataFrame): Point destinations to label.

**Returns:**

- geopandas GeoDataFrame: Input destinations with `MCP_DESTS_NAME` column added.

---

## `generate_roads_lc_friction(lc_file, sel_roads, lc_travel_table=None, min_lc_val=0.01, min_road_speed=0.01, speed_col='speed', resolution=100, out_file='')`

**Description:**
Combine a landcover dataset and a road network to create a friction surface.

**Parameters:**

- `lc_file` (rasterio): Landcover file.
- `sel_roads` (geopandas GeoDataFrame): Road network to burn into the raster.
- `lc_travel_table` (dict | None): Travel speeds per landcover class in km/h; defaults to `conversion_tables.copernicus_landcover`.
- `min_lc_val` (float): Minimum speed for landcover cells.
- `min_road_speed` (float): Minimum travel speed for areas without roads.
- `speed_col` (str): Column in `sel_roads` defining speed in km/h.
- `resolution` (int): Raster resolution in metres.
- `out_file` (str): Optional path to write the combined friction surface.

**Returns:**

- dict: Dictionary with `meta` (raster metadata) and `friction` (numpy array).

---

## `generate_lc_friction(lc_file, lc_travel_table=None, min_val=0.01, resolution=100)`

**Description:**
Convert a landcover dataset to a friction surface based on a travel speed table.

**Parameters:**

- `lc_file` (rasterio): Landcover file.
- `lc_travel_table` (dict | None): Travel speeds per landcover class in km/h; defaults to `conversion_tables.copernicus_landcover`.
- `min_val` (float): Minimum speed for the landcover dataset.
- `resolution` (int): Resolution of the raster in metres.

**Returns:**

- numpy array: Friction surface describing travel speed per landcover class.

---

## `generate_road_friction(inH, sel_roads, no_road_speed=0.01, speed_col='speed', resolution=100)`

**Description:**
Create a raster of network travel times from a road network measured as seconds to cross each cell.

**Parameters:**

- `inH` (rasterio): Template raster defining shape, resolution, and CRS.
- `sel_roads` (geopandas GeoDataFrame): Road network to burn into the raster.
- `no_road_speed` (int): Minimum travel speed for areas without roads.
- `speed_col` (str): Column in `sel_roads` defining speed in km/h.
- `resolution` (int): Resolution of the raster in metres.

**Returns:**

- numpy array: Friction surface derived from the road network.

---

## `calculate_travel_time(inH, mcp, destinations, out_raster='')`

**Description:**
Calculate travel time from all cells to the set of destinations using an MCP graph.

**Parameters:**

- `inH` (rasterio): Raster from which to grab indices for MCP calculations.
- `mcp` (skimage.graph.MCP_Geometric): Input graph for travel time.
- `destinations` (geopandas GeoDataFrame): Destination points.
- `out_raster` (str): Optional path to write the travel time raster.

**Returns:**

- tuple: `(costs, traceback)` arrays from MCP results.

---

## `generate_feature_vectors(network_r, mcp, inH, threshold, featIdx='tempID', verbose=True)`

**Description:**
Generate individual market sheds for each feature in the input dataset.

**Parameters:**

- `network_r` (rasterio): Raster used for MCP indexing.
- `mcp` (skimage.graph.MCP_Geometric): Graph for travel time computation.
- `inH` (geopandas GeoDataFrame): Features from which to calculate sheds.
- `threshold` (list[int]): Travel thresholds for vector generation.
- `featIdx` (str): Column name in `inH` to include in the output.
- `verbose` (bool): Print progress during processing.

**Returns:**

- geopandas GeoDataFrame: Market shed geometries with thresholds and IDs.

---

## `generate_market_sheds(inR, inH, out_file='', verbose=True, factor=1000, bandIdx=0, column_id=None, reclass=True)`

**Description:**
Identify pixel-level maps of market sheds based on travel time.

**Parameters:**

- `inR` (rasterio): Raster used for MCP indexing.
- `inH` (geopandas GeoDataFrame): Destinations for market sheds.
- `out_file` (str): Optional path to write the market shed raster.
- `verbose` (bool): Print progress during processing.
- `factor` (int): Multiplier applied to the raster before processing.
- `bandIdx` (int): Band index to use from the raster.
- `column_id` (int | None): Column with unique identifiers in `inH` to preserve.
- `reclass` (bool): When true, remap sheds to their original identifiers.

**Returns:**

- numpy array: Market shed labels by raster cell.

---

## `generate_market_sheds_old(img, mcp, inH, out_file='', verbose=True)`

**Description:**
Identify pixel-level maps of market sheds based on travel time (legacy approach).

**Parameters:**

- `img` (rasterio): Raster used for MCP indexing.
- `mcp` (skimage.graph.MCP_Geometric): Graph for travel time computation.
- `inH` (geopandas GeoDataFrame): Destinations for market sheds.
- `out_file` (str): Optional path to write the market shed raster.
- `verbose` (bool): Print progress during processing.

**Returns:**

- numpy array: Market shed labels by raster cell.

---

## `summarize_travel_time_populations(popR, ttR, dests, mcp, zonalD, out_tt_file='', calc_thresh=True, calc_weighted=True, thresholds=[30, 60, 120, 180, 240])`

**Description:**
Summarize population according to travel time thresholds and weighted travel time.

**Parameters:**

- `popR` (rasterio): Population raster.
- `ttR` (rasterio): Travel time raster.
- `dests` (geopandas GeoDataFrame): Destinations for travel time calculation.
- `mcp` (skimage.graph.MCP_Geometric): Graph created from `ttR`.
- `zonalD` (geopandas GeoDataFrame): Zones for summarizing population.
- `out_tt_file` (str): Optional path to output travel time results.
- `calc_thresh` (bool): Calculate population within travel time thresholds.
- `calc_weighted` (bool): Calculate population-weighted travel time.
- `thresholds` (list[int]): Travel times at which to summarize population.

**Returns:**

- geopandas GeoDataFrame: Zones with additional population and travel time metrics.

---

## `calculate_gravity(inH, mcp, dests, gravity_col, outfile='', decayVals=[0.01, 0.005, 0.001, 0.0007701635, 0.0003850818, 0.0001925409, 0.0000962704, 0.0000385082, 0.00001])`

**Description:**
Run a gravity model over a friction surface to evaluate access to destinations.

**Parameters:**

- `inH` (rasterio object): Friction surface used for travel time.
- `mcp` (skimage.graph.MCP_Geometric): Graph matching the friction surface.
- `dests` (geopandas GeoDataFrame): Destinations with attractiveness values.
- `gravity_col` (str): Column in `dests` describing destination attractiveness.
- `outfile` (str): Path to save gravity model output; defaults to no write.
- `decayVals` (list[float]): Decay values controlling gravity decay with travel time.

**Returns:**

- numpy array: Gravity model results per decay value.

---
