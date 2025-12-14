# Highway features are reclassified to 4 OSMLR classes for simplification and standardization
#   https://mapzen.com/blog/osmlr-2nd-technical-preview/
OSMLR_Classes = {
    "motorway": "OSMLR level 1",
    "motorway_link": "OSMLR level 1",
    "trunk": "OSMLR level 1",
    "trunk_link": "OSMLR level 1",
    "primary": "OSMLR level 1",
    "primary_link": "OSMLR level 1",
    "secondary": "OSMLR level 2",
    "secondary_link": "OSMLR level 2",
    "tertiary": "OSMLR level 2",
    "tertiary_link": "OSMLR level 2",
    "unclassified": "OSMLR level 3",
    "unclassified_link": "OSMLR level 3",
    "residential": "OSMLR level 3",
    "residential_link": "OSMLR level 3",
    "track": "OSMLR level 4",
    "service": "OSMLR level 4",
}

osm_speed_dict = {
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
modis_umd = {  # https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MCD12Q1#bands
    0: 0.50,  # Water Bodies: at least 60% of area is covered by permanent water bodies
    1: 3.0,  # 05450a Evergreen Needleleaf Forests: dominated by evergreen conifer trees (canopy >2m). Tree cover >60%.
    2: 4.0,  # 086a10 Evergreen Broadleaf Forests: dominated by evergreen broadleaf and palmate trees (canopy >2m). Tree cover >60%.
    3: 3.0,  # 54a708 Deciduous Needleleaf Forests: dominated by deciduous needleleaf (larch) trees (canopy >2m). Tree cover >60%.
    4: 4.0,  # 78d203 Deciduous Broadleaf Forests: dominated by deciduous broadleaf trees (canopy >2m). Tree cover >60%.
    5: 3.5,  # 009900 Mixed Forests: dominated by neither deciduous nor evergreen (40-60% of each) tree type (canopy >2m). Tree cover >60%.
    6: 3.50,  # c6b044 Closed Shrublands: dominated by woody perennials (1-2m height) >60% cover.
    7: 4.20,  # dcd159 Open Shrublands: dominated by woody perennials (1-2m height) 10-60% cover.
    8: 3.50,  # dade48 Woody Savannas: tree cover 30-60% (canopy >2m).
    9: 4.20,  # fbff13 Savannas: tree cover 10-30% (canopy >2m).
    10: 4.20,  # b6ff05	Grasslands: dominated by herbaceous annuals (<2m).
    11: 1.50,  # 27ff87	Permanent Wetlands: permanently inundated lands with 30-60% water cover and >10% vegetated cover.
    12: 4.0,  # c24f44	Cropland.
    13: 5.0,  # a5a5a5	Urban and Built-up Lands: at least 30% impervious surface area including building materials, asphalt and vehicles.
    14: 4.20,  # ff6d4c	Cropland/Natural Vegetation Mosaics: mosaics of small-scale cultivation 40-60% with natural tree, shrub, or herbaceous vegetation.
    15: 4.50,  # f9ffa4	Non-Vegetated Lands: at least 60% of area is non-vegetated barren (sand, rock, soil) or permanent snow and ice with less than 10% vegetation.
}

esaacci_landcover = {
    0: 0.1,  # No data
    10: 2.50,  # Cropland, rainfed
    11: 4.00,  # ???Herbaceous cover
    12: 4.20,  # ???Tree or shrub cover
    20: 3.24,  # Cropland, irrigated or post-flooding
    30: 2.00,  # Mosaic cropland (>50%) / natural vegetation (tree, shrub, herbaceous cover) (<50%)
    40: 3.24,  # Mosaic natural vegetation (tree, shrub, herbaceous cover) (>50%) / cropland (<50%)
    50: 4.00,  # Tree cover, broadleaved, evergreen, closed to open (>15%)
    60: 3.50,  # Tree cover, broadleaved, deciduous, closed to open (>15%)
    61: 3.00,  # Tree cover, broadleaved, deciduous, closed (>40%)
    62: 2.50,  # Tree cover, broadleaved, deciduous, open (15-40%)
    70: 3.50,  # Tree cover, needleleaved, evergreen, closed to open (>15%)
    71: 3.00,  # Tree cover, needleleaved, evergreen, closed (>40%)
    72: 3.24,  # Tree cover, needleleaved, evergreen, open (15-40%)
    80: 3.50,  # Tree cover, needleleaved, deciduous, closed to open (>15%)
    81: 3.00,  # Tree cover, needleleaved, deciduous, closed (>40%)
    82: 3.24,  # Tree cover, needleleaved, deciduous, open (15-40%)
    90: 3.24,  # Tree cover, mixed leaf type (broadleaved and needleleaved)
    100: 3.00,  # Mosaic tree and shrub (>50%) / herbaceous cover (<50%)
    110: 3.00,  # Mosaic herbaceous cover (>50%) / tree and shrub (<50%)
    120: 4.20,  # Shrubland
    121: 3.24,  # Shrubland evergreen
    122: 3.24,  # Shrubland deciduous
    130: 4.86,  # Grassland
    140: 0.00,  # ???Lichens and mosses
    150: 4.20,  # Sparse vegetation (tree, shrub, herbaceous cover) (<15%)
    151: 4.00,  # Sparse tree (<15%)
    152: 4.00,  # Sparse shrub (<15%)
    153: 4.00,  # Sparse herbaceous cover (<15%)
    160: 2.00,  # Tree cover, flooded, fresh or brakish water
    170: 2.00,  # Tree cover, flooded, saline water
    180: 2.00,  # Shrub or herbaceous cover, flooded, fresh/saline/brakish water
    190: 5.00,  # Urban areas
    200: 4.75,  # Bare areas
    201: 4.50,  # Consolidated bare areas
    202: 4.50,  # Unconsolidated bare areas
    210: 0.50,  # Water bodies
    220: 1.00,  # Permanent snow and ice
}
""" https://www.nature.com/articles/s41591-020-1059-1#MOESM2
evergreen needleleaf forest = 3.24
evergreen broadleaf forest = 1.62
deciduous needleleaf forest = 3.24
deciduous broadleaf forest = 4.00
mixed forest = 3.24,
closed shrublands = 3.00
open shrublands = 4.20
woody savannas = 4.86,
savannas = 4.86
grasslands = 4.86
permanent wetlands = 2.00
croplands = 2.50,
cropland/natural vegetation = 3.24
snow and ice = 1.62
barren or sparsely vegetated = 3.00
"""
copernicus_landcover = {
    0: 0.1,
    10: 2.50,
    20: 3.00,
    30: 4.20,
    40: 2.50,
    50: 5,
    60: 4.86,
    70: 1.62,
    80: 0.5,
    90: 2.00,
    100: 4.86,
    111: 3.24,
    113: 3.24,
    112: 1.62,
    114: 4.00,
    115: 3.24,
    116: 3.24,
    121: 3.75,
    123: 3.75,
    122: 2.10,
    124: 4.00,
    125: 3.75,
    126: 3.75,
    200: 0,
}
