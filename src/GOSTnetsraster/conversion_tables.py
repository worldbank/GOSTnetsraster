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
esaacci_landcover = {
    0: 0.1,  #No data
    10: 2.50, #Cropland, rainfed
    11: 4.00, # ???Herbaceous cover
    12: 4.20, # ???Tree or shrub cover
    20: 3.24, #Cropland, irrigated or post-flooding
    30: 2.00, #Mosaic cropland (>50%) / natural vegetation (tree, shrub, herbaceous cover) (<50%)
    40: 3.24, #Mosaic natural vegetation (tree, shrub, herbaceous cover) (>50%) / cropland (<50%) 
    50: 4.00, #Tree cover, broadleaved, evergreen, closed to open (>15%)
    60: 3.50, #Tree cover, broadleaved, deciduous, closed to open (>15%)
    61: 3.00, #Tree cover, broadleaved, deciduous, closed (>40%)
    62: 2.50, #Tree cover, broadleaved, deciduous, open (15-40%)
    70: 3.50, #Tree cover, needleleaved, evergreen, closed to open (>15%)
    71: 3.00, #Tree cover, needleleaved, evergreen, closed (>40%)
    72: 3.24, #Tree cover, needleleaved, evergreen, open (15-40%)
    80: 3.50, #Tree cover, needleleaved, deciduous, closed to open (>15%)
    81: 3.00, #Tree cover, needleleaved, deciduous, closed (>40%)
    82: 3.24, #Tree cover, needleleaved, deciduous, open (15-40%)
    90: 3.24, #Tree cover, mixed leaf type (broadleaved and needleleaved)
    100: 3.00, #Mosaic tree and shrub (>50%) / herbaceous cover (<50%)
    110: 3.00, #Mosaic herbaceous cover (>50%) / tree and shrub (<50%)
    120: 4.20, #Shrubland
    121: 3.24, #Shrubland evergreen
    122: 3.24, #Shrubland deciduous
    130: 4.86, #Grassland
    140: 0.00, # ???Lichens and mosses
    150: 4.20, #Sparse vegetation (tree, shrub, herbaceous cover) (<15%)
    151: 4.00, #Sparse tree (<15%)
    152: 4.00, #Sparse shrub (<15%)
    153: 4.00, #Sparse herbaceous cover (<15%)
    160: 2.00, #Tree cover, flooded, fresh or brakish water
    170: 2.00, #Tree cover, flooded, saline water
    180: 2.00, #Shrub or herbaceous cover, flooded, fresh/saline/brakish water
    190: 5.00, #Urban areas
    200: 4.75, #Bare areas
    201: 4.50, #Consolidated bare areas
    202: 4.50, #Unconsolidated bare areas
    210: 0.50, #Water bodies
    220: 1.00, #Permanent snow and ice
}
''' https://www.nature.com/articles/s41591-020-1059-1#MOESM2
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
'''
copernicus_landcover = {
    0:0.1,
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

