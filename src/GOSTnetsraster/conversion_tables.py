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

copernicus_landcover = {
    0:0.1,
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
    20: 3.00,
    30: 4.20,
    90: 2.00,
    100: 4.86,
    60: 4.86,
    40: 2.50,
    50: 5,
    70: 1.62,
    80: 0.5,
    200: 0,
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