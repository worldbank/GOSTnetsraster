# GOSTNets Raster Tools

This repository includes tools and tutorials for generating travel time/market access data from raster-based measures of travel friction. Foundationally, this builds upon the [global friction surface](https://developers.google.com/earth-engine/datasets/catalog/Oxford_MAP_friction_surface_2019) generated as part of the [Malaria Atlas Project](https://malariaatlas.org/), but includes functions for generating custom travel time surfaces based on landcover and road networks (notably OSM).

Instructions on use are best found in the Notebooks folder, where jupyter notebooks explore the basic functions.
![Example of mapping access to health facilities](https://github.com/worldbank/GOSTNets_Raster/blob/master/images/TT_any_facility.png)

## Installation tips

The following instructions have been helpful for installing in a conda environment on a windows machine, so your results may very. These are also specific to our organizational security environment.

```powershell
C:\> conda create --name gnr python=3.12

C:\> conda activate gnr

[gnr] pip install geopandas ipykernel

[gnr] C:\Path_to_GOSTnetsraster code\> pip install . -e

[gnr] C:\Path_to_GOSTnetsraster code\> jupyter-notebook
```
