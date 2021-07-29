from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='GOSTNetsRaster',
    packages=['GOSTNetsRaster'],
    install_requires=[
        'rasterio',
        'geopandas',
        'pandas',
        'numpy',
        'osmnx',
        'GOSTNets',
        'scikit-image',
        'pyproj',
        'GOSTRocks',
        'scipy',
        'pandana'
    ],
    version='0.0.1',
    description='Generate travel time rasters from friction surfaces',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/worldbank/GOSTNets_Raster",    
    author='Benjamin P. Stewart',
    license='MIT',    
    package_dir= {'':'src'}
)
