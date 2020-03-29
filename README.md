# <div align="left"><img src="https://rapids.ai/assets/images/rapids_logo.png" width="90px"/>&nbsp;cuSpatial - GPU-Accelerated Spatial and Trajectory Data Management and Analytics Library</div>

[![Build Status](https://gpuci.gpuopenanalytics.com/job/rapidsai/job/gpuci/job/cuspatial/job/branches/job/cuspatial-branch-pipeline/badge/icon)](https://gpuci.gpuopenanalytics.com/job/rapidsai/job/gpuci/job/cuspatial/job/branches/job/cuspatial-branch-pipeline/)

**NOTE:** cuSpatial depends on [cuDF](https://github.com/rapidsai/cudf) and
[RMM](https://github.com/rapidsai/rmm) from [RAPIDS](https://rapids.ai/).

## Implemented operations in 0.11:
1. [Spatial window query](./docs/basic_spatial_trajectory_opertators.pdf)
2. [Point-in-polygon test](./docs/basic_spatial_trajectory_opertators.pdf)
3. Haversine distance
4. Hausdorff distance
5. [Deriving trajectories from point location data](./docs/basic_spatial_trajectory_opertators.pdf)
6. [Computing distance/speed of trajectories](./docs/basic_spatial_trajectory_opertators.pdf)
7. [Computing spatial bounding boxes of trajectories](./docs/basic_spatial_trajectory_opertators.pdf)

## Implemented operations in C++ ready for tests 
Quadtree indexing and Point-in-Polygon test based spatial join<br>
1 [construct quadtree on large-scale point data](./docs/quadtree_indexing_spatial_join.pdf)
2 [compute polygon bounding boxes using parallel primitives](./docs/quadtree_indexing_spatial_join.pdf)
3 [quadtree-polygon pairing for spatial filtering](./docs/quadtree_indexing_spatial_join.pdf)
4 [spatial refinement to pair up points and polygons based on point-in-polygon test](./docs/quadtree_indexing_spatial_join.pdf)

## Future support is planned for the following operations.
1. Point-to-polyline nearest neighbor distance
2. Grid-based indexing for points and polygons
3. R-Tree-based indexing for Polygons/Polylines

## Install from Conda
To install via conda [(0.12 and prior)]:(https://anaconda.org/rapidsai/cuspatial)
```
conda install -c conda-forge -c rapidsai-nightly cuspatial
```

## Install from Source
To build and install cuSpatial from source:

### Install dependencies

Currently, building cuSpatial requires a source installation of cuDF. Install
cuDF by following the [instructions](https://github.com/rapidsai/cudf/blob/branch-0.11/CONTRIBUTING.md#script-to-build-cudf-from-source)

The rest of steps assume the environment variable `CUDF_HOME` points to the 
root directory of your clone of the cuDF repo, and that the `cudf_dev` Anaconda
environment created in step 3 is active.

### Clone, build and install cuSpatial

1. export `CUSPATIAL_HOME=$(pwd)/cuspatial`
2. clone the cuSpatial repo

```
git clone https://github.com/zhangjianting/cuspatial/ $CUSPATIAL_HOME
```

3. Compile and install 
Similar to cuDF, simplely run 'build.sh' diectly under $CUSPATIAL_HOME<br>
Note that a "build" dir is created automatically under $CUSPATIAL_HOME/cpp

4. Run C++/Python test code <br>

Some tests using inline data can be run directly, e.g.,
```
$CUSPATIAL_HOME/cpp/build/gtests/HAUSDORFF_TEST
$CUSPATIAL_HOME/cpp/build/gtests/POINT_IN_POLYGON_TEST
python python/cuspatial/cuspatial/tests/test_hausdorff_distance.py
python python/cuspatial/cuspatial/tests/test_pip.py
```

Some other tests involve I/O from data files under $CUSPATIAL_HOME/test_fixtures.
For example, $CUSPATIAL_HOME/cpp/build/gtests/SHAPEFILE_POLYGON_READER_TEST requires three
pre-generated polygon shapefiles that contain 0, 1 and 2 polygons, respectively. They are available at 
$CUSPATIAL_HOME/test_fixtures/shapefiles <br>

##running tests on NYC taxi trip data with multiple polygon datasets
URLs to polygon datasets are embedded in code. <br>
Point data can be downloaded [here](http://geoteci.engr.ccny.cuny.edu/nyctaxidata/) </br>
Moidify 2009.cat used in the [test code](./cpp/tests/join/spatial_join_nyctaxi_test.cu) to include data of any months</br>
Or, create your own .cat file (one line per month data file) and use it in the test code. 

To run the test:
```
cd $CUSPATIAL_HOME/cpp/build/
./gtest/SPATIAL_JOIN_NYCTAXIE_TEST
```

**NOTE:** Currently, cuSpatial supports reading point/polyine/polygon data using
Structure of Array (SoA) format and a [shapefile reader](./cpp/src/io/shp)
to read polygon data from a shapefile.
Alternatively, python users can read any point/polyine/polygon data using
existing python packages, e.g., [Shapely](https://pypi.org/project/Shapely/) 
and [Fiona](https://github.com/Toblerity/Fiona),to generate numpy arrays and feed them to
[cuSpatial python APIs](python/cuspatial/cuspatial).
