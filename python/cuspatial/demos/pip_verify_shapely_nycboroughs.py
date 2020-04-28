# point data dowloaded from
# https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_2009-01.csv

import time
import shapefile
from shapely.geometry import shape, Point, Polygon
import geopandas as gpd
from cudf import Series, read_csv
import cuspatial
from cuspatial._lib.spatial import cpp_point_in_polygon_bitmap

start = time.time()
df = read_csv("yellow_tripdata_2009-01.csv")
end = time.time()
print("data ingesting time (from SSD) in ms={}".format((end - start) * 1000))

start = time.time()
x1 = Series(df["Start_Lon"])
y1 = Series(df["Start_Lat"])
x2 = Series(df["End_Lon"])
y2 = Series(df["End_Lat"])
end = time.time()
print(
    "data frame to gdf column conversion time in ms={}".format(
        (end - start) * 1000
    )
)

NYC_boroughs = gpd.read_file('https://data.cityofnewyork.us/api/geospatial/tqmj-j8zm?method=export&format=GeoJSON')
NYC_boroughs.to_file('NYC_boroughs.shp')
NYC_gpu = cuspatial.read_polygon_shapefile('NYC_boroughs.shp')

plyreader = shapefile.Reader("NYC_boroughs.shp")
polygons = plyreader.shapes()
plys = []
for ply in polygons:
    plys.append(shape(ply))

start = time.time()
bm1 = cpp_point_in_polygon_bitmap( x1, y1, NYC_gpu[0], NYC_gpu[1], NYC_gpu[2]['x'], NYC_gpu[2]['y'])
bm2 = cpp_point_in_polygon_bitmap(x2, y2, NYC_gpu[0], NYC_gpu[1], NYC_gpu[2]['x'], NYC_gpu[2]['y'])
end = time.time()
print("Python GPU Time in ms (end-to-end)={}".format((end - start) * 1000))

bm1a = bm1.to_array()
pntx = x1.to_array()
pnty = y1.to_array()

start = time.time()
mis_match = 0
#for i in range(len(pntx)):
for i in range(10000):
    pt = Point(pntx[i], pnty[i])
    res = 0
    for j in range(len(plys)):
        pip = plys[len(plys) - 1 - j].contains(pt)
        if pip:
            res |= 0x01 << (len(plys) - 1 - j)
    #print("cpu={}, gpu={}".format(res,bm1a[i]))
    #print("{},{},{},{}".format(pntx[i], pnty[i],bm1a[i],res))
    if res != bm1a[i]:
        mis_match = mis_match + 1

end = time.time()
print(end - start)
print(
    "python(shapely) CPU Time in ms (end-to-end)={}".format(
        (end - start) * 1000
    )
)

print("CPU and GPU results mismatch={}".format(mis_match))
