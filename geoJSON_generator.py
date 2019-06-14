# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 15:05:32 2019

@author: arianh
"""
import json
import sys
import glob
from graph import Graph
from TrafficData import read_json, read_json3, add_edges, add_traffic_info

g = Graph()
''' load data'''
data_files = glob.glob('data\\BostonData_Diff_Zooms2\\NetworkData_zoom*.json')
for f in data_files:
    add_edges(g, read_json(f))
# load traffic pattern data
TrafficPatternData = read_json('data\\BostonData_Diff_Zooms2\\TrafficPatternData_zoom9.json')
data_files = glob.glob('data\\BostonData_Diff_Zooms2\\TrafficPatternData_zoom*.json')
for f in data_files:
    TrafficPatternData.update(read_json(f))
TrafficPatternTable = read_json3('data\\BostonData\\traffic_pattern_table2.json')
# add speed info to edges
#add_traffic_info(g, TrafficPatternData, TrafficPatternTable)

# creating a GeoJSON file including the links, lat long values and speed



geojson = {
    "type": "FeatureCollection",
    "features": [
    {
        "type": "Feature",
        "geometry" : {
            "type": "LineString",
            "coordinates": [[float(e.ref_node.LON)*1e-5, float(e.ref_node.LAT)*1e-5],
                             [float(e.non_ref_node.LON)*1e-5, float(e.non_ref_node.LAT)*1e-5]
                            ],
            },
        "properties" : e,
     } for e in g.edges]
}

output = open('out_file.json', 'w')
json.dump(geojson, output)
