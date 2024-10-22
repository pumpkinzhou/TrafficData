import json
import glob
from ast import literal_eval
import pickle
import sys
import matplotlib.pyplot as plt
import calendar
calendar.setfirstweekday(calendar.SUNDAY)

from graph import Graph
from car import HCar
from routing import Routing


''' main() '''
def read_json(file):
    with open(file, 'r') as fp:
        data = json.load(fp)
        return data
    print('file not open!')
    return None


def read_json2(file):
    with open(file) as fp:
        mainlist = [literal_eval(line) for line in fp]
        raw_data = mainlist[0] if mainlist else None     #data processing
        d = {}
        for item in raw_data:
            d[item['LINK_ID']] = item
        return d
    print('file not open!')
    return None


def read_json3(file):
    with open(file) as fp:
        mainlist = [literal_eval(line) for line in fp]
        raw_data = mainlist[0] if mainlist else None  #data processing
        d = {}
        for item in raw_data:
            d[item['PATTERN_ID']] = item
        return d
    print('file not open!')
    return None


def add_edges(graph, data):
    # add geographical edge without considering the travel directions
    # travel directions will be added by add_traffic_info.py
    for link_id in data:
        '''edge_id, ref_node_id, LAT_from, LON_from, non_ref_node
        _id, LAT_to, LON_to, new_edge_length)'''
        link = data[link_id]
        LAT = link['LAT'].split(',')
        LAT[1] = str(int(LAT[0]) + int(LAT[1]))
        LON = link['LON'].split(',')
        LON[1] = str(int(LON[0]) + int(LON[1]))
        graph.insert_edge(link['LINK_ID'],
                          link['REF_NODE_ID'], LAT[0], LON[0],
                          link['NONREF_NODE_ID'], LAT[1], LON[1],
                          link['LINK_LENGTH'])


def add_traffic_info(graph, pattern_data, pattern_table):
    for e in graph.edges:
        edge_traffic_pattern = pattern_data.get(e.edge_id, None) # read traffic pattern for e
        if edge_traffic_pattern:
            e.travel_direction = edge_traffic_pattern['TRAVEL_DIRECTION']
            e.traffic_info['AVG_SPEED'] = float(edge_traffic_pattern['AVG_SPEED'])
            e.traffic_info['FREE_FLOW_SPEED'] = float(edge_traffic_pattern['FREE_FLOW_SPEED'])

            if edge_traffic_pattern['F_WEEKDAY']:
                e.traffic_info['F_WEEKDAY'] = edge_traffic_pattern['F_WEEKDAY'].split(',')

            if edge_traffic_pattern['T_WEEKDAY']:
                e.traffic_info['T_WEEKDAY'] = edge_traffic_pattern['T_WEEKDAY'].split(',')

            e.traffic_info['F_SPEED'], e.traffic_info['T_SPEED'] = [], []
            if e.traffic_info['F_WEEKDAY']:
                for pattern_id in e.traffic_info['F_WEEKDAY']:
                    if pattern_id in pattern_table:
                        e.traffic_info['F_SPEED'].append(list(map(float, pattern_table[pattern_id]['SPEED_VALUES'].split(','))))
                        # e.speed['F_SPEED'] = pattern_table[pattern_id]['SPEED_VALUES'].split(',')
                    else:
                        print('pattern_id {} not found in pattern_table'.format(pattern_id))

            if e.traffic_info['T_WEEKDAY']:
                for pattern_id in e.traffic_info['T_WEEKDAY']:
                    if pattern_id in pattern_table:
                        e.traffic_info['T_SPEED'].append(list(map(float, pattern_table[pattern_id]['SPEED_VALUES'].split(','))))
                        # e.speed['T_SPEED'] = pattern_table[pattern_id]['SPEED_VALUES'].split(',')
                    else:
                        print('pattern_id {} not found in pattern_table'.format(pattern_id))

        else:
            e.traffic_info['FREE_FLOW_SPEED'] = 0    # 0 indicates e.edge_id not found pattern_data


def save_graph(filename, graph):
    # print (sys.getrecursionlimit())  # default is 1000
    sys.setrecursionlimit(5000)
    # print ('Recursion limit:', sys.getrecursionlimit())
    with open(filename, 'wb') as fp:
        pickle.dump(graph, fp, pickle.HIGHEST_PROTOCOL)


def load_graph(filename):
    with open(filename, 'rb') as fp:
        graph = pickle.load(fp)
        return graph
    print('Graph not loaded!')
    return None


def plot_path(path, colour = 'k'):
    if not path:
        print("Path is empty!")
        return
    xs = [int(g._node_map[node_id].LAT) for node_id in path]
    ys = [int(g._node_map[node_id].LON) for node_id in path]
    # plt.figure()
    # plt.plot(ys, xs, marker='o', color = colour)
    plt.plot(ys, xs, linewidth=5.0, color=colour)


def random_walk(s, level):
    # need to be updated
    """random_walk.
        ARGUMENTS: start_node_id, level (steps)
        RETURN: path ([node_id ....])."""
    path = [s]
    dist = 0
    for i in range(level):
        cur_node = g.find_node(s)
        for e in cur_node.edges:
            nei = e.ref_node if cur_node.node_id == e.non_ref_node.node_id else e.non_ref_node
            if nei.node_id not in path:
                s = nei.node_id
                path.append(s)
                dist = dist + float(e.length)
                break
    return path, dist


g = Graph()
# data = read_json('TrafficData\TrafficData_test.json')  #for test purpose only
# print("# of disconnected subgraphs = ", g.num_of_subgraphs())
''' load data'''
# data = read_json('TrafficData\\data\\BostonData\\NetworkData.json')     # load main data setv
# add_edges(g, data)
# add data from different zoom levels (zoom12 matters most)
data_files = glob.glob('TrafficData\\data\\BostonData_Diff_Zooms2\\NetworkData_zoom*.json')
for f in data_files:
    add_edges(g, read_json(f))
# load traffic pattern data
TrafficPatternData = read_json('TrafficData\\data\\BostonData_Diff_Zooms2\\TrafficPatternData_zoom9.json')
data_files = glob.glob('TrafficData\\data\\BostonData_Diff_Zooms2\\TrafficPatternData_zoom*.json')
for f in data_files:
    TrafficPatternData.update(read_json(f))
TrafficPatternTable = read_json3('TrafficData\\data\\BostonData\\traffic_pattern_table2.json')
# add speed info to edges
add_traffic_info(g, TrafficPatternData, TrafficPatternTable)

''' graph analysis'''
direction = True
traffic_color = True
# g.plot_graph(direction, traffic_color) # Default Parameters: direction=False, speed_color=False, day=1, time=36
# g.plot_subgraphs(g.num_of_subgraphs())
# print("# of disconnected subgraphs = ", g.num_of_subgraphs())
# g_adj = g.get_adjacency_matrix()
# g_adj = g.get_sparse_adjacency_list()

''' save graph using pickle'''
# save_graph('CODES\NetworkData.pckl', g)
# g_loaded = load_graph('CODES\TrafficGraph.pckl')

''' path finding'''
car = HCar('1')
car.set_origin('41775277')
car.set_destination('41695498')
print(car.get_od_pair())
routing_algo = Routing(g)
# path, path_length = random_walk('41770342', 50)
# goal_node_id = path[-1]
# g._clear_visited()
# one_path = g.find_path('41770395', '41771374')
# plot_path(one_path, 'b')
bfs_path, bfs_dist = g.bfs('41770342', '41770310')
plot_path(bfs_path, 'g')
bfs_path, bfs_dist = routing_algo.bfs('41770342', '41770310')
plot_path(bfs_path, 'k')
dijkstra_path, dijkstra_obj = g.dijkstra('41770342', '41770310')
plot_path(dijkstra_path, 'y')
dijkstra_path, dijkstra_obj = routing_algo.dijkstra('41770342', '41770310')
plot_path(dijkstra_path, 'm')

# cdf_path, energy_cost = alg.cdf_dijkstra(car.origin, car.destination, car.battery_level)
# plot_path(cdf_path, 'm')

# car.set_battery_level(2.0)
# dijkstra_path, energy_cost = g.cdf_dijkstra(car.origin, car.destination, car.battery_level)
# plot_path(dijkstra_path, 'm')


# bfs_path, bfs_dist = g.bfs('41770395', '41733133')
# dijkstra_path, dijkstra_obj = g.dijkstra('41770395', '41733133', 20000)


# if __name__ == "__main__":
#     main()
