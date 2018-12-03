import json
import glob
from ast import literal_eval
import pickle
import sys
import matplotlib.pyplot as plt
import calendar
calendar.setfirstweekday(calendar.SUNDAY)

import graph


''' main() '''
def read_json(file):
    with open(file, 'r') as fp:
        data = json.load(fp)
        fp.close()
        return data
    print('file not open!')
    return None


def read_json2(file):
    with open(file) as fp:
        mainlist = [literal_eval(line) for line in fp]
        fp.close()
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
        fp.close()
        raw_data = mainlist[0] if mainlist else None  #data processing
        d = {}
        for item in raw_data:
            d[item['PATTERN_ID']] = item
        return d
    print('file not open!')
    return None


def add_edges(graph, data):
    for link_id in data:
        '''edge_id, node_from_id, LAT_from, LON_from, node_to_id, LAT_to, LON_to, new_edge_length, new_speed)'''
        link = data[link_id]
        LAT = link['LAT'].split(',')
        LAT[1] = str(int(LAT[0]) + int(LAT[1]))
        LON = link['LON'].split(',')
        LON[1] = str(int(LON[0]) + int(LON[1]))
        graph.insert_edge(link['LINK_ID'], link['REF_NODE_ID'], LAT[0], LON[0], link['NONREF_NODE_ID'], LAT[1],
                          LON[1], link['LINK_LENGTH'])


def add_speed_info(graph, pattern_data, pattern_table):
    for e in graph.edges:
        traffic_pattern = pattern_data.get(e.edge_id, None)
        if traffic_pattern:
            e.speed['AVG_SPEED'] = float(traffic_pattern['AVG_SPEED'])
            e.speed['FREE_FLOW_SPEED'] = float(traffic_pattern['FREE_FLOW_SPEED'])

            if traffic_pattern['F_WEEKDAY']:
                e.speed['F_WEEKDAY'] = traffic_pattern['F_WEEKDAY'].split(',')

            if traffic_pattern['T_WEEKDAY']:
                e.speed['T_WEEKDAY'] = traffic_pattern['T_WEEKDAY'].split(',')

            e.speed['F_SPEED'], e.speed['T_SPEED'] = [], []
            if e.speed['F_WEEKDAY']:
                for pattern_id in e.speed['F_WEEKDAY']:
                    if pattern_id in pattern_table:
                        e.speed['F_SPEED'].append(list(map(float, pattern_table[pattern_id]['SPEED_VALUES'].split(','))))
                        # e.speed['F_SPEED'] = pattern_table[pattern_id]['SPEED_VALUES'].split(',')
                    else:
                        print('pattern_id {} not found in pattern_table'.format(pattern_id))

            if e.speed['T_WEEKDAY']:
                for pattern_id in e.speed['T_WEEKDAY']:
                    if pattern_id in pattern_table:
                        e.speed['T_SPEED'].append(list(map(float, pattern_table[pattern_id]['SPEED_VALUES'].split(','))))
                        # e.speed['T_SPEED'] = pattern_table[pattern_id]['SPEED_VALUES'].split(',')
                    else:
                        print('pattern_id {} not found in pattern_table'.format(pattern_id))

        else:
            e.speed['FREE_FLOW_SPEED'] = 0    # 0 indicates e.edge_id not found pattern_data


def save_graph(filename, graph):
    # print (sys.getrecursionlimit())  # default is 1000
    sys.setrecursionlimit(5000)
    # print ('Recursion limit:', sys.getrecursionlimit())
    with open(filename, 'wb') as fp:
        pickle.dump(graph, fp, pickle.HIGHEST_PROTOCOL)
        fp.close()


def load_graph(filename):
    with open(filename, 'rb') as fp:
        graph = pickle.load(fp)
        fp.close()
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
    plt.plot(ys, xs, marker='o', color = colour)


def random_walk(s, level):
    """random_walk.
        ARGUMENTS: start_node_id, level (steps)
        RETURN: path ([node_id ....])."""
    path = [s]
    dist = 0
    for i in range(level):
        cur_node = g.find_node(s)
        for e in cur_node.edges:
            nei = e.node_from if cur_node.node_id == e.node_to.node_id else e.node_to
            if nei.node_id not in path:
                s = nei.node_id
                path.append(s)
                dist = dist + float(e.length)
                break
    return path, dist


g = graph.Graph()
# data = read_json('TrafficData\TrafficData_test.json')  #for test purpose only
# print("# of disconnected subgraphs = ", g.num_of_subgraphs())
''' load data'''
data = read_json('TrafficData\\data\\BostonData\\NetworkData.json')     # load main data setv
add_edges(g, data)
# add data from different zoom levels (zoom12 matters most)
data_files = glob.glob('TrafficData\\data\\BostonData_Diff_Zooms\\NetworkData_zoom*.json')
for f in data_files:
    add_edges(g, read_json(f))
# load traffic pattern data
TrafficPatternData = read_json('TrafficData\\data\\BostonData\\TrafficPatternData.json')
data_files = glob.glob('TrafficData\\data\\BostonData_Diff_Zooms\\TrafficPatternData_zoom*.json')
for f in data_files:
    TrafficPatternData.update(read_json(f))
TrafficPatternTable = read_json3('TrafficData\\data\\BostonData\\traffic_pattern_table.json')
# add speed info to edges
add_speed_info(g, TrafficPatternData, TrafficPatternTable)

''' graph analysis'''
g.plot_graph(True, 1, 37)
# g.plot_graph(True)
# print("# of disconnected subgraphs = ", g.num_of_subgraphs())
# g.plot_subgraphs(g.num_of_subgraphs())

''' save graph using pickle'''
# save_graph('CODES\NetworkData.pckl', g)
# g_loaded = load_graph('CODES\TrafficGraph.pckl')

''' path finding'''
# path, path_length = random_walk('41770342', 50)
# goal_node_id = path[-1]
# g._clear_visited()
# one_path = g.find_path('41770395', '41771374')
# plot_path(one_path, 'b')
# bfs_path, bfs_dist = g.bfs('41770395', '41771374')
# plot_path(bfs_path, 'g')
# dijkstra_path, dijkstra_obj = g.dijkstra('41770395', '41771374', 20000)
bfs_path, bfs_dist = g.bfs('41775277', '41695498')
plot_path(bfs_path, 'k')
dijkstra_path, dijkstra_obj = g.dijkstra('41775277', '41695498')
plot_path(dijkstra_path, 'm')

# bfs_path, bfs_dist = g.bfs('41770395', '41733133')
# dijkstra_path, dijkstra_obj = g.dijkstra('41770395', '41733133', 20000)


# if __name__ == "__main__":
#     main()
