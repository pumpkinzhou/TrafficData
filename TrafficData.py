import json
import glob
from ast import literal_eval
import pickle
import sys
import pprint
import matplotlib.pyplot as plt
import numpy as np
from heapq import heapify, heappush, heappop
from collections import defaultdict


class UnionFind:

    def __init__(self):
        self.count = 0
        self.father = {}

    def init_matrix(self, i):
        self.father[i] = i
        self.count += 1

    def find(self, x):
        if x == self.father.get(x, 0):
            return x
        self.father[x] = self.find(self.father[x])
        return self.father[x]

    def union(self, a, b):
        fatherA = self.find(a)
        fatherB = self.find(b)
        if fatherA != fatherB:
            self.father[fatherA] = fatherB
            self.count -= 1


class Node(object):
    def __init__(self, node_id, LAT ='', LON = ''):
        self.node_id = node_id
        self.LAT = LAT
        self.LON = LON
        self.edges = []
        self.subgraph = 0
        self.visited = False

    def __repr__(self):
        return '{self.__class__.__name__}'.format(self = self)

    def __str__(self):
        return '{self.__class__.__name__}: node_id = {self.node_id}'.format(self = self)


class Edge(object):
    def __init__(self, edge_id, node_from, node_to, length):
        self.edge_id = edge_id
        self.node_from = node_from
        self.node_to = node_to
        self.length = length
        self.speed = defaultdict(dict)     # speed dictionary {}

    def __repr__(self):
        return '{self.__class__.__name__}'.format(self = self)

    def __str__(self):
        return '{self.__class__.__name__}: edge_id = {self.edge_id}'.format(self=self)


class Graph(object):
    def __init__(self, nodes=None, edges=None):
        self.nodes = nodes or []
        self.edges = edges or []
        self._node_map = {}
        self._edge_map = {}

    def __repr__(self):
        return '{self.__class__.__name__}'.format(self=self)

    def insert_node(self, new_node_id, LAT, LON):
        "Insert a new node with value new_node_val"
        if new_node_id in self._node_map:
            print('insert an existing node!')
            return self._node_map[new_node_id]
        new_node = Node(new_node_id, LAT, LON)
        self.nodes.append(new_node)
        self._node_map[new_node_id] = new_node
        return new_node

    def find_node(self, node_id):
        "Return the node with value node_number or None"
        return self._node_map.get(node_id)

    def _clear_visited(self):
        for node in self.nodes:
            node.visited = False

    def insert_edge(self, edge_id, node_from_id, LAT_from, LON_from,
                    node_to_id, LAT_to, LON_to, new_edge_length):
        "Insert a new edge, creating new nodes if necessary"
        if edge_id in self._edge_map:
            print("insert an existing edge!")
            return
        node_from = self._node_map.get(node_from_id) or self.insert_node(node_from_id, LAT_from, LON_from)
        node_to = self._node_map.get(node_to_id) or self.insert_node(node_to_id, LAT_to, LON_to)
        new_edge = Edge(edge_id, node_from, node_to, new_edge_length)
        node_from.edges.append(new_edge)
        node_to.edges.append(new_edge)
        self.edges.append(new_edge)
        self._edge_map[edge_id] = new_edge

    def find_edge(self, edge_id):
        "Return the node with value node_number or None"
        return self._edge_map.get(edge_id)

    def get_edge_list(self):
        """Return a list of triples that looks like this:
        (edge_id, node_from, node_to, length, speed)"""
        return [(e.edge_id, e.node_from.node_id, e.node_to.node_id, e.length, e.speed) for e in self.edges]

    def add_edge_speed(self):
        pass

    def find_path(self, start_node_id, end_node_id, path = []):
        """find path use dfs. The output is a feasible path or None.
        ARGUMENTS: start_node_id, end_node_id
        RETURN: path ([node_id ....])."""

        path = path + [start_node_id]
        node = self.find_node(start_node_id)
        node.visited = True

        if start_node_id == end_node_id:
            return path
        if not node.edges:
            return None


        for e in node.edges:
            nei = e.node_from if node.node_id == e.node_to.node_id else e.node_to
            if nei.visited:
                continue
            elif nei not in path:
                newpath = self.find_path(nei.node_id, end_node_id, path)
                if newpath:
                    return newpath
        # print('There is no path from node {} to node {}'.format(start_node_id, end_node_id))
        return None

    def bfs(self, start_node_id, end_node_id):
        """BFS iterating through a node's edges. The output is the shortest path + length.
        ARGUMENTS: start_node_id, end_node_id
        RETURN: path (node_ids), path_length."""
        if start_node_id == end_node_id:
            return  [start_node_id], 0

        node = self.find_node(start_node_id)
        self._clear_visited()
        queue = [node]
        dist = defaultdict(dict)
        dist[start_node_id]['val'] = 0
        dist[start_node_id]['parent'] = -1  # distance to the start, parent
        node.visited = True
        dist[end_node_id]['val'] = float('inf')
        dist[end_node_id]['parent'] = start_node_id
        # max_queue_length = 0
        while queue:
            node = queue.pop(0)
            # max_queue_length = max(len(queue), max_queue_length)
            if node.node_id == end_node_id:  # terminal condition
                if dist[node.node_id]['val'] < dist[end_node_id]['val']:
                    dist[end_node_id]['val'] = dist[node.node_id]['val']
                    dist[end_node_id]['parent'] = dist[node.node_id]['parent']

            for e in node.edges:
                nei = e.node_from if node.node_id == e.node_to.node_id else e.node_to
                new_dist = dist[node.node_id]['val'] + float(e.length)
                if not nei.visited:
                    nei.visited = True
                    queue.append(nei)
                    dist[nei.node_id]['val'] = new_dist
                    dist[nei.node_id]['parent'] = node.node_id
                elif new_dist < dist[nei.node_id]['val'] :
                    queue.append(nei)
                    dist[nei.node_id]['val'] = new_dist
                    dist[nei.node_id]['parent'] = node.node_id

        if dist[end_node_id]['val'] != float('inf'):
            path = [end_node_id]
            par = dist[end_node_id]['parent']
            while par != -1:
                path.insert(0, par)
                par = dist[par]['parent']
            return path, dist[end_node_id]['val']
        else:
            print('There is no path from node {} to node {}'.format(start_node_id, end_node_id))
            return None

    def dijkstra(self, start_node_id, end_node_id, K):
        """dijkstra uses a priority queue. The output is the shortest path and path_length.
        ARGUMENTS: start_node_id, end_node_id, K is the maximum number of nodes we want to search (approximate)
        RETURN: shortest_path, path_length.."""
        node = self.find_node(start_node_id)
        pq = [(0, 0, node)]

        # dist = {start_node_id: (0, -1)}  # distance to the start, parent
        dist = defaultdict(dict)
        dist[start_node_id]['val'] = 0
        dist[start_node_id]['parent'] = -1  # distance to the start, parent
        # max_queue_length = 0
        sq = 0  #index the node inserted to the priority queue
        while pq:
            # max_queue_length = max(len(pq), max_queue_length)
            obj, _, node = heappop(pq)

            if sq > K:
                break

            if node.node_id == end_node_id: # terminal condition
                path = [end_node_id]
                par = dist[end_node_id]['parent']
                while par != -1:
                    path.insert(0, par)
                    par = dist[par]['parent']
                return path, dist[end_node_id]['val']
                # return path, dist[end_node_id]['val'], max_queue_length

            for e in node.edges:
                nei = e.node_from if node.node_id == e.node_to.node_id else e.node_to
                if e.speed['AVG_SPEED'] is None:
                    speed = 30
                elif e.speed['AVG_SPEED'] == -1:
                    speed = 30
                else:
                    speed = e.speed['AVG_SPEED']
                # new_obj = dist[node.node_id]['val'] + float(e.length)  # shortest length

                new_obj = dist[node.node_id]['val'] + float(e.length) / (speed * 1.6 * 1000 / 3600)   #minimal time unit: seconds

                if nei.node_id not in dist:
                    dist[nei.node_id]['val'] = new_obj
                    dist[nei.node_id]['parent']= node.node_id
                    sq += 1
                    heappush(pq, (new_obj, sq, nei))
                elif new_obj < dist[nei.node_id]['val']:
                    dist[nei.node_id]['val'] = new_obj
                    dist[nei.node_id]['parent'] = node.node_id
                    # sq += 1
                    # heappush(pq, (new_obj, sq, nei))

        print('There is no path from node {} to node {} within K hops'.format(start_node_id, end_node_id))
        return None

    def num_of_subgraphs(self):
        uf = UnionFind()
        for node in self.nodes:
            uf.init_matrix(node.node_id)
            for e in node.edges:
                nei = e.node_from if node.node_id == e.node_to.node_id else e.node_to
                if nei.node_id in uf.father:
                    uf.union(node.node_id, nei.node_id)

        for node in self.nodes:
            node.subgraph = uf.find(node.node_id)

        # print(uf.father.keys())
        return uf.count

    def plot_graph(self, traffic_color= 0):
        """ Plot all links in the graph"""
        i = 0
        if traffic_color == 0:
            for e in self.edges:
                plt.plot([float(e.node_from.LON), float(e.node_to.LON)], [float(e.node_from.LAT), float(e.node_to.LAT)], marker='o')
                i += 1
                if i % 1000 == 0:
                    print(i)
                    plt.show()
        if traffic_color == 1:
            for e in self.edges:
                speed = e.speed['AVG_SPEED']
                if speed is None:
                    speed= 30
                speed = float(speed)
                if speed >= 40:
                    plt.plot([float(e.node_from.LON), float(e.node_to.LON)], [float(e.node_from.LAT), float(e.node_to.LAT)], marker='o', color='g')
                elif 30 <= speed < 40:
                    plt.plot([float(e.node_from.LON), float(e.node_to.LON)], [float(e.node_from.LAT), float(e.node_to.LAT)], marker='o', color='y')
                else:
                    plt.plot([float(e.node_from.LON), float(e.node_to.LON)], [float(e.node_from.LAT), float(e.node_to.LAT)], marker='o', color='r')
                i += 1
                if i % 1000 == 0:
                    print(i)
                    plt.show()
        return


    def plot_subgraphs(self, n):
        """ scatter plot: nodes that belong to the n largest disconnected subgraphs"""
        colors= np.random.rand(n)

        subgraph_dict = defaultdict(int)
        for node in self.nodes:
            subgraph_dict[node.subgraph] += 1

        subgraph_ids = [item[0] for item in sorted(subgraph_dict.items(), key=lambda v: v[1], reverse=True)]
        subgraph_ids = subgraph_ids[:n]

        color_dic = {}
        for i, subgraph in enumerate(subgraph_dict):
            color_dic[subgraph]  = colors[i]

        x, y, color = [], [], []
        for node in self.nodes:
            if node.subgraph in subgraph_ids:
                x.append(int(node.LON))
                y.append(int(node.LAT))
                color.append(color_dic[node.subgraph])

        plt.figure()
        plt.scatter(x, y, c = color)
        plt.show()



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
        e_traffic_pattern = pattern_data.get(e.edge_id, None)
        if e_traffic_pattern:
            e.speed['AVG_SPEED'] = float(e_traffic_pattern['AVG_SPEED'])
            e.speed['FREE_FLOW_SPEED'] = float(e_traffic_pattern['FREE_FLOW_SPEED'])

            if e_traffic_pattern['F_WEEKDAY']:
                e.speed['F_WEEKDAY'] = e_traffic_pattern['F_WEEKDAY'].split(',')

            if e_traffic_pattern['T_WEEKDAY']:
                e.speed['T_WEEKDAY'] = e_traffic_pattern['T_WEEKDAY'].split(',')

            if e.speed['F_WEEKDAY']:
                for pattern_id in e.speed['F_WEEKDAY']:
                    if pattern_id not in pattern_table:
                        print('pattern_id {} not found in pattern_table'.format(pattern_id))
                    else:
                        e.speed['FS'] = pattern_table[pattern_id]['SPEED_VALUES']
            if e.speed['T_WEEKDAY']:
                for pattern_id in e.speed['T_WEEKDAY']:
                    if pattern_id not in pattern_table:
                        print('pattern_id {} not found in pattern_table'.format(pattern_id))
                    else:
                        e.speed['TS'] = pattern_table[pattern_id]['SPEED_VALUES']
        else:
            e.speed['AVG_SPEED'] = -1


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


g = Graph()
# data = read_json('TrafficData\TrafficData_test.json')  #for test purpose only
# print("# of disconnected subgraphs = ", g.num_of_subgraphs())
''' load data'''
data = read_json('TrafficData\\data\\NetworkData.json')     # load main data setv
add_edges(g, data)
# add data from different zoom levels (zoom12 matters most)
data_files = glob.glob('TrafficData\\data\\Boston Data_Diff_Zooms\\NetworkData_zoom*.json')
for f in data_files:
    add_edges(g, read_json(f))

TrafficPatternData = read_json('TrafficData\\data\\BostonData\\TrafficPatternData.json')
TrafficPatternTable = read_json3('TrafficData\\data\\BostonData\\traffic_pattern_table.json')

add_speed_info(g, TrafficPatternData, TrafficPatternTable)



''' graph analysis'''
print("# of disconnected subgraphs = ", g.num_of_subgraphs())
g.plot_subgraphs(g.num_of_subgraphs())
# g.plot_graph()  # ATTENTION: IT TAKES VERY LONG TIME!


# save_graph('CODES\NetworkData.pckl', g)
# g_loaded = load_graph('CODES\TrafficGraph.pckl')
# path, path_length = random_walk('41770342', 50)
# goal_node_id = path[-1]
# g._clear_visited()
# one_path = g.find_path('41770395', '41771374')
# plot_path(one_path, 'b')
# bfs_path, bfs_dist = g.bfs('41770395', '41771374')
# plot_path(bfs_path, 'g')
# dijkstra_path, dijkstra_obj = g.dijkstra('41770395', '41771374', 20000)
# plot_path(dijkstra_path, 'y')

# bfs_path, bfs_dist = g.bfs('41770395', '41733133')
# dijkstra_path, dijkstra_obj = g.dijkstra('41770395', '41733133', 20000)


# if __name__ == "__main__":
#     main()
