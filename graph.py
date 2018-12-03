import matplotlib.collections as mc
import numpy as np
from heapq import heapify, heappush, heappop
from collections import defaultdict
import pprint
import matplotlib.pyplot as plt

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

    def dijkstra(self, start_node_id, end_node_id, K = float('inf')):
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
                if e.speed['FREE_FLOW_SPEED'] is None:
                    speed = 30
                elif e.speed['FREE_FLOW_SPEED'] == 0:
                    speed = 30
                else:
                    speed = e.speed['FREE_FLOW_SPEED']
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

    def plot_graph(self, speed_color=False, day = 1, time = 36):
        """ Plot all links in the graph, day: 0 == Sun, 1 == Mon, ..., 6 == Sat, time: 0 == 0:00, 1 == 0:15, ...."""

        lines = []
        for e in self.edges:
            lines.append([(float(e.node_from.LON), float(e.node_from.LAT)),
                          (float(e.node_to.LON), float(e.node_to.LAT))])

        # print([(float(e.node_from.LON), float(e.node_from.LAT)), (float(e.node_to.LON), float(e.node_to.LAT))])
        # plt.plot([float(e.node_from.LON), float(e.node_to.LON)],
        #          [float(e.node_from.LAT), float(e.node_to.LAT)], marker='o')
        # plt.show()
        if not speed_color:
            line_segments = mc.LineCollection(lines)
        else:
            c = []
            for e in self.edges:
                if e.speed['FREE_FLOW_SPEED']:
                    if not (e.speed['F_SPEED'] or e.speed['T_SPEED']):
                        print (e.speed['F_SPEED'], e.speed['T_SPEED'])
                        print(e.edge_id)
                    speedFactor = (e.speed['F_SPEED'] or e.speed['T_SPEED'])[day][time]/e.speed['FREE_FLOW_SPEED']
                else:
                    speedFactor =  1

                if speedFactor < 0.5:
                    c.append('r')
                elif 0.5 <= speedFactor < 0.75:
                    c.append('y')
                # elif 0.75 <= speedFactor < 1:
                #     c.append('g')
                else:
                    # c.append('k')
                    c.append('g')
            line_segments = mc.LineCollection(lines, colors=c)

        fig, ax = plt.subplots()
        ax.add_collection(line_segments)
        ax.autoscale()
        return

    def plot_subgraphs(self, n):
        """ plot: colored n largest disconnected subgraphs"""
        colors= np.concatenate((np.random.rand(3, n), np.ones((1,n))))

        subgraph_dict = defaultdict(int)
        for node in self.nodes:
            subgraph_dict[node.subgraph] += 1

        subgraph_ids = [item[0] for item in sorted(subgraph_dict.items(), key=lambda v: v[1], reverse=True)]
        subgraph_ids = subgraph_ids[:n]

        color_dic = {}
        for i, subgraph in enumerate(subgraph_dict):
            color_dic[subgraph]  = colors[:,i]

        lines = []
        c = []
        for e in self.edges:
            if e.node_from.subgraph in subgraph_ids:
                lines.append([(float(e.node_from.LON), float(e.node_from.LAT)),
                              (float(e.node_to.LON), float(e.node_to.LAT))])
                c.append(color_dic[e.node_from.subgraph])

        line_segments = mc.LineCollection(lines, colors=c)
        fig, ax = plt.subplots()
        ax.add_collection(line_segments)
        ax.autoscale()

        # x, y, color = [], [], []
        # for node in self.nodes:
        #     if node.subgraph in subgraph_ids:
        #         x.append(int(node.LON))
        #         y.append(int(node.LAT))
        #         color.append(color_dic[node.subgraph])
        # plt.figure()
        # plt.scatter(x, y, c = color)
        # plt.show()
