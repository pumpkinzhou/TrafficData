import matplotlib.collections as mc
import networkx as nx
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
    def __init__(self, node_id, LAT='', LON=''):
        self.node_id = node_id
        self.LAT = LAT
        self.LON = LON
        self.edges = []
        self.subgraph = 0
        self.visited = False

    def __repr__(self):
        return '{self.__class__.__name__}'.format(self=self)

    def __str__(self):
        return '{self.__class__.__name__}: node_id = {self.node_id}'.format(self=self)


class Edge(object):
    def __init__(self, edge_id, ref_node, non_ref_node, length, edges_connected_node_from, edges_connected_node_to):
        self.edge_id = edge_id
        self.ref_node = ref_node
        self.non_ref_node = non_ref_node
        self.length = length
        self.travel_direction = ''
        self.edges_connected_node_from = edges_connected_node_from
        self.edges_connected_node_to = edges_connected_node_to
        self.traffic_info = defaultdict(dict)  # traffic dictionary {}

    def get_travel_direction(self):
        return self.travel_direction

    def set_travel_direction(self, dir):  # dir = 'B' or 'F' or 'T'
        self.travel_direction = dir

    def get_traffic_info(self):
        return self.traffic_info

    def __repr__(self):
        return '{self.__class__.__name__}'.format(self=self)

    def __str__(self):
        return '{self.__class__.__name__}: edge_id = {self.edge_id}'.format(self=self)


class Graph(object):
    def __init__(self, nodes=None, edges=None):
        self.nodes = nodes or []
        self.edges = edges or []
        self._node_map = {}  # node_id --> node
        self._edge_map = {}
        self.node_id_to_num = {}  # node_id --> index
        self.edge_id_to_num = {}  # edge_id --> index

    def __repr__(self):
        return '{self.__class__.__name__}'.format(self=self)

    def build_node_id_to_num(self):
        """
        Node numbers are 0 based (starting at 0).
        """
        i = 0
        for node in self.nodes:
            self.node_id_to_num[node.node_id] = i
            i += 1

    def get_node_num(self, id):
        """
        this can only be called after build_node_id_to_num()
        :param id: node_id
        :return: node_num
        """
        return self.node_id_to_num[id]

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

    def insert_edge(self, edge_id, ref_node_id, LAT_from, LON_from,
                    non_ref_node_id, LAT_to, LON_to, new_edge_length,
                    edges_connected_node_from, edges_connected_node_to):
        "Insert a new edge, creating new nodes if necessary"
        if edge_id in self._edge_map:
            print("insert an existing edge!")
            return
        ref_node = self._node_map.get(ref_node_id) or self.insert_node(ref_node_id, LAT_from, LON_from)
        non_ref_node = self._node_map.get(non_ref_node_id) or self.insert_node(non_ref_node_id, LAT_to, LON_to)
        new_edge = Edge(edge_id, ref_node, non_ref_node, new_edge_length,
                        edges_connected_node_from, edges_connected_node_to)
        ref_node.edges.append(new_edge)
        non_ref_node.edges.append(new_edge)
        self.edges.append(new_edge)
        self._edge_map[edge_id] = new_edge

    '''
    def find_edge(self, edge_id):
        "Return the node with value node_number or None"
        return self._edge_map.get(edge_id)
    '''

    def get_edge_list(self):
        """Return a list of triples that looks like this:
        (edge_id, ref_node, non_ref_node, length, travel_direction, traffic_info)"""
        return [(e.edge_id, e.ref_node.node_id, e.non_ref_node.node_id, e.length, e.travel_direction, e.traffic_info)
                for e in self.edges]

    ###############################################################################

    def get_node_id(self, num):
        "Return the node_id related to node number"
        for key, value in self.node_id_to_num.items():
            if num == value:
                return key
        return "node_id does not exist"

    def find_edge_by_nodes(self, ref_node_id, non_ref_node_id):
        "Return the edge related to one ref_node and one node_to"
        for e in self.edges:
            if ref_node_id == e.ref_node.node_id and non_ref_node_id == e.non_ref_node.node_id:
                return e
        return "edge does not exist"

    def find_edge(self, edge_id):
        "Return the edge related to edge_id"
        for key, value in self._edge_map.items():
            if edge_id == key:
                return value
        return "edge_id doesn't exist. (find_edge)"

    def build_edge_id_to_num(self):
        """
        Edge numbers are 0 based (starting at 0).
        """
        i = 0
        for edge in self.edges:
            self.edge_id_to_num[edge.edge_id] = i
            i += 1

    def find_edge_num(self, edge_id):
        "Return the edge number related to edge_id"
        self.build_edge_id_to_num()
        for key, value in self.edge_id_to_num.items():
            if edge_id == key:
                return value
        # print('edge_id: %s does not exist' % edge_id)
        return "edge_id doesn't exist. (find_edge_num)"

    def get_edge_id(self, num):
        "Return the edge_id related to edge number"
        for key, value in self.edge_id_to_num.items():
            if num == value:
                return key
        return "edge_id does not exist"

    def find_incoming_outgoing_edges(self, edges_connected_node_from, edges_connected_node_to):
        "find incoming and outgoing edges to one edge"
        incoming_edges = []
        outgoing_edges = []
        incomings = []
        edges_connected = edges_connected_node_from + edges_connected_node_to
        if edges_connected:
            for edge in edges_connected:
                if edge != '':
                    # if self.find_edge(edge) != "edge_id doesn't exist. (find_edge)":
                    if edge[0] == '-':
                        incomings.append(edge)
                    else:
                        outgoing_edges.append(edge)
        for ins in incomings:
            ins = ins.replace("-", "")
            incoming_edges.append(ins)

        return incoming_edges, outgoing_edges

    '''
    def ref_nonRef_edges_Not_exist_in_graph(self):
        "find edges which are exist in Ref and NonRef edges, but they are not exist in graph _edge_map"
        edges_Not_exist_in_graph = []
        for e in self.edges:
            edges_connected = []
            edges_connected_edge_ids = []
            edges_connected = e.edges_connected_node_from + e.edges_connected_node_to

            for ee in edges_connected:
                ee = ee.replace("-", "")
                edges_connected_edge_ids.append(ee)

            for edge_id in edges_connected_edge_ids:
                if edge_id != '':
                    if edge_id in self._edge_map:
                        continue
                    else:
                        edges_Not_exist_in_graph.append(edge_id)
        return edges_Not_exist_in_graph

    def remove_marginal_edges(self):
        "remove all edges which are exist in Ref and NonRef edges, but they are not exist in graph._edge_map"
        for e in self.edges:
            marginal_edges = []
            for ee in e.edges_connected_node_from:
                edge_id = ee.replace("-", "")
                if edge_id in self._edge_map:
                    continue
                else:
                    marginal_edges.append(ee)
            for eee in marginal_edges:
                e.edges_connected_node_from.remove(eee)

        for e in self.edges:    
            marginal_edges = []
            for ee in e.edges_connected_node_to:
                edge_id = ee.replace("-", "")
                if edge_id in self._edge_map:
                    continue
                else:
                    marginal_edges.append(ee)
            for eee in marginal_edges:
                e.edges_connected_node_to.remove(eee)

    def add_edges_to_make_graph_closed(self):
        "first,find edges not connected to other edges in one side (REF or NonREF)"
        "then, add one edge exactly in the reverse direction of our edge in order to make the graph closed"
        for e in self.edges:
            ref_links = []
            non_ref_links = []

            ref_links = e.edges_connected_node_from
            non_ref_links = e.edges_connected_node_to

            if len(ref_links) == 0:
                r = []
                n = []
                old_id = e.edge_id
                new_edge_id = old_id + str(111) # ? is link_id correct in this way ?
                r.append(str('-') + e.edge_id)
                new_edge_ref_links = non_ref_links + r
                n.append(e.edge_id)
                new_edge_non_ref_links = ref_links + n
                # insert_edge(edge_id, node_from, node_to, new_edge_length, 
                #             edges_connected_node_from, edges_connected_node_to)
                self.insert_edge(new_edge_id, e.node_to.node_id, e.node_to.LAT, e.node_to.LON,
                                 e.node_from.node_id, e.node_from.LAT, e.node_from.LON, e.length,
                                 new_edge_ref_links, new_edge_non_ref_links)
                # add_speed_info
                new_edge = self.find_edge(new_edge_id)
                new_edge.speed['AVG_SPEED'] = e.speed['AVG_SPEED']
                new_edge.speed['FREE_FLOW_SPEED'] = e.speed['FREE_FLOW_SPEED']
                new_edge.speed['F_WEEKDAY'] = e.speed['T_WEEKDAY']
                new_edge.speed['T_WEEKDAY'] = e.speed['F_WEEKDAY']
                new_edge.speed['F_SPEED'] = e.speed['T_SPEED']
                new_edge.speed['T_SPEED'] = e.speed['F_SPEED']

                # update old edge REF and NONREF links
                e.edges_connected_node_from.append(str('-') + new_edge_id)
                e.edges_connected_node_to.append(new_edge_id)

                n = e.node_to
                nei_edges = n.edges
                for nei_edge in nei_edges:
                    if nei_edge.edge_id != e.edge_id and nei_edge.edge_id != new_edge_id:
                        if nei_edge.node_from.node_id == n.node_id:
                            nei_edge.edges_connected_node_from.append(new_edge_id)
                        elif nei_edge.node_to.node_id == n.node_id:
                            nei_edge.edges_connected_node_to.append(new_edge_id)


            if len(non_ref_links) == 0:
                r = []
                n = []
                old_id = e.edge_id
                new_edge_id = old_id + str(111) # ? is link_id correct in this way ?             
                r.append(str('-') + e.edge_id)
                new_edge_ref_links = non_ref_links + r
                n.append(e.edge_id)
                new_edge_non_ref_links = ref_links + n                    
                # insert_edge(edge_id, node_from, node_to, new_edge_length, 
                #             edges_connected_node_from, edges_connected_node_to)
                self.insert_edge(new_edge_id, e.node_to.node_id, e.node_to.LAT, e.node_to.LON,
                                 e.node_from.node_id, e.node_from.LAT, e.node_from.LON, e.length,
                                 new_edge_ref_links, new_edge_non_ref_links)
                # add_speed_info
                new_edge = self.find_edge(new_edge_id)
                new_edge.speed['AVG_SPEED'] = e.speed['AVG_SPEED']
                new_edge.speed['FREE_FLOW_SPEED'] = e.speed['FREE_FLOW_SPEED']
                new_edge.speed['F_WEEKDAY'] = e.speed['T_WEEKDAY']
                new_edge.speed['T_WEEKDAY'] = e.speed['F_WEEKDAY']
                new_edge.speed['F_SPEED'] = e.speed['T_SPEED']
                new_edge.speed['T_SPEED'] = e.speed['F_SPEED']

                # update old edge REF and NONREF links
                e.edges_connected_node_from.append(str('-') + new_edge_id)
                e.edges_connected_node_to.append(new_edge_id)

                n = e.node_from
                nei_edges = n.edges
                for nei_edge in nei_edges:
                    if nei_edge.edge_id != e.edge_id and nei_edge.edge_id != new_edge_id:
                        if nei_edge.node_from.node_id == n.node_id:
                            nei_edge.edges_connected_node_from.append(str('-') + new_edge_id)
                        elif nei_edge.node_to.node_id == n.node_id:
                            nei_edge.edges_connected_node_to.append(str('-') + new_edge_id)

    def make_adjacency_matrix_symmetric(self, pattern_table):
        "add edges in the inverse direction of existing edge" 
        "in order to make the adjacency matrix symmetric"

        adjacency_matrix = self.get_adjacency_matrix()
        max_index = len(adjacency_matrix)
        existing_edges_indices = []

        for i in range (max_index):
            for j in range (max_index):
                if adjacency_matrix[(i,j)] != 0:
                    existing_edges_indices.append((i,j))


        for element in existing_edges_indices:
            i = element[0]
            j = element[1]            
            adjacency_matrix[(j,i)] = adjacency_matrix[(i,j)]

            node_from_id = self.get_node_id(i)
            node_to_id = self.get_node_id(j)
            e = self.find_edge_by_nodes(node_from_id, node_to_id)

            ref_links = []
            non_ref_links = []
            ref_links = e.edges_connected_node_from
            non_ref_links = e.edges_connected_node_to

            r = []
            n = []
            old_id = e.edge_id
            new_edge_id = old_id + str(111) # ? is link_id correct in this way ?             
            r.append(str('-') + e.edge_id)
            new_edge_ref_links = non_ref_links + r
            n.append(e.edge_id)
            new_edge_non_ref_links = ref_links + n                    
            # insert_edge(edge_id, node_from, node_to, new_edge_length, 
            #             edges_connected_node_from, edges_connected_node_to)
            self.insert_edge(new_edge_id, e.node_to.node_id, e.node_to.LAT, e.node_to.LON,
                             e.node_from.node_id, e.node_from.LAT, e.node_from.LON, e.length,
                             new_edge_ref_links, new_edge_non_ref_links)
            # add_speed_info
            new_edge = self.find_edge(new_edge_id)
            new_edge.speed['AVG_SPEED'] = e.speed['AVG_SPEED']
            new_edge.speed['FREE_FLOW_SPEED'] = e.speed['FREE_FLOW_SPEED']
            new_edge.speed['F_WEEKDAY'] = e.speed['T_WEEKDAY']
            new_edge.speed['T_WEEKDAY'] = e.speed['F_WEEKDAY']
            new_edge.speed['F_SPEED'] = e.speed['T_SPEED']
            new_edge.speed['T_SPEED'] = e.speed['F_SPEED']

            # update old edge REF and NONREF links
            e.edges_connected_node_from.append(str('-') + new_edge_id)
            e.edges_connected_node_to.append(new_edge_id)

            # update REF and NONREF links of nei edges to e.node_from 
            n = e.node_from
            nei_edges = n.edges
            for nei_edge in nei_edges:
                if nei_edge.edge_id != e.edge_id and nei_edge.edge_id != new_edge_id:
                    if nei_edge.node_from.node_id == n.node_id:
                        nei_edge.edges_connected_node_from.append(str('-') + new_edge_id)
                    elif nei_edge.node_to.node_id == n.node_id:
                        nei_edge.edges_connected_node_to.append(str('-') + new_edge_id)

            # update REF and NONREF links of nei edges to e.node_to
            n = e.node_to
            nei_edges = n.edges
            for nei_edge in nei_edges:
                if nei_edge.edge_id != e.edge_id and nei_edge.edge_id != new_edge_id:
                    if nei_edge.node_from.node_id == n.node_id:
                        nei_edge.edges_connected_node_from.append(new_edge_id)
                    elif nei_edge.node_to.node_id == n.node_id:
                        nei_edge.edges_connected_node_to.append(str('-') + new_edge_id)


    '''

    ###############################################################################

    def get_adjacency_matrix(self):
        """Return a matrix, or 2D list.
        Row numbers represent from nodes,
        column numbers represent to nodes.
        Store the edge values in each spot,
        and a 0 if no edge exists."""

        if len(self.node_id_to_num) < len(self.nodes):
            self.build_node_id_to_num()

        max_index = len(self.node_id_to_num)
        adjacency_matrix = np.zeros((max_index, max_index))
        for edg in self.edges:
            from_index, to_index = self.node_id_to_num[edg.ref_node.node_id], self.node_id_to_num[edg.node_to.node_id]
            adjacency_matrix[from_index][to_index] = float(edg.length)
        return adjacency_matrix

    def get_sparse_adjacency_list(self):
        if len(self.node_id_to_num) < len(self.nodes):
            self.build_node_id_to_num()
        if len(self.edge_id_to_num) < len(self.edges):
            self.build_edge_id_to_num()
        return [(e.edge_id, self.edge_id_to_num[e.edge_id], self.node_id_to_num[e.ref_node.node_id],
                 self.node_id_to_num[e.node_to.node_id], float(e.length),
                 e.edges_connected_node_from, e.edges_connected_node_to) for e in self.edges]

    def num_of_subgraphs(self):
        uf = UnionFind()
        for node in self.nodes:
            uf.init_matrix(node.node_id)
            for e in node.edges:
                nei = e.ref_node if node.node_id == e.non_ref_node.node_id else e.non_ref_node
                if nei.node_id in uf.father:
                    uf.union(node.node_id, nei.node_id)

        for node in self.nodes:
            node.subgraph = uf.find(node.node_id)

        # print(uf.father.keys())
        return uf.count

    def plot_graph(self, direction=False, traffic_color=False, day=1, time=36):
        """ Plot all links in the graph, day: 0 == Sun, 1 == Mon, ..., 6 == Sat, time: 0 == 0:00, 1 == 0:15, ...."""

        lines = []
        c = []
        for e in self.edges:
            if direction:
                if e.travel_direction == 'F':
                    lines.append([(float(e.ref_node.LON), float(e.ref_node.LAT)),
                                  (float(e.non_ref_node.LON), float(e.non_ref_node.LAT))])
                elif e.travel_direction == 'T':
                    lines.append([(float(e.non_ref_node.LON) + 1, float(e.non_ref_node.LAT) + 1),
                                  (float(e.ref_node.LON) + 1, float(e.ref_node.LAT) + 1)])
                elif e.travel_direction == 'B':
                    lines.append([(float(e.ref_node.LON), float(e.ref_node.LAT)),
                                  (float(e.non_ref_node.LON), float(e.non_ref_node.LAT))])
                    lines.append([(float(e.non_ref_node.LON) + 1, float(e.non_ref_node.LAT) + 1),
                                  (float(e.ref_node.LON) + 1, float(e.ref_node.LAT) + 1)])
                else:
                    print(e.edge_id, ': no edge direction info!')
            else:
                lines.append([(float(e.ref_node.LON), float(e.ref_node.LAT)),
                              (float(e.non_ref_node.LON), float(e.non_ref_node.LAT))])

            if traffic_color:
                if e.traffic_info['FREE_FLOW_SPEED']:
                    speed_factor = []
                    if e.travel_direction == 'F':
                        if e.traffic_info['F_SPEED']:
                            speed_factor.append(
                                e.traffic_info['F_SPEED'][day][time] / e.traffic_info['FREE_FLOW_SPEED'])
                        else:
                            print(e.edge_id, "F but no F_SPEED traffic info")
                    elif e.travel_direction == 'T':
                        if e.traffic_info['T_SPEED']:
                            speed_factor.append(
                                e.traffic_info['T_SPEED'][day][time] / e.traffic_info['FREE_FLOW_SPEED'])
                        else:
                            print(e.edge_id, "T but no T_SPEED traffic info")
                    elif e.travel_direction == 'B':
                        if e.traffic_info['F_SPEED']:
                            speed_factor.append(
                                e.traffic_info['F_SPEED'][day][time] / e.traffic_info['FREE_FLOW_SPEED'])
                        else:
                            print(e.edge_id, "bidirectional but no F_SPEED traffic info")
                        if e.traffic_info['T_SPEED']:
                            speed_factor.append(
                                e.traffic_info['T_SPEED'][day][time] / e.traffic_info['FREE_FLOW_SPEED'])
                        else:
                            print(e.edge_id, "bidirectional but no T_SPEED traffic info")
                    else:
                        print(e.traffic_info['F_SPEED'], e.traffic_info['T_SPEED'])
                        print(e.edge_id, ': no traffic info!')
                else:
                    print(e.edge_id, ': no free flow speed info!')
                    speed_factor = [2] if e.travel_direction == 'F' or 'T' else [2, 2]

                if not speed_factor:
                    speed_factor = [2] if e.travel_direction == 'F' or 'T' else [2, 2]
                if e.travel_direction == 'B' and len(speed_factor) == 1:
                    if e.traffic_info['F_SPEED']:
                        speed_factor.append(2)
                    else:
                        speed_factor = [2] + speed_factor

                for s in speed_factor:
                    if s == 2:
                        c.append('k')
                    elif s < 0.5:
                        c.append('r')
                    elif 0.5 <= s < 0.75:
                        c.append('y')
                    elif 0.75 <= s < 1:
                        c.append('g')
                    else:
                        c.append('m')

        if traffic_color:
            line_segments = mc.LineCollection(lines, colors=c)
        else:
            line_segments = mc.LineCollection(lines)

        fig, ax = plt.subplots()
        ax.add_collection(line_segments)
        ax.autoscale()

        if direction:
            x, y, u, v = [], [], [], []
            for line in lines:
                x.append(line[0][0])
                y.append(line[0][1])
                u.append(line[1][0] - line[0][0])
                v.append(line[1][1] - line[0][1])

            ax.quiver(x, y, u, v, scale_units='xy', angles='xy', scale=2, headwidth=7)
            ax.set_aspect('equal', 'box')

        plt.show()
        return

    def plot_subgraphs(self, n):
        """ plot: colored n largest disconnected subgraphs"""
        colors = np.concatenate((np.random.rand(3, n), np.ones((1, n))))

        subgraph_dict = defaultdict(int)
        for node in self.nodes:
            subgraph_dict[node.subgraph] += 1

        subgraph_ids = [item[0] for item in sorted(subgraph_dict.items(), key=lambda v: v[1], reverse=True)]
        subgraph_ids = subgraph_ids[:n]

        color_dic = {}
        for i, subgraph in enumerate(subgraph_dict):
            color_dic[subgraph] = colors[:, i]

        lines = []
        c = []
        for e in self.edges:
            if e.ref_node.subgraph in subgraph_ids:
                lines.append([(float(e.ref_node.LON), float(e.ref_node.LAT)),
                              (float(e.non_ref_node.LON), float(e.non_ref_node.LAT))])
                c.append(color_dic[e.ref_node.subgraph])

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

    def find_path(self, start_node_id, end_node_id, path=[]):
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
            nei = e.ref_node if node.node_id == e.non_ref_node.node_id else e.non_ref_node
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
            return [start_node_id], 0

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
                nei = e.ref_node if node.node_id == e.non_ref_node.node_id else e.non_ref_node
                new_dist = dist[node.node_id]['val'] + float(e.length)
                if not nei.visited:
                    nei.visited = True
                    queue.append(nei)
                    dist[nei.node_id]['val'] = new_dist
                    dist[nei.node_id]['parent'] = node.node_id
                elif new_dist < dist[nei.node_id]['val']:
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

    def dijkstra(self, start_node_id, end_node_id, K=float('inf')):
        """dijkstra uses a priority queue. The output is the shortest path and path_length.
        ARGUMENTS: start_node_id, end_node_id, K is the maximum number of nodes we want to search (approximate)
        RETURN: minimal time path"""
        node = self.find_node(start_node_id)
        pq = [(0, 0, node)]

        # dist = {start_node_id: (0, -1)}  # distance to the start, parent
        dist = defaultdict(dict)
        dist[start_node_id]['val'] = 0
        dist[start_node_id]['parent'] = -1  # distance to the start, parent
        # max_queue_length = 0
        sq = 0  # index the node inserted to the priority queue
        while pq:
            # max_queue_length = max(len(pq), max_queue_length)
            obj, _, node = heappop(pq)

            if sq > K:
                break

            if node.node_id == end_node_id:  # terminal condition
                path = [end_node_id]
                par = dist[end_node_id]['parent']
                while par != -1:
                    path.insert(0, par)
                    par = dist[par]['parent']
                return path, dist[end_node_id]['val']
                # return path, dist[end_node_id]['val'], max_queue_length

            for e in node.edges:
                nei = e.ref_node if node.node_id == e.non_ref_node.node_id else e.non_ref_node

                new_obj = obj + float(e.length)  # shortest length
                if nei.node_id not in dist:
                    dist[nei.node_id]['val'] = new_obj
                    dist[nei.node_id]['parent'] = node.node_id
                    sq += 1
                    heappush(pq, (new_obj, sq, nei))
                elif new_obj < dist[nei.node_id]['val']:
                    dist[nei.node_id]['val'] = new_obj
                    dist[nei.node_id]['parent'] = node.node_id
                    sq += 1
                    heappush(pq, (new_obj, sq, nei))

        print('There is no path from node {} to node {} within K hops'.format(start_node_id, end_node_id))
        return None

    def nx_dijkstra(self, start_node_id, end_node_id):
        """To be updated"""
        G = nx.DiGraph(directed=True)
        arcs = []
        for e in self.edges:
            arcs.append(
                (self.node_id_to_num[e.ref_node.node_id], self.node_id_to_num[e.non_ref_node.node_id], float(e.length)))
        G.add_weighted_edges_from(arcs)
        G = nx.dodecahedral_graph()
        nx.draw(G)
        # nx.draw_networkx_edges(G, arrows=True)
        plt.show()
        # print(nx.dijkstra_path(G, self.node_id_to_num[start_node_id], self.node_id_to_num[end_node_id]))
        return None


if __name__ == "__main__":
    g = Graph()






