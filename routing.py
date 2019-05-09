from collections import defaultdict
from heapq import heapify, heappush, heappop

import sys
import math
import itertools
import numpy as np
from gurobipy import *
from gurobipy import Model
from gurobipy import GRB
from gurobipy import quicksum


class Routing():
    def __init__(self, graph):
        # input shall be a graph with traffic info
        self.graph = graph

    def __repr__(self):
        return '{self.__class__.__name__}'.format(self=self)

    def bfs(self, start_node_id, end_node_id):
        """BFS iterating through a node's edges.
        !!!Travel direction is considered!!!
        The output is the shortest path + length.
        ARGUMENTS: start_node_id, end_node_id
        RETURN: path (node_ids), path_length."""
        if start_node_id == end_node_id:
            return [start_node_id], 0

        self.graph._clear_visited()
        node = self.graph.find_node(start_node_id)
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
                if e.get_travel_direction() == 'B':
                    nei = e.ref_node if node.node_id == e.non_ref_node.node_id else e.non_ref_node
                elif e.get_travel_direction() == 'F':
                    nei = e.non_ref_node if node.node_id == e.ref_node.node_id else None
                elif e.get_travel_direction() == 'T':
                    nei = e.ref_node if node.node_id == e.non_ref_node.node_id else None
                else:
                    print(e.edge_id, ': travel direction is missing!')

                if nei:
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

    def dijkstra(self, start_node_id, end_node_id, K=float('inf'), day=1, time=36):
        """dijkstra uses a priority queue.
        !!! Travel direction is considered !!!
        The output is the shortest path and path_length.
        ARGUMENTS: start_node_id, end_node_id, K is the maximum number of nodes we want to search (approximate)
        RETURN: minimal time path"""
        node = self.graph.find_node(start_node_id)
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
                if e.get_travel_direction() == 'B':
                    nei = e.ref_node if node.node_id == e.non_ref_node.node_id else e.non_ref_node
                elif e.get_travel_direction() == 'F':
                    nei = e.non_ref_node if node.node_id == e.ref_node.node_id else None
                elif e.get_travel_direction() == 'T':
                    nei = e.ref_node if node.node_id == e.non_ref_node.node_id else None
                else:
                    print(e.edge_id, ': travel direction is missing!')

                if not nei:
                    continue

                try:
                    if nei == e.non_ref_node:
                        speeds = e.traffic_info['F_SPEED']
                        if not speeds:
                            continue
                    else:
                        speeds = e.traffic_info['T_SPEED']
                        if not speeds:
                            continue
                    speed = speeds[day][time]
                except:
                    speed = e.traffic_info['AVG_SPEED']
                    print(e.edge_id, e.traffic_info)
                # new_obj = dist[node.node_id]['val'] + float(e.length)  # shortest length

                new_obj = obj + float(e.length) / (speed * 1000 / 3600)  # minimal time unit: seconds

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

    def cdf_dijkstra(self, start_node_id, end_node_id, battery_level, day=1, time=36):
        """dijkstra uses a priority queue. The output is the shortest path and path_length.
        ARGUMENTS: start_node_id, end_node_id, K is the maximum number of nodes we want to search (approximate)
        RETURN: path with minimal energy consumption using CDF mode"""

        Cele = 0.114  # $ / kWh   # Energy price
        Cgas = 2.75  # $ / gallon
        mu_CD_Value = [3.137, 4.386, 4.135]  # mi / kWh
        mu_CS_Value = [28.88, 49.034, 47.11]  # mi / gal

        node = self.graph.find_node(start_node_id)
        pq = [(0, 0, node, battery_level)]

        # dist = {start_node_id: (0, -1)}  # start node: (cost, parent)
        dist = defaultdict(dict)
        dist[start_node_id]['val'] = 0
        dist[start_node_id]['parent'] = -1  # distance to the start, parent
        # max_queue_length = 0
        sq = 0  # index the node inserted to the priority queue
        while pq:
            # max_queue_length = max(len(pq), max_queue_length)
            obj, _, node, battery_level = heappop(pq)

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

                try:
                    if nei == e.non_ref_node:
                        speeds = e.speed['F_SPEED']
                        if not speeds:
                            continue
                    else:
                        speeds = e.speed['T_SPEED']
                        if not speeds:
                            continue
                    speed = speeds[day][time]
                except:
                    speed = e.speed['AVG_SPEED']
                    print(e.edge_id, e.speed)

                if speed < 20:
                    mode = 0  # low
                elif 20 <= speed < 40:
                    mode = 1  # median
                elif speed >= 40:
                    mode = 2  # high
                else:
                    print('speed error, speed = ', speed)

                if battery_level:
                    if battery_level >= float(e.length) / 1600 / mu_CD_Value[mode]:
                        cost_per_link = Cele * float(e.length) / 1600 / mu_CD_Value[mode]
                        new_battery_level = battery_level - float(e.length) / 1600 / mu_CD_Value[mode]
                    else:
                        cost_per_link = Cele * battery_level + \
                                        Cgas * (float(e.length) / 1600 - mu_CD_Value[mode] * battery_level) / \
                                        mu_CS_Value[mode]
                        new_battery_level = 0
                else:
                    cost_per_link = Cgas * float(e.length) / 1600 / mu_CS_Value[mode]
                    new_battery_level = 0

                new_obj = obj + cost_per_link

                if nei.node_id not in dist:
                    dist[nei.node_id]['val'] = new_obj
                    dist[nei.node_id]['parent'] = node.node_id
                    sq += 1
                    heappush(pq, (new_obj, sq, nei, new_battery_level))
                elif new_obj < dist[nei.node_id]['val']:
                    dist[nei.node_id]['val'] = new_obj
                    dist[nei.node_id]['parent'] = node.node_id
                    sq += 1
                    heappush(pq, (new_obj, sq, nei, new_battery_level))

        print('There is no path from node {} to node {} within K hops'.format(start_node_id, end_node_id))
        return None

    def crptc_1D(self, start_node_id, end_node_id, battery_level, day=1, time=36):

        Cele = 0.114  # $ / kWh   # Energy price
        Cgas = 2.75  # $ / gallon
        mu_CD_Value = [3.137, 4.386, 4.135]  # mi / kWh
        mu_CS_Value = [28.88, 49.034, 47.11]  # mi / gal

        graph_adj = self.graph.get_sparse_adjacency_list()

        x = {}
        y = {}
        z = {}
        numberOfEdges = len(graph_adj)
        d = np.zeros(numberOfEdges)
        c = np.zeros((numberOfEdges,), dtype=int)

        m = Model('MINLP')
        # Create variables
        for i in range(numberOfEdges):
            x[i] = m.addVar(vtype=GRB.BINARY, name='x%d' % i)
            y[i] = m.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name='y%d' % i)
            z[i] = m.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name='z%d' % i)

        m.update()

        # Add Constraints
        # Zij <= Yij
        # Xij - Zij + Yij <= 1
        # Xij >= Zij
        for i in range(numberOfEdges):
            m.addConstr(z[i] <= y[i])
            m.addConstr(x[i] - z[i] + y[i] <= 1)
            m.addConstr(x[i] >= z[i])

        # dij and cij
        for i in range(numberOfEdges):
            # dij
            d[i] = float(graph_adj[i][4])
            # cij
            e = self.graph.find_edge(graph_adj[i][0])

            try:
                speeds = e.speed['F_SPEED']
                # if not speeds:
                #    speeds = e.speed['T_SPEED']
                speed = speeds[day][time]
            except:
                speed = e.speed['AVG_SPEED']
                print(e.edge_id)

            # speed = edg.speed['AVG_SPEED']
            if speed < 20:
                mode = 0  # low
            elif 20 <= speed < 40:
                mode = 1  # median
            elif speed >= 40:
                mode = 2  # high
            else:
                print('speed error, speed = ', speed)
            c[i] = mode

        # Add Constraints
        # Sum (dij/mu_CD*zij <= battery_level)
        m.addConstr(quicksum((d[i] / 1600 / mu_CD_Value[c[i]]) * z[i] for i in range(numberOfEdges)) <= battery_level)

        # Add Constraints
        # Sum (outgoing edges) - Sum (incoming edges) = bi
        # bi = 0 if start_node and end_node are connected by one edge
        # bi = 1 if e = start edge
        # bi = -1 if e = end edge
        # bi = 0 if otherwise
        for e in self.graph.edges:
            in_list = []
            out_list = []

            incoming_edges, outgoing_edges = self.graph.find_incoming_outgoing_edges(e.edges_connected_node_from,
                                                                                     e.edges_connected_node_to)

            for e_in in incoming_edges:
                num = self.graph.find_edge_num(e_in)
                if num != "edge_id doesn't exist. (find_edge_num)":
                    in_list.append(num)
            for e_out in outgoing_edges:
                num = self.graph.find_edge_num(e_out)
                if num != "edge_id doesn't exist. (find_edge_num)":
                    out_list.append(num)
            # Add Constraints
            # when start node and end node are ref_node and non_ref_node of one edge
            if e.ref_node.node_id == start_node_id and e.non_ref_node.node_id == end_node_id:

                m.addConstr((quicksum(x[i] for i in out_list) - quicksum(x[i] for i in in_list)) == 0)

            # edges connected to start node
            elif e.ref_node.node_id == start_node_id or e.non_ref_node.node_id == start_node_id:

                m.addConstr((quicksum(x[i] for i in out_list) - quicksum(x[i] for i in in_list)) == 1)

            # edges connected to end node
            elif e.non_ref_node.node_id == end_node_id or e.ref_node.node_id == end_node_id:

                m.addConstr((quicksum(x[i] for i in out_list) - quicksum(x[i] for i in in_list)) == -1)

            # other edges
            else:

                m.addConstr((quicksum(x[i] for i in out_list) - quicksum(x[i] for i in in_list)) == 0)

        Alpha = np.zeros(numberOfEdges)
        Beta = np.zeros(numberOfEdges)

        # Solving the optimization problem
        for i in range(numberOfEdges):
            Alpha[i] = Cgas * d[i] / 1600 / mu_CS_Value[c[i]]
            Beta[i] = Cele * d[i] / 1600 / mu_CD_Value[c[i]] - Cgas * d[i] / 1600 / mu_CS_Value[c[i]]

        m.setObjective(quicksum((Alpha[i] * x[i] + Beta[i] * z[i]) for i in range(numberOfEdges)), GRB.MINIMIZE)

        m.optimize()

        # print crptc_energy-cost
        print('\n')
        print('crptc energy cost: %f' % m.objVal)

        # plot crptc_path
        count = 0
        nodes_num = []
        path = []

        for v in m.getVars():
            if v.x != 0:
                if v.varname[0] == 'x' or v.varname[0] == 'z':
                    print(v.varName, v.x)
        print('\n')
        for v in m.getVars():
            if v.x != 0:
                if v.varname[0] == 'x':
                    edge_number = int(v.varname[1:])
                    print('d%d = %f' % (edge_number, d[edge_number]))
                    print('c%d = %d' % (edge_number, c[edge_number]))
        '''
        print('\n')
        for v in m.getVars():
            if v.x != 0:
                if v.varname[0] == 'y':
                    print(v.varName, v.x)
        '''
        for v in m.getVars():
            if v.x != 0:
                if v.varname[0] == 'x':
                    count = count + 1
                    e_num = v.varName[1:]
                    e_id = self.graph.get_edge_id(int(e_num))
                    e = self.graph.find_edge(e_id)
                    ref_node_id = e.ref_node.node_id
                    non_ref_node_id = e.non_ref_node.node_id
                    nodes_num.append((ref_node_id, non_ref_node_id))

        if nodes_num:
            for e in nodes_num:
                if e[0] == start_node_id:
                    path.append(e[0])
                    path.append(e[1])
                    nodes_num.remove(e)
                    break

            for i in range(count - 1):
                for e in nodes_num:
                    node_id = e[0]
                    if node_id == path[i + 1]:
                        path.append(e[1])
                        nodes_num.remove(e)
        return path, c, d

    def crptc_2D(self, start_node_id, end_node_id, battery_level, day=1, time=36):

        Cele = 0.114  # $ / kWh   # Energy price
        Cgas = 2.75  # $ / gallon
        mu_CD_Value = [3.137, 4.386, 4.135, 1]  # mi / kWh
        mu_CS_Value = [28.88, 49.034, 47.11, 1]  # mi / gal

        adjacency_matrix = self.graph.get_adjacency_matrix()
        max_index = len(adjacency_matrix)

        x = {}
        y = {}
        z = {}
        d = np.zeros((max_index, max_index))
        c = np.zeros((max_index, max_index))

        m = Model('MINLP')
        # Create variables
        for i in range(max_index):
            for j in range(max_index):
                x[(i, j)] = m.addVar(vtype=GRB.BINARY, name='x%d,%d' % (i, j))
                y[(i, j)] = m.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name='y%d,%d' % (i, j))
                z[(i, j)] = m.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name='z%d,%d' % (i, j))

        m.update()

        # Add Constraints
        # Zij <= Yij
        # Xij - Zij + Yij <= 1
        # Xij >= Zij
        for i in range(max_index):
            for j in range(max_index):
                m.addConstr(z[(i, j)] <= y[(i, j)])
                m.addConstr(x[(i, j)] - z[(i, j)] + y[(i, j)] <= 1)
                m.addConstr(x[(i, j)] >= z[(i, j)])

        # or:
        m.addConstrs(x[i, j] == 0 for i in range(max_index) for j in range(max_index) if adjacency_matrix[i][j] == 0)

        # dij
        for i in range(max_index):
            for j in range(max_index):
                d[(i, j)] = adjacency_matrix[i][j]

                # cij
        for i in range(max_index):
            for j in range(max_index):
                if adjacency_matrix[i][j] != 0:
                    ref_node_id = self.graph.get_node_id(i)
                    non_ref_node_id = self.graph.get_node_id(j)
                    edge = self.graph.find_edge_by_nodes(ref_node_id, non_ref_node_id)

                    try:
                        speeds = edge.speed['F_SPEED']
                        speed = speeds[day][time]
                    except:
                        speed = edge.speed['AVG_SPEED']

                    ############################speed = edge.speed['AVG_SPEED']
                    if speed < 20:
                        mode = 0  # low
                    elif 20 <= speed < 40:
                        mode = 1  # median
                    elif speed >= 40:
                        mode = 2  # high
                    else:
                        print('speed error, speed = ', speed)

                    c[(i, j)] = mode
                else:
                    c[(i, j)] = 3

        # Add Constraints
        m.addConstr(quicksum(quicksum((d[(i, j)] / 1600 / mu_CD_Value[int(c[(i, j)])] * z[(i, j)]) \
                                      for i in range(max_index)) for j in range(max_index)) <= battery_level)

        # Add Constraints
        for i in range(max_index):
            node_id = self.graph.get_node_id(i)
            if node_id == start_node_id:
                m.addConstr(quicksum(x[(i, j)] for j in range(max_index)) \
                            - quicksum(x[(j, i)] for j in range(max_index)) == 1)
            elif node_id == end_node_id:
                m.addConstr(quicksum(x[(i, j)] for j in range(max_index)) \
                            - quicksum(x[(j, i)] for j in range(max_index)) == -1)
            else:
                m.addConstr(quicksum(x[(i, j)] for j in range(max_index)) \
                            - quicksum(x[(j, i)] for j in range(max_index)) == 0)

        Alpha = {}
        Beta = {}
        # Solving the optimization problem
        for i in range(max_index):
            for j in range(max_index):
                Alpha[(i, j)] = Cgas * d[(i, j)] / 1600 / mu_CS_Value[int(c[(i, j)])]
                Beta[(i, j)] = Cele * d[(i, j)] / 1600 / mu_CD_Value[int(c[(i, j)])] - \
                               Cgas * d[(i, j)] / 1600 / mu_CS_Value[int(c[(i, j)])]

        m.setObjective(quicksum(quicksum((Alpha[(i, j)] * x[(i, j)] + Beta[(i, j)] * z[(i, j)]) \
                                         for i in range(max_index)) for j in range(max_index)))

        m.optimize()

        # print crptc_energy-cost
        print('\n')
        print('crptc energy cost: %f' % m.objVal)

        # plot crptc_path
        count = 0
        nodes_num = []
        start_node_num = self.graph.node_id_to_num[start_node_id]
        path = []

        for v in m.getVars():
            if v.x != 0:
                if v.varname[0] == 'x' or v.varname[0] == 'z':
                    print(v.varName, v.x)
        '''
        print('\n')
        for v in m.getVars():
            if v.x != 0:
                if v.varname[0] == 'y':
                    print(v.varName, v.x)
        '''
        for v in m.getVars():
            if v.x != 0:
                if v.varname[0] == 'x':
                    count = count + 1
                    name = v.varName[1:]
                    node_numbers = name.split(',')
                    nodes_num.append((node_numbers[0], node_numbers[1]))

        if nodes_num:
            for e in nodes_num:
                if int(e[0]) == start_node_num:
                    path.append(self.graph.get_node_id(int(e[0])))
                    path.append(self.graph.get_node_id(int(e[1])))
                    nodes_num.remove(e)
                    break

            for i in range(count - 1):
                for e in nodes_num:
                    node_id = self.graph.get_node_id(int(e[0]))
                    if node_id == path[i + 1]:
                        path.append(self.graph.get_node_id(int(e[1])))
                        nodes_num.remove(e)
        return path, c, d




