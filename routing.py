from collections import defaultdict
from heapq import heapify, heappush, heappop

import sys
import math
import itertools
from gurobipy import *

class Routing():
    def __init__(self, graph):
        self.graph = graph

    def __repr__(self):
        return '{self.__class__.__name__}'.format(self=self)

    def bfs(self, start_node_id, end_node_id):
        """BFS iterating through a node's edges. The output is the shortest path + length.
        ARGUMENTS: start_node_id, end_node_id
        RETURN: path (node_ids), path_length."""
        if start_node_id == end_node_id:
            return  [start_node_id], 0

        node = self.graph.find_node(start_node_id)
        self.graph._clear_visited()
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

    def dijkstra(self, start_node_id, end_node_id, K = float('inf'), day = 1, time = 36):
        """dijkstra uses a priority queue. The output is the shortest path and path_length.
        ARGUMENTS: start_node_id, end_node_id, K is the maximum number of nodes we want to search (approximate)
        RETURN: minimal time path"""
        node = self.graph.find_node(start_node_id)
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
                try:
                    if nei == e.node_to:
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
                # new_obj = dist[node.node_id]['val'] + float(e.length)  # shortest length

                new_obj = obj + float(e.length) / (speed * 1000 / 3600)   #minimal time unit: seconds

                if nei.node_id not in dist:
                    dist[nei.node_id]['val'] = new_obj
                    dist[nei.node_id]['parent']= node.node_id
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
                nei = e.node_from if node.node_id == e.node_to.node_id else e.node_to

                try:
                    if nei == e.node_to:
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

    def crptc(self, start_node_id, end_node_id, battery_level,day=1, time=36):
        m = Model('MINLP')
        # Create variables

        x = m.addVar(vtype=GRB.BINARY, name='x')
        y = m.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name='y')



