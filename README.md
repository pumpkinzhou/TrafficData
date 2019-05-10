# Traffic Map and Eco-Routing for Smart Transportation
Drivers who commute everyday has experienced the misery of rush hours. Our project aims to ease commuting and the resulting air pollution by developing efficient traffic control and routing strategies for building a smart transportation system.

## Software Architecture
### Class
#### car.py: 
1. Car(): tradictional gas car 
2. ECar(): eletrical car
3. HCar(): hybrid car powered by gas and battery
#### graph.py
1. Node()

2. Edge()

3. Graph() 
   * num_of_subgraphs
   * search algorithms on geographic graph without travel directions: 
      * dfs(find_path)
      * bfs
      * dijkstra

4. UnionFind() 
   * find the num_of_subgraphs
#### routing.py
1. Routing()
   * search algorithms with travel directions: 
      * bfs
      * dijkstra
      * cdf_dijkstra: charge depleting first, dijkstra with energy cost (To be updated)
      * crptc: combined routing and power-train control, MILP solved by Gurobi (To be updated)
#### TrafficData.py (main file)
##### Workflow
1. read data from json
2. build graph
3. add traffic info
4. graph analysis
5. path finding
