# Traffic Map and Eco-Routing for Smart Transportation
Drivers who commute everyday has experienced the misery of rush hours. Our project aims to ease commuting and the resulting air pollution by developing efficient traffic control and routing strategies for building a smart transportation system.

## Software Architecture
### Class
#### 1. car.py 
   * Car(): tradictional gas car 
   * ECar(): eletrical car
   * HCar(): hybrid car powered by gas and battery
#### 2. graph.py
* Node()
* Edge()
* Graph() 
   * num_of_subgraphs
   * search algorithms on geographic graph without travel directions: 
      * dfs(find_path)
      * bfs
      * dijkstra
* UnionFind() 
   * find the num_of_subgraphs
#### 3. routing.py
* Routing()
   * search algorithms with travel directions: 
      * bfs
      * dijkstra
      * cdf_dijkstra: charge depleting first, dijkstra with energy cost (To be updated)
      * crptc: combined routing and power-train control, MILP solved by Gurobi (To be updated)
#### TrafficData.py (main file)
* Workflow
   * read data from json
   * build graph
   * add traffic info
   * analyze graph 
   * find path/route
