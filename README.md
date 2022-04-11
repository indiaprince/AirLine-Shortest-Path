# AirLine-Shortest-Path
Airline shortest path depending on the weather conditions using dijkstra algorithm
# Algorithm
#### **1) Importing necessary libraries**
``` python
from google.colab import drive
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from queue import PriorityQueue 
import heapq
%matplotlib inline
```
---
#### **2) Mounting Google Drive**
```ts
drive.mount('/content/drive')
```
---
#### **3) Importing Data**
DATA : South_Korea_airport_toy_example.csv
```ts
df = pd.read_csv('/content/drive/MyDrive/~~~~/South_Korea_airport_toy_example.csv')
df.head()
```
[Expected Result:]
<img width="522" alt="Screen Shot 2022-04-08 at 3 42 09 AM" src="https://user-images.githubusercontent.com/89503971/162345122-02660d69-e7cd-4bac-988f-5d6cccfd06b9.png">

<il>Name : Name of Airports
    
    
<il>Type : Whether International or Domestic
    
    
<il>IATA : Code Given by IATA
    
    
<il>ICAO : RK	South Korea (Republic of Korea)
    
    
<il>Longitude (deg) : Latitude of Airport
    
    
<il>Latitude (deg) : Longitude of Airport

---
#### **4) Constructing adjacency lists**
```ts
print(df['Name'])
airports = [[1,2,3], [0,2,5],[0,1,4,5], [0,4], [2,3,5], [1,2,4]]
print(airports)

for i in range(len(airports)):
  print(f'Vertex {i} {df["Name"][i]}')
  for node in airports[i]:
    print(f'\t is connected to {node} {df["Name"][node]}')
```
---
#### **5) Idea Airpots Routes (Dense)**

---
#### **6) Calculating distance between airports**
Estimation of Weight of Edges : Haversine Formula
 - The haversine formula determines the great-circle distance between two points on a sphere given their longitudes and latitudes.
 -  Distance between two points are look like an arc in three dimension.

---
#### **7. CONSTRUCTING CURRENT AIRPORT ROUTES WITH DISTANCE**
IN FINE WEATHER CONDITION

```ts
airports_route_fine_condition = FineCond_Graph_Construct(df)
```
---
#### **8) Constructing Graph Node, Vertex and Dijkstra method**

```ts
D = fine_graph.dijkstra(1)
print(D)
```
```ts
#print(fine_graph.visited)
route = FindPath(df,fine_graph,1,3)
print(route)
```

#(1: JEJU) TO (3: YANGYANG)      : **581.63 km**  is the shortest path

---
#### **10) Area of Severe Weather Condition**
```ts
Polygon_information = ([(128.7, 36.6), (128.7, 37.6), (129.3, 37.6), (129.3, 36.6)])

#### **11) Checking whether the edge is in Severe Condition**
```ts
def isPathInSevereCond(x_ls,y_ls, route_x, route_y):  # to find intersaction between path and bad weather
  mn_x, mx_x =  min(x_ls),max(x_ls)
  mn_y, mx_y =  min(y_ls),max(y_ls)

  route_x_s,route_x_e = min(route_x), max(route_x)
  route_y_s,route_y_e = min(route_y), max(route_y)

  for i in (np.arange(route_x_s,route_x_e,0.01)):
      y_val = round(((route_y[1] - route_y[0])/(route_x[1] - route_x[0]))*(i-route_x[0]) + route_y[0],4)
      if(mn_x<=i and i<=mx_x):
        if(mn_y<=y_val and y_val<=mx_y): return True
  return False
```
#### **12) CONSTRUCTING CURRENT AIRPORT ROUTES WITH DISTANCE**
IN SEVERE WEATHER CONDITION

```ts 
airports_route_severe_condition = SevereCond_Graph_Construct(df,Polygon_information)
```
```ts 
airports_route_severe_condition
```
---
#### **13) Shortest Path in Severe Condition**
```ts 
severe_graph = Graph(6)
for i in range(len(airports_route_severe_condition)):
  for node,dist in airports_route_severe_condition[i]:
    severe_graph.add_edge(i, node, dist)
```
```ts
D = severe_graph.dijkstra(1)
print(D)
```
```ts
route = FindPath(df, severe_graph,1,3)
print(route)
```
---
#### **14) Result in Other Severe Conditions**
```ts
Severe_cond1 = ([(129.0, 35.5), (129.0, 37.6), (129.3, 37.6), (129.3, 35.5)])
cond1_routes = SevereCond_Graph_Construct(df,Severe_cond1)
severe_cond1_graph = Graph(6)
for i in range(len(cond1_routes)):
  for node,dist in cond1_routes[i]:
    #print(f'{df["Name"][i]} to {df["Name"][node]} {dist}')
    severe_cond1_graph.add_edge(i, node, dist)
D = severe_cond1_graph.dijkstra(1)
print(D)
route = FindPath(df, severe_cond1_graph,1,3)
print(route)
```
```ts
Severe_cond2 = ([(127.5, 37.3), (127.5, 38.0), (128.0, 38.0), (128.0, 37.3)])
cond2_routes = SevereCond_Graph_Construct(df,Severe_cond2)
severe_cond2_graph = Graph(6)
for i in range(len(cond2_routes)):
  for node,dist in cond2_routes[i]:
    severe_cond2_graph.add_edge(i, node, dist)
D = severe_cond2_graph.dijkstra(1)
print(D)
route = FindPath(df, severe_cond2_graph,1,3)
print(route)
```

