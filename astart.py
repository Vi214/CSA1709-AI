from queue import PriorityQueue

class astart:
    def __init__(self):
        self.nodes = {}
    
    def add_node(self, name, heuristic):
        self.nodes[name] = {"heuristic": heuristic, "neighbors": {}}
    
    def add_edge(self, node1, node2, cost):
        self.nodes[node1]["neighbors"][node2] = cost
        self.nodes[node2]["neighbors"][node1] = cost
    
    def a_star_search(self, start, goal):
        open_list = PriorityQueue()
        open_list.put((0, start))
        came_from = {}
        g_score = {node: float('inf') for node in self.nodes}
        g_score[start] = 0
        
        while not open_list.empty():
            _, current = open_list.get()
            
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path
            
            for neighbor, cost in self.nodes[current]["neighbors"].items():
                temp_g_score = g_score[current] + cost
                
                if temp_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = temp_g_score
                    f_score = temp_g_score + self.nodes[neighbor]["heuristic"]
                    open_list.put((f_score, neighbor))
                    print(f"Node: {neighbor}, g(n): {temp_g_score}, h(n): {self.nodes[neighbor]['heuristic']}, f(n): {f_score}")
        
        return None  # No path found

# Example Usage
graph = astart()
graph.add_node("A", 6)
graph.add_node("B", 4)
graph.add_node("C", 4)
graph.add_node("D", 2)
graph.add_node("E", 0)
graph.add_edge("A", "B", 1)
graph.add_edge("A", "C", 3)
graph.add_edge("B", "D", 3)
graph.add_edge("C", "D", 1)
graph.add_edge("D", "E", 2)

start_node = "A"
goal_node = "E"
path = graph.a_star_search(start_node, goal_node)
print("Optimal Path:", path)
