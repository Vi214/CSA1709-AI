from collections import deque

class Graph:
    def __init__(self):
        self.graph = {}

    def add_edge(self, node, neighbors):
        """
        Adds an edge to the graph. 'neighbors' should be a list of (neighbor_node, cost) tuples.
        """
        self.graph[node] = neighbors

    def breadth_first_search(self, start, goal):
        """
        Performs a Breadth-First Search to find the shortest path from 'start' to 'goal'.  Prints the explored tree.

        Returns:
        - path: A list representing the shortest path from start to goal (or None if no path exists).
        - cost: The total cost of the path (or None if no path exists).
        """

        queue = deque([(start, [start], 0)])  # (node, path, cost)
        visited = {start}
        level = 0  # Keep track of the level for printing the tree structure

        print("Explored Tree:")
        print(f"Level {level}: {start}") # Print initial node

        while queue:
            node, path, cost = queue.popleft()

            if node == goal:
                print(f"\nGoal {goal} found!")
                return path, cost

            neighbors = self.graph.get(node, [])
            if neighbors:
                level += 1
                print(f"Level {level}: ", end="")

            neighbor_strings = []
            for neighbor, edge_cost in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_path = path + [neighbor]
                    new_cost = cost + edge_cost
                    queue.append((neighbor, new_path, new_cost))
                    neighbor_strings.append(f"{neighbor}(cost:{edge_cost})")
            print(", ".join(neighbor_strings))  # Print neighbors at this level

        print(f"\nNo path found from {start} to {goal}")
        return None, None  # No path found

# Example Usage:
if __name__ == '__main__':
    graph = Graph()
    graph.add_edge('A', [('B', 5), ('C', 2)])
    graph.add_edge('B', [('D', 4), ('E', 1)])
    graph.add_edge('C', [('F', 3)])
    graph.add_edge('D', [])
    graph.add_edge('E', [('F', 7)])
    graph.add_edge('F', [])

    start_node = 'A'
    goal_node = 'F'

    path, cost = graph.breadth_first_search(start_node, goal_node)

    if path:
        print(f"Path from {start_node} to {goal_node}: {path}")
        print(f"Total cost: {cost}")
    else:
        print(f"No path found from {start_node} to {goal_node}")
