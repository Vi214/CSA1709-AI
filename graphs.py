class graphs:
    def __init__(self):
        self.graph = {}

    def add_edge(self, node, neighbors):
        """
        Adds an edge to the graph. 'neighbors' should be a list of (neighbor_node, cost) tuples.
        """
        self.graph[node] = neighbors

    def depth_first_search(self, start, goal):
        """
        Performs a Depth-First Search to find a path from 'start' to 'goal'.
        Prints the explored tree.  This implementation prioritizes exploration
        and does not guarantee the *shortest* path in terms of cost.

        Returns:
        - path: A list representing a path from start to goal (or None if no path exists).
        - cost: The total cost of the path (or None if no path exists).
        """

        visited = set()
        path = []
        total_cost = 0

        def dfs(node, current_path, current_cost, level):
            nonlocal path, total_cost  # Allow modification of outer scope variables

            visited.add(node)
            current_path.append(node)

            print(f"{'  ' * level}Visiting: {node}") # Print the tree structure

            if node == goal:
                path = current_path[:]  # Copy the path
                total_cost = current_cost
                return True  # Signal that the goal has been found

            for neighbor, edge_cost in self.graph.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor, current_path, current_cost + edge_cost, level + 1):
                        return True  # Goal found in a sub-branch, propagate the signal

            # If the goal was not found in this branch, backtrack.
            current_path.pop()  # Remove the current node from the path

            return False

        if dfs(start, [], 0, 0):
            return path, total_cost
        else:
            return None, None

# Example Usage:
if __name__ == '__main__':
    graph = graphs()
    graph.add_edge('A', [('B', 5), ('C', 2)])
    graph.add_edge('B', [('D', 4), ('E', 1)])
    graph.add_edge('C', [('F', 3)])
    graph.add_edge('D', [])
    graph.add_edge('E', [('F', 7)])
    graph.add_edge('F', [])

    start_node = 'A'
    goal_node = 'F'

    path, cost = graph.depth_first_search(start_node, goal_node)

    if path:
        print(f"Path from {start_node} to {goal_node}: {path}")
        print(f"Total cost: {cost}")
    else:
        print(f"No path found from {start_node} to {goal_node}")
