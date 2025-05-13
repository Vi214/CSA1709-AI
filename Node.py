import heapq

class Node:
    def __init__(self, state, parent=None, cost=0, heuristic=0):
        self.state = state
        self.parent = parent
        self.cost = cost
        self.heuristic = heuristic

    def __lt__(self, other):
        # Used for priority queue (heap) ordering.  Nodes with lower f-scores are prioritized.
        return (self.cost + self.heuristic) < (other.cost + other.heuristic)

def a_star(graph, start, goal, heuristic_func):
    """
    A* search algorithm to find the shortest path from a start node to a goal node.

    Args:
        graph: A dictionary representing the graph where keys are nodes (states)
               and values are dictionaries of neighboring nodes with their
               corresponding costs (distances).  Example:
               graph = {
                   'A': {'B': 10, 'C': 15},
                   'B': {'A': 10, 'D': 12, 'E': 15},
                   'C': {'A': 15, 'F': 12},
                   'D': {'B': 12, 'G': 10},
                   'E': {'B': 15, 'H': 10},
                   'F': {'C': 12, 'I': 10},
                   'G': {'D': 10},
                   'H': {'E': 10},
                   'I': {'F': 10}
               }
        start: The starting node (state).
        goal: The goal node (state).
        heuristic_func: A function that takes a node (state) as input and returns
                        an estimated cost (heuristic) to reach the goal from that node.
                        The heuristic function must be admissible (never overestimates
                        the actual cost).

    Returns:
        A tuple containing:
        - The path (list of nodes) from start to goal, or None if no path exists.
        - The total cost of the path, or None if no path exists.
    """

    open_set = []  # Priority queue (heap) of nodes to explore
    closed_set = set()  # Set of visited nodes
    start_node = Node(start, cost=0, heuristic=heuristic_func(start))
    heapq.heappush(open_set, start_node)  # Add the start node to the open set

    while open_set:
        current_node = heapq.heappop(open_set)  # Get the node with the lowest f-score

        if current_node.state == goal:
            # Reconstruct the path
            path = []
            node = current_node
            while node:
                path.append(node.state)
                node = node.parent
            return path[::-1], current_node.cost  # Reverse the path to get the correct order

        closed_set.add(current_node.state)  # Mark the current node as visited

        for neighbor, cost in graph.get(current_node.state, {}).items():
            if neighbor in closed_set:
                continue  # Skip already visited neighbors

            tentative_cost = current_node.cost + cost
            neighbor_node = Node(neighbor, parent=current_node, cost=tentative_cost, heuristic=heuristic_func(neighbor))

            # Check if the neighbor is already in the open set with a lower cost
            in_open_set = False
            for node in open_set:
                if node.state == neighbor and node.cost <= tentative_cost:
                    in_open_set = True
                    break
            if in_open_set:
                continue

            heapq.heappush(open_set, neighbor_node)  # Add the neighbor to the open set

    return None, None  # No path found


# Example Usage:
if __name__ == '__main__':
    # Graph representation
    graph = {
        'A': {'B': 10, 'C': 15},
        'B': {'A': 10, 'D': 12, 'E': 15},
        'C': {'A': 15, 'F': 12},
        'D': {'B': 12, 'G': 10},
        'E': {'B': 15, 'H': 10},
        'F': {'C': 12, 'I': 10},
        'G': {'D': 10},
        'H': {'E': 10},
        'I': {'F': 10}
    }

    start_node = 'A'
    goal_node = 'I'

    # Heuristic function (example: straight-line distance, must be admissible)
    # In a real-world scenario, you would replace this with a more appropriate heuristic
    def heuristic(node):
        # This heuristic is admissible because it always returns 0 (underestimates the cost)
        # A better heuristic would give a more accurate estimate.
        heuristic_values = { # Example heuristic values (must be admissible)
            'A': 20, 'B': 15, 'C': 10, 'D': 12, 'E': 13, 'F': 0,
            'G': 0, 'H': 0, 'I': 0
        }
        return heuristic_values.get(node, 0)  #Return heuristic or 0 if not found

    path, cost = a_star(graph, start_node, goal_node, heuristic)

    if path:
        print("Path:", path)
        print("Cost:", cost)
    else:
        print("No path found.")
