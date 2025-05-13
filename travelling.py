import itertools

def traveling_salesman(graph, start_node):
    """
    Solves the Traveling Salesman Problem (TSP) using brute force.

    Args:
        graph: A dictionary representing the graph where keys are nodes
               and values are dictionaries of neighboring nodes with their
               corresponding costs (distances).  For example:
               graph = {
                   'A': {'B': 10, 'C': 15, 'D': 20},
                   'B': {'A': 10, 'C': 35, 'D': 25},
                   'C': {'A': 15, 'B': 35, 'D': 30},
                   'D': {'A': 20, 'B': 25, 'C': 30}
               }
        start_node: The node to start and end the tour.

    Returns:
        A tuple containing:
        - The best tour (list of nodes in order).
        - The cost of the best tour.
    """

    nodes = list(graph.keys())
    nodes.remove(start_node)  # Remove start node to form permutations of other nodes

    best_tour = None
    best_cost = float('inf')  # Initialize with infinity

    for permutation in itertools.permutations(nodes):
        # Construct the tour by adding the start node at the beginning and end
        tour = [start_node] + list(permutation) + [start_node]
        cost = 0

        # Calculate the cost of the tour
        for i in range(len(tour) - 1):
            current_node = tour[i]
            next_node = tour[i+1]
            if next_node not in graph[current_node]:
                # Handle missing edges (very important!)
                cost = float('inf')
                break  # Invalidate this tour
            cost += graph[current_node][next_node]

        # Update the best tour if the current tour is better
        if cost < best_cost:
            best_tour = tour
            best_cost = cost

    return best_tour, best_cost


# Example Usage:
if __name__ == '__main__':
    graph = {
        'A': {'B': 10, 'C': 15, 'D': 20},
        'B': {'A': 10, 'C': 35, 'D': 25},
        'C': {'A': 15, 'B': 35, 'D': 30},
        'D': {'A': 20, 'B': 25, 'C': 30}
    }
    start_node = 'A'

    best_tour, best_cost = traveling_salesman(graph, start_node)

    if best_tour:
        print("Best Tour:", best_tour)
        print("Cost:", best_cost)
    else:
        print("No tour found.")
