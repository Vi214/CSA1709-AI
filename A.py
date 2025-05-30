import heapq

class Node:
    def __init__(self, x, y, parent=None):
        self.x = x
        self.y = y
        self.parent = parent
        self.g = 0  # Cost from start to current node
        self.h = 0  # Heuristic estimate to goal
        self.f = 0  # Total cost (g + h)

    def __lt__(self, other):
        return self.f < other.f

def astar(grid, start, end):
    """A* pathfinding algorithm implementation
    
    Parameters:
    grid (2D list): 0 = walkable, 1 = obstacle
    start (tuple): (x, y) starting position
    end (tuple): (x, y) target position
    
    Returns:
    list: Path as list of (x, y) tuples or None if no path
    """
    # Define movement directions (4-way movement)
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    # Initialize open and closed lists
    open_list = []
    closed_set = set()
    
    # Create start and end nodes
    start_node = Node(*start)
    end_node = Node(*end)
    
    heapq.heappush(open_list, start_node)
    
    while open_list:
        current_node = heapq.heappop(open_list)
        closed_set.add((current_node.x, current_node.y))
        
        # Goal check
        if (current_node.x, current_node.y) == (end_node.x, end_node.y):
            path = []
            while current_node:
                path.append((current_node.x, current_node.y))
                current_node = current_node.parent
            return path[::-1]  # Return reversed path
        
        # Generate children
        children = []
        for dx, dy in directions:
            x = current_node.x + dx
            y = current_node.y + dy
            
            # Check grid boundaries
            if not (0 <= x < len(grid) and 0 <= y < len(grid[0])):
                continue
                
            # Check walkability
            if grid[x][y] != 0:
                continue
                
            children.append(Node(x, y, current_node))
        
        # Process children
        for child in children:
            if (child.x, child.y) in closed_set:
                continue
                
            # Calculate costs
            child.g = current_node.g + 10  # Base cost for horizontal/vertical move
            child.h = 10 * (abs(child.x - end_node.x) + abs(child.y - end_node.y))  # Manhattan heuristic
            child.f = child.g + child.h
            
            # Check if child is already in open list with lower g
            in_open = False
            for open_node in open_list:
                if (open_node.x, open_node.y) == (child.x, child.y):
                    in_open = True
                    if child.g > open_node.g:
                        continue
                    else:
                        open_list.remove(open_node)
                        heapq.heapify(open_list)
                        break
            
            heapq.heappush(open_list, child)
    
    return None  # No path found

# Example usage
if __name__ == "__main__":
    grid = [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0]
    ]
    start = (0, 0)
    end = (4, 4)
    
    path = astar(grid, start, end)
    print("Path found:", path)

output:
Path found: [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4)]


