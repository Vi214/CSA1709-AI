import heapq

class PuzzleNode:
    def __init__(self, state, parent=None, move=None, cost=0, heuristic=0):
        self.state = state  # Current state of the puzzle (2D list)
        self.parent = parent  # Parent node
        self.move = move  # Move taken to reach this state
        self.cost = cost  # Cost to reach this state (g)
        self.heuristic = heuristic  # Heuristic cost (h)
        self.total_cost = cost + heuristic  # Total cost (f = g + h)
    
    def __lt__(self, other):
        return self.total_cost < other.total_cost

def manhattan_distance(state, goal):
    """Calculate the Manhattan distance heuristic."""
    distance = 0
    for i in range(3):
        for j in range(3):
            if state[i][j] != 0:  # Ignore empty tile
                x, y = divmod(goal.index(state[i][j]), 3)
                distance += abs(x - i) + abs(y - j)
    return distance

def get_neighbors(state):
    """Generate possible moves for the blank space."""
    moves = []
    row, col = [(i, j) for i in range(3) for j in range(3) if state[i][j] == 0][0]
    directions = [(-1, 0, 'Up'), (1, 0, 'Down'), (0, -1, 'Left'), (0, 1, 'Right')]
    
    for dr, dc, move in directions:
        new_row, new_col = row + dr, col + dc
        if 0 <= new_row < 3 and 0 <= new_col < 3:
            new_state = [row[:] for row in state]  # Deep copy of the state
            new_state[row][col], new_state[new_row][new_col] = new_state[new_row][new_col], new_state[row][col]
            moves.append((new_state, move))
    
    return moves

def print_puzzle(state):
    """Print the puzzle state in a readable format."""
    for row in state:
        print(" ".join(str(x) if x != 0 else "_" for x in row))
    print("\n")

def a_star_search(start, goal):
    """A* search algorithm for solving the 8-puzzle problem."""
    goal_flat = sum(goal, [])  # Flatten goal state for easier index lookup
    open_list = []
    heapq.heappush(open_list, PuzzleNode(start, cost=0, heuristic=manhattan_distance(start, goal_flat)))
    visited = set()
    nosteps = 0
    
    while open_list:
        current_node = heapq.heappop(open_list)
        nosteps += 1
        print(f"Step {nosteps}:")
        print_puzzle(current_node.state)
        print(f"Total Cost (f = g + h): {current_node.total_cost}\n")
        
        if current_node.state == goal:
            path = []
            while current_node.parent:
                path.append(current_node.move)
                current_node = current_node.parent
            print(f"Number of steps: {nosteps}")
            return path[::-1]
        
        visited.add(tuple(map(tuple, current_node.state)))
        
        for new_state, move in get_neighbors(current_node.state):
            if tuple(map(tuple, new_state)) not in visited:
                new_node = PuzzleNode(new_state, current_node, move, current_node.cost + 1, manhattan_distance(new_state, goal_flat))
                heapq.heappush(open_list, new_node)
    
    return None  # No solution found

# Example usage
start_state = [[1, 2, 3], [4, 0, 5], [6, 7, 8]]  # Initial puzzle state
goal_state = [[1, 2, 3], [7, 8, 0], [4, 5, 6]]  # Goal state

solution = a_star_search(start_state, goal_state)
print("Solution Steps:", solution)
