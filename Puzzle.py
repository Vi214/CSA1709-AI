import heapq

class Puzzle:
    def __init__(self, state, parent=None, move=None, cost=0):
        self.state = state
        self.parent = parent
        self.move = move
        self.cost = cost
        self.size = len(state)
        self.blank_position = self.find_blank()

    def __lt__(self, other):
        return (self.cost + self.heuristic()) < (other.cost + other.heuristic())

    def __eq__(self, other):
        return self.state == other.state

    def __hash__(self):
        return hash(tuple(map(tuple, self.state)))

    def find_blank(self):
        for r in range(self.size):
            for c in range(self.size):
                if self.state[r][c] == 0:
                    return (r, c)

    def get_neighbors(self):
        row, col = self.blank_position
        neighbors = []
        possible_moves = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}

        for move, (dr, dc) in possible_moves.items():
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < self.size and 0 <= new_col < self.size:
                new_state = [list(row) for row in self.state]
                new_state[row][col], new_state[new_row][new_col] = new_state[new_row][new_col], new_state[row][col]
                new_state = tuple(tuple(row) for row in new_state)
                neighbors.append(Puzzle(new_state, parent=self, move=move, cost=self.cost + 1))

        return neighbors

    def heuristic(self):
        distance = 0
        for r in range(self.size):
            for c in range(self.size):
                value = self.state[r][c]
                if value != 0:
                    target_row = (value - 1) // self.size
                    target_col = (value - 1) % self.size
                    distance += abs(r - target_row) + abs(c - target_col)
        return distance

    def is_goal(self, goal_state):
        return self.state == goal_state

def solve_8_puzzle(initial_state, goal_state):
    initial_node = Puzzle(initial_state)
    frontier = [initial_node]
    heapq.heapify(frontier)
    explored = set()

    while frontier:
        current_node = heapq.heappop(frontier)

        if current_node.is_goal(goal_state):
            return reconstruct_path(current_node)

        explored.add(current_node)

        for neighbor in current_node.get_neighbors():
            if neighbor not in explored:
                heapq.heappush(frontier, neighbor)

    return None

def reconstruct_path(node):
    path = []
    while node.parent:
        path.append((node.move, node.state))
        node = node.parent
    path.reverse()
    return path

if __name__ == '__main__':
    initial_state = (
        (1, 2, 3),
        (8, 0, 4),
        (7, 6, 5)
    )

    goal_state = (
        (1, 2, 3),
        (8, 4, 0),
        (7, 6, 5)
    )

    solution = solve_8_puzzle(initial_state, goal_state)

    if solution:
        print("Solution found:")
        for move, state in solution:
            print(f"Move: {move}")
            for row in state:
                print(row)
            print("---")
    else:
        print("No solution found.")
