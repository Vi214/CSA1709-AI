from collections import deque

class BFS:
    def __init__(self, board, empty_tile_pos, moves=0, previous=None):
        self.board = board
        self.empty_tile_pos = empty_tile_pos
        self.moves = moves
        self.previous = previous
        self.size = int(len(board)**0.5)

    def is_goal(self):
        return self.board == list(range(1, self.size * self.size)) + [0]

    def get_possible_moves(self):
        x, y = self.empty_tile_pos
        moves = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right

        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < self.size and 0 <= new_y < self.size:
                moves.append((new_x, new_y))

        return moves

    def move(self, new_empty_tile_pos):
        new_board = self.board[:]
        x, y = self.empty_tile_pos
        new_x, new_y = new_empty_tile_pos
        new_board[x * self.size + y], new_board[new_x * self.size + new_y] = new_board[new_x * self.size + new_y], new_board[x * self.size + y]
        return BFS(new_board, new_empty_tile_pos, self.moves + 1, self)

def breadth_first_search(initial_state):
    """Solves the 8-puzzle using Breadth-First Search (BFS)."""

    queue = deque([initial_state])  # Use a deque for efficient FIFO
    explored = set()  # Keep track of visited states

    while queue:
        current_state = queue.popleft()  # FIFO:  Get the oldest state

        if current_state.is_goal():
            return current_state

        explored.add(tuple(current_state.board))  # Mark state as visited

        for move in current_state.get_possible_moves():
            new_state = current_state.move(move)
            if tuple(new_state.board) not in explored:
                queue.append(new_state)  # Enqueue the new state

    return None  # No solution found

def print_solution(solution):
    """Prints the solution path."""
    path = []
    while solution:
        path.append(solution.board)
        solution = solution.previous

    for state in reversed(path):
        # Print the board
        for i in range(0, len(state), int(len(state)**0.5)):
            print(state[i:i + int(len(state)**0.5)])
        print("---")  # Separator between states


if __name__ == "__main__":
    initial_board = [1, 2, 3, 4, 5, 6, 0, 7, 8]  # Example initial state
    empty_tile_pos = (2, 0)  # Position of the empty tile (0)
    initial_state = BFS(initial_board, empty_tile_pos)

    solution = breadth_first_search(initial_state)

    if solution:
        print("Solution found:")
        print_solution(solution)
    else:
        print("No solution exists.")
