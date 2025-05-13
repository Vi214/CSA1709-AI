from collections import deque

initial_state = (3, 3, 1)
goal_state = (0, 0, 0)
moves = [(1, 0), (2, 0), (0, 1), (0, 2), (1, 1)]  

def is_valid(state):
    """Check if the state is valid (no missionaries eaten)."""
    M, C, _ = state
    M_opposite, C_opposite = 3 - M, 3 - C

    if (M < C and M > 0) or (M_opposite < C_opposite and M_opposite > 0):
        return False
    return True

def get_next_states(state):
    """Generate possible next states from the given state."""
    M, C, B = state
    next_states = []

    direction = -1 if B == 1 else 1  

    for m, c in moves:
        new_state = (M + direction * m, C + direction * c, B + direction)
        if 0 <= new_state[0] <= 3 and 0 <= new_state[1] <= 3 and is_valid(new_state):
            next_states.append(new_state)

    return next_states

def solve():
    """Solve the Missionaries and Cannibals problem using BFS."""
    queue = deque([(initial_state, [])])  
    visited = set()

    while queue:
        state, path = queue.popleft()

        if state in visited:
            continue
        visited.add(state)
        if state == goal_state:
            return path + [state]
        for next_state in get_next_states(state):
            queue.append((next_state, path + [state]))

    return None  
solution = solve()
if solution:
    for step in solution:
        print(f"Missionaries: {step[0]}, Cannibals: {step[1]}, Boat: {'Left' if step[2] == 1 else 'Right'}")
else:
    print("No solution found.")
