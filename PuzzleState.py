def solve(state, goal, path=[]):
    if state == goal: return path + [state]
    (m, c, b) = state
    moves = [(1, 0), (0, 1), (1, 1), (2, 0), (0, 2)]
    for move in moves:
        new_state = (m - move[0]*b, c - move[1]*b, 1 - b)
        if 0 <= new_state[0] <= 3 and 0 <= new_state[1] <= 3 and (new_state[0] == 0 or new_state[0] >= new_state[1]):
            if new_state not in path:
                result = solve(new_state, goal, path + [state])
                if result: return result
print(solve((3, 3, 1), (0, 0, 0)))
