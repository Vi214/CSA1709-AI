import math

def minimax(depth, node_index, is_max, values):
    if depth == 3:  
        return values[node_index]
    
    if is_max:
        max_eval = -math.inf
        for i in range(2):
            eval = minimax(depth + 1, node_index * 2 + i, False, values)
            max_eval = max(max_eval, eval)
        print(f"Max Node: Depth {depth}, Node {node_index}, Value {max_eval}")
        return max_eval
    else:
        min_eval = math.inf
        for i in range(2):
            eval = minimax(depth + 1, node_index * 2 + i, True, values)
            min_eval = min(min_eval, eval)
        print(f"Min Node: Depth {depth}, Node {node_index}, Value {min_eval}")
        return min_eval
values = [3, 5, 6, 9, 1, 2, 0, -1]
result = minimax(0, 0, True, values)
print("Final optimal value:", result)
