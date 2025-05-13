import math

def alpha_beta_pruning(depth, node_index, is_max, values, alpha, beta, trace):
    if depth == 3:
        trace.append((depth, node_index, values[node_index]))
        return values[node_index]
    
    if is_max:
        best = -math.inf
        for i in range(2):  # Assuming each node has 2 children
            val = alpha_beta_pruning(depth + 1, node_index * 2 + i, False, values, alpha, beta, trace)
            best = max(best, val)
            alpha = max(alpha, best)
            if beta <= alpha:
                trace.append((depth, node_index, best, "Pruned"))
                break  # Beta Cut-off
        trace.append((depth, node_index, best))
        return best
    else:
        best = math.inf
        for i in range(2):  # Assuming each node has 2 children
            val = alpha_beta_pruning(depth + 1, node_index * 2 + i, True, values, alpha, beta, trace)
            best = min(best, val)
            beta = min(beta, best)
            if beta <= alpha:
                trace.append((depth, node_index, best, "Pruned"))
                break  # Alpha Cut-off
        trace.append((depth, node_index, best))
        return best

# Example usage
values = [3, 5, 6, 9, 1, 2, 0, -1]  # Terminal values at leaf nodes
trace = []
optimal_value = alpha_beta_pruning(0, 0, True, values, -math.inf, math.inf, trace)

# Print detailed output
print("Step-by-step execution:")
for step in trace:
    if len(step) == 4:
        print(f"Depth {step[0]}, Node {step[1]}, Value {step[2]}, {step[3]}")
    else:
        print(f"Depth {step[0]}, Node {step[1]}, Value {step[2]}")

print("\nOptimal Value:", optimal_value)
