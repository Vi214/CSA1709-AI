from collections import deque

def water_jug_problem(capacity_a, capacity_b, target):
    """
    Solves the water jug problem using Breadth-First Search (BFS).

    Args:
        capacity_a: Capacity of jug A.
        capacity_b: Capacity of jug B.
        target: Target amount of water to be measured in either jug.

    Returns:
        A list of tuples representing the sequence of actions to reach the target,
        or None if no solution is found.  Each tuple is of the form:
        (action, (amount_a, amount_b)), where action is a string describing
        the action taken, and (amount_a, amount_b) is the state of the jugs
        after the action.
    """

    initial_state = (0, 0)  # (amount_a, amount_b)
    queue = deque([(initial_state, [])])  # Queue of (state, path)
    explored = set()  # Set of visited states

    while queue:
        (amount_a, amount_b), path = queue.popleft()

        if amount_a == target or amount_b == target:
            return path + [("Goal Reached", (amount_a, amount_b))]

        if (amount_a, amount_b) in explored:
            continue

        explored.add((amount_a, amount_b))

        # Possible actions:
        # 1. Fill jug A
        queue.append(((capacity_a, amount_b), path + [("Fill A", (capacity_a, amount_b))]))

        # 2. Fill jug B
        queue.append(((amount_a, capacity_b), path + [("Fill B", (amount_a, capacity_b))]))

        # 3. Empty jug A
        queue.append(((0, amount_b), path + [("Empty A", (0, amount_b))]))

        # 4. Empty jug B
        queue.append(((amount_a, 0), path + [("Empty B", (amount_a, 0))]))

        # 5. Pour A into B
        pour_amount = min(amount_a, capacity_b - amount_b)
        new_amount_a = amount_a - pour_amount
        new_amount_b = amount_b + pour_amount
        queue.append(((new_amount_a, new_amount_b), path + [("Pour A to B", (new_amount_a, new_amount_b))]))

        # 6. Pour B into A
        pour_amount = min(amount_b, capacity_a - amount_a)
        new_amount_a = amount_a + pour_amount
        new_amount_b = amount_b - pour_amount
        queue.append(((new_amount_a, new_amount_b), path + [("Pour B to A", (new_amount_a, new_amount_b))]))

    return None  # No solution found


# Example Usage:
if __name__ == "__main__":
    capacity_a = 4  # Capacity of jug A
    capacity_b = 3  # Capacity of jug B
    target = 2      # Target amount

    solution = water_jug_problem(capacity_a, capacity_b, target)

    if solution:
        print("Solution found:")
        for action, (amount_a, amount_b) in solution:
            print(f"{action}: Jug A = {amount_a}, Jug B = {amount_b}")
    else:
        print("No solution exists.")
