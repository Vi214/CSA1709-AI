from copy import deepcopy
from collections import deque
import sys
import time

class State(object):
    def __init__(self, missionaries, cannibals , boats):
        self.missionaries = missionaries
        self.cannibals = cannibals
        self.boats = boats

    def successors(self):
        if self.boats == 1:
            sgn = -1
            direction = "from the original shore to the new shore"
        else:
            sgn = 1
            direction = "back from the new shore to the original shore"
        for m in range(3):
            for c in range(3):
                newState = State(self.missionaries+sgn*m, self.cannibals+sgn*c, self.boats+sgn*1);
                if m+c >= 1 and m+c <= 2 and newState.isValid():
                    action = "take %d missionaries and %d cannibals %s. %r" % ( m, c, direction, newState)
                    yield action, newState

    def isValid(self):
        if self.missionaries < 0 or self.cannibals < 0 or self.missionaries > 3 or self.cannibals > 3 or (self.boats != 0 and self.boats != 1):
            return False

        if self.cannibals > self.missionaries and self.missionaries > 0:
            return False
        if self.cannibals < self.missionaries and self.missionaries < 3:
            return False
        return True

    def is_goal_state(self):
        return self.cannibals == 0 and self.missionaries == 0 and self.boats == 0

    def __repr__(self):
        return "< State (%d, %d, %d) >" % (self.missionaries, self.cannibals, self.boats)

    def __eq__(self, other):
        return self.missionaries == other.missionaries and self.cannibals == other.cannibals and self.boats == other.boats

    def __hash__(self):
        return hash((self.missionaries, self.cannibals, self.boats))

class Node(object):
    def __init__(self, parent_node, state, action, depth):
        self.parent_node = parent_node
        self.state = state
        self.action = action
        self.depth = depth

    def expand(self):
        for (action, succ_state) in self.state.successors():
            succ_node = Node( parent_node=self, state=succ_state, action=action, depth=self.depth + 1)
            yield succ_node

    def extract_solution(self):
        solution = []
        node = self
        while node.parent_node is not None:
            solution.append(node.action)
            node = node.parent_node
        solution.reverse()
        return solution

def breadth_first_search():
    initial_state = State(3,3,1)
    if initial_state.is_goal_state():
        return initial_state

    explored = set()
    frontier = deque([Node(None, initial_state, None, 0)])
    node_count = 0 # Counter for number of nodes explored

    while frontier:
        node = frontier.popleft()
        node_count += 1

        print(f"Exploring Node {node_count}: Depth = {node.depth}, State = {node.state}")  # Detailed output

        if node.state in explored:
            print(f"  State {node.state} already explored. Skipping.")
            continue

        explored.add(node.state)

        if node.state.is_goal_state():
            print("  Goal state found!")
            return node.extract_solution()

        print("  Expanding node...")
        for succ_node in node.expand():
            print(f"    Adding successor to frontier: State = {succ_node.state}, Action = {succ_node.action}") #Detailed output
            frontier.append(succ_node)

        print(f"  Frontier size: {len(frontier)}") #Detailed output

    return None  # No solution found

if __name__ == '__main__':
    start_time = time.time()
    solution = breadth_first_search()
    end_time = time.time()

    if solution:
        print ("\nSolution:")
        for step in solution:
            print(step)
    else:
        print("No solution found.")

    print ("\nTime to find a solution: %f seconds" % (end_time-start_time))
