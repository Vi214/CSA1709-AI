class MapColoring:
    def __init__(self, states, neighbors, colors):
        self.states = states  # List of states or regions
        self.neighbors = neighbors  # Dictionary mapping states to their neighboring states
        self.colors = colors  # List of available colors
        self.state_colors = {}  # Dictionary to store assigned colors

    def is_valid(self, state, color):
        """Check if the current color assignment is valid."""
        for neighbor in self.neighbors.get(state, []):
            if neighbor in self.state_colors and self.state_colors[neighbor] == color:
                return False
        return True

    def solve(self, index=0):
        """Solve the map coloring problem using backtracking."""
        if index == len(self.states):
            return True  # All states have been colored successfully
        
        state = self.states[index]
        for color in self.colors:
            print(f"Trying to color {state} with {color}")
            if self.is_valid(state, color):
                self.state_colors[state] = color  # Assign color
                print(f"{state} is colored with {color}")
                if self.solve(index + 1):  # Recur for next state
                    return True
                print(f"Backtracking on {state}, removing color {color}")
                del self.state_colors[state]  # Backtrack
        
        return False  # No valid color assignment found

    def get_coloring(self):
        """Returns the assigned colors after solving."""
        if self.solve():
            return self.state_colors
        else:
            return "No valid coloring possible"

# Example usage
states = ["A", "B", "C", "D"]
neighbors = {
    "A": ["B", "C"],
    "B": ["A", "C", "D"],
    "C": ["A", "B", "D"],
    "D": ["B", "C"]
}
colors = ["Red", "Green", "Blue"]

map_coloring = MapColoring(states, neighbors, colors)
solution = map_coloring.get_coloring()
print("Final Coloring:", solution)
