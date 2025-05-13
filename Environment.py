import random

class Environment:
    """Represents the environment for the vacuum cleaner."""

    def __init__(self, width, height):
        """
        Initializes the environment.

        Args:
            width (int): The width of the environment (number of columns).
            height (int): The height of the environment (number of rows).
        """
        self.width = width
        self.height = height
        self.grid = [[random.choice([True, False]) for _ in range(width)] for _ in range(height)]  # True = dirty, False = clean
        self.agent_x = random.randint(0, width - 1)
        self.agent_y = random.randint(0, height - 1)
        self.performance = 0  # Agent's performance score

    def display(self):
        """Displays the current state of the environment."""
        for y in range(self.height):
            row = ""
            for x in range(self.width):
                if self.agent_x == x and self.agent_y == y:
                    row += "A "  # Agent
                elif self.grid[y][x]:
                    row += "* "  # Dirty
                else:
                    row += "_ "  # Clean
            print(row)
        print(f"Agent Location: ({self.agent_x}, {self.agent_y})")
        print(f"Performance: {self.performance}")
        print("-" * (self.width * 2))  # Separator

    def is_dirty(self, x, y):
        """Checks if a specific location is dirty."""
        return self.grid[y][x]

    def clean(self, x, y):
        """Cleans a specific location."""
        if self.is_dirty(x, y):
            self.grid[y][x] = False
            self.performance += 10  # Reward for cleaning
            print(f"Cleaning cell ({x}, {y})")
        else:
            self.performance -= 1  # Penalty for cleaning an already clean cell
            print(f"Cell ({x}, {y}) already clean")

    def move_up(self):
        """Moves the agent up, if possible."""
        if self.agent_y > 0:
            self.agent_y -= 1
            self.performance -= 1  # Penalty for moving
            print("Moving Up")
        else:
            self.performance -= 5 # Penalty for bumping into wall
            print("Cannot move up - Bumping into wall!")

    def move_down(self):
        """Moves the agent down, if possible."""
        if self.agent_y < self.height - 1:
            self.agent_y += 1
            self.performance -= 1 # Penalty for moving
            print("Moving Down")
        else:
            self.performance -= 5 # Penalty for bumping into wall
            print("Cannot move down - Bumping into wall!")


    def move_left(self):
        """Moves the agent left, if possible."""
        if self.agent_x > 0:
            self.agent_x -= 1
            self.performance -= 1 # Penalty for moving
            print("Moving Left")
        else:
            self.performance -= 5 # Penalty for bumping into wall
            print("Cannot move left - Bumping into wall!")

    def move_right(self):
        """Moves the agent right, if possible."""
        if self.agent_x < self.width - 1:
            self.agent_x += 1
            self.performance -= 1 # Penalty for moving
            print("Moving Right")
        else:
            self.performance -= 5 # Penalty for bumping into wall
            print("Cannot move right - Bumping into wall!")


class Agent:
    """Represents the vacuum cleaner agent."""

    def __init__(self, environment):
        """Initializes the agent with a reference to the environment."""
        self.environment = environment

    def simple_reflex_agent(self):
        """A simple reflex agent that cleans if the current location is dirty."""
        x, y = self.environment.agent_x, self.environment.agent_y

        if self.environment.is_dirty(x, y):
            self.environment.clean(x, y)
        else:
            # Move randomly if the current cell is clean
            move = random.choice(["up", "down", "left", "right"])
            if move == "up":
                self.environment.move_up()
            elif move == "down":
                self.environment.move_down()
            elif move == "left":
                self.environment.move_left()
            else:
                self.environment.move_right()

    def deliberate_agent(self):
        """A deliberate agent which scans the environment and then cleans."""
        #This is a very basic implementation and can be improved
        dirty_cells = []

        #Scan the environment
        for y in range(self.environment.height):
            for x in range(self.environment.width):
                if self.environment.is_dirty(x, y):
                    dirty_cells.append((x,y))

        if not dirty_cells:
            print("No dirty cells left. Halting.")
            return

        #Move to the nearest dirty cell (very basic)
        nearest_dirty = dirty_cells[0]
        dx = nearest_dirty[0] - self.environment.agent_x
        dy = nearest_dirty[1] - self.environment.agent_y

        if dx > 0:
            self.environment.move_right()
        elif dx < 0:
            self.environment.move_left()
        elif dy > 0:
            self.environment.move_down()
        elif dy < 0:
            self.environment.move_up()
        else: #At the dirty cell
            self.environment.clean(self.environment.agent_x, self.environment.agent_y)



# Steps to solve the Vacuum Cleaner Problem

# 1. Define the Environment:
#    - Create a grid representing the environment.
#    - Randomly assign dirt (True) or cleanliness (False) to each cell.
#    - Keep track of the agent's location (x, y coordinates).
#    - Maintain a performance score (initially 0).

# 2. Define the Agent:
#    - The agent has a reference to the environment.
#    - The agent has a perception of the environment (e.g., is the current cell dirty?).
#    - The agent can perform actions (e.g., move, suck dirt).
#    - Implement a decision-making process (the agent's "brain").  I have implemented two: simple reflex and deliberate.

# 3. Implement Agent Actions:
#    - `clean(x, y)`: If the cell is dirty, clean it (set grid[x][y] to False) and update the performance score.
#    - `move_up()`, `move_down()`, `move_left()`, `move_right()`:  Move the agent to the adjacent cell (if possible) and update the performance score.  Penalize bumping into walls.

# 4. Implement Agent Logic:
#   - Simple Reflex Agent: Cleans if the current cell is dirty, otherwise moves randomly.
#   - Deliberate Agent: Scans the environment to find dirty cells, then moves toward the nearest one and cleans it.

# 5. Run the Simulation:
#    - Create an Environment.
#    - Create an Agent, passing the Environment to it.
#    - Run a loop for a certain number of steps.
#    - In each step:
#      - Display the environment.
#      - Call the agent's `simple_reflex_agent()` or `deliberate_agent`  method to make a decision and act.

# Example Usage

width = 5
height = 4
env = Environment(width, height)
agent = Agent(env)

# Run the simulation for a certain number of steps using simple_reflex_agent
print("Running Simple Reflex Agent:")
for i in range(10):
    env.display()
    agent.simple_reflex_agent()

#Reset environment and agent
env = Environment(width,height)
agent = Agent(env)

print("\nRunning Deliberate Agent:")
for i in range(10):
    env.display()
    agent.deliberate_agent()
