import random

class VacuumCleaner:
    def __init__(self):  # Corrected the constructor name
        self.rooms = {"A": random.randint(0, 1), "B": random.randint(0, 1)}
        self.position = random.choice(["A", "B"])  
        self.steps = 0 

    def display_status(self):
        """Displays the current status of the environment."""
        print(f"Room A: {'Dirty' if self.rooms['A'] else 'Clean'}, Room B: {'Dirty' if self.rooms['B'] else 'Clean'}")
        print(f"Vacuum is in Room {self.position}\n")

    def clean_room(self):
        """Cleans the room if it is dirty."""
        if self.rooms[self.position] == 1:
            print(f"Cleaning Room {self.position}...")
            self.rooms[self.position] = 0
        else:
            print(f"Room {self.position} is already clean.")

    def move(self):
        """Moves the vacuum cleaner to the other room."""
        self.position = "A" if self.position == "B" else "B"
        print(f"Moving to Room {self.position}...")

    def run(self):
        """Runs the vacuum cleaner until all rooms are clean."""
        print("Initial State:")
        self.display_status()

        while any(self.rooms.values()):
            self.clean_room()
            if any(self.rooms.values()):  
                self.move()
            self.steps += 1  
            self.display_status()

        print(f"All rooms are clean! Task completed in {self.steps} steps.")

vacuum = VacuumCleaner()
vacuum.run()
