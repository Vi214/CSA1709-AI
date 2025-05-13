def is_safe(board, row, col):
    """Checks if it's safe to place a queen at board[row][col]"""

    # Check the same column
    for i in range(row):
        if board[i][col] == 1:
            return False

    # Check upper left diagonal
    for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
        if board[i][j] == 1:
            return False

    # Check upper right diagonal
    for i, j in zip(range(row, -1, -1), range(col, len(board), 1)):
        if board[i][j] == 1:
            return False

    return True

def solve_n_queens_util(board, row):
    """Recursive utility function to solve N-Queens problem"""

    # Base case: If all queens are placed, return True
    if row >= len(board):
        return True

    # Consider this row and try placing this queen in all columns one by one
    for col in range(len(board)):

        if is_safe(board, row, col):
            # Place this queen in board[row][col]
            board[row][col] = 1

            # Recur to place rest of the queens
            if solve_n_queens_util(board, row + 1):
                return True

            # If placing queen in board[row][col] doesn't lead to a solution,
            # then backtrack: Remove the queen from board[row][col]
            board[row][col] = 0

    # If the queen cannot be placed in any column in this row, return False
    return False

def solve_n_queens(n):
    """Solves the N-Queens problem"""
    board = [[0 for _ in range(n)] for _ in range(n)]  # Create an empty board

    if not solve_n_queens_util(board, 0):
        print("Solution does not exist")
        return False

    print_solution(board)
    return True

def print_solution(board):
    """Prints the board with the queen positions"""
    for i in range(len(board)):
        for j in range(len(board)):
            print(board[i][j], end=" ")
        print()

# Example usage:
if __name__ == "__main__":
    n = 8  # Change this to solve for a different number of queens
    solve_n_queens(n)
