import random

def print_board(board):
    for row in board:
        print(" | ".join(row))
        print("-" * 5)

def check_winner(board, player):
    """Check if the player has won the game."""
    for row in board:
        if all(s == player for s in row):
            print(f"Player {player} wins by completing a row: {row}")
            return True
    
    for col in range(3):
        if all(board[row][col] == player for row in range(3)):
            print(f"Player {player} wins by completing column {col}")
            return True
    
    if all(board[i][i] == player for i in range(3)):
        print(f"Player {player} wins by completing the main diagonal")
        return True
    
    if all(board[i][2 - i] == player for i in range(3)):
        print(f"Player {player} wins by completing the anti-diagonal")
        return True
    
    return False

def is_full(board):
    """Check if the board is full."""
    full = all(all(cell != " " for cell in row) for row in board)
    if full:
        print("Board is full. The game is a tie!")
    return full

def tic_tac_toe():
    board = [[" " for _ in range(3)] for _ in range(3)]
    players = ["X", "O"]
    current_player = random.choice(players)
    
    while True:
        print_board(board)
        print(f"Current board state: {board}")
        row, col = map(int, input(f"Player {current_player}, enter row and column (0-2): ").split())
        
        if board[row][col] != " ":
            print("Cell already taken. Try again.")
            continue
        
        print(f"Player {current_player} places at ({row}, {col})")
        board[row][col] = current_player
        
        if check_winner(board, current_player):
            print_board(board)
            print(f"Game Over! Player {current_player} wins!")
            break
        
        if is_full(board):
            print_board(board)
            print("It's a tie!")
            break
        
        current_player = "X" if current_player == "O" else "O"
        print(f"Next turn: Player {current_player}")

tic_tac_toe()
