import itertools

def solve_cryptarithmetic(equation):
    """
    Solves a cryptarithmetic puzzle using brute-force search.

    Args:
        equation: A string representing the cryptarithmetic equation in the form
                  "SEND + MORE = MONEY".  The equation must use uppercase letters.

    Returns:
        A dictionary mapping letters to digits if a solution is found,
        otherwise None.
    """

    # 1. Extract the words and letters from the equation
    parts = equation.replace(" ", "").split("=")
    left_side = parts[0].split("+")
    right_side = parts[1]
    words = left_side + [right_side]
    unique_letters = "".join(sorted(set("".join(words))))
    n = len(unique_letters)

    # 2. Generate all possible digit assignments
    for digits in itertools.permutations(range(10), n):
        # 3. Create a mapping of letters to digits
        letter_to_digit = dict(zip(unique_letters, digits))

        # 4. Check if the assignment is valid
        if any(letter_to_digit[word[0]] == 0 for word in words):
            # Leading zeros are not allowed
            continue

        # 5. Evaluate the equation using the digit assignment
        values = []
        for word in words:
            value = 0
            for letter in word:
                value = value * 10 + letter_to_digit[letter]
            values.append(value)

        # 6. Check if the equation holds true
        if sum(values[:-1]) == values[-1]:
            return letter_to_digit

    return None  # No solution found


# Example Usage:
if __name__ == "__main__":
    equation = "SEND + MORE = MONEY"
    solution = solve_cryptarithmetic(equation)

    if solution:
        print("Solution found:")
        for letter, digit in solution.items():
            print(f"{letter} = {digit}")
    else:
        print("No solution exists.")
