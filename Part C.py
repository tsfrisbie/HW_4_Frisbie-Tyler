#Chat GPT helped with this code
#Started hw to late and was unable to find a partner

import numpy as np # Imports numpy as variable np

def solve_equation_1():
    """
    Defines, and solves equations one which is the first matrix in this problem.
    Returns:
        numpy.ndarray: Solution vector [x1, x2, x3].
    """
    # Define the coefficient matrix for the first equation
    A = np.array([[3, 1, -1],
                  [1, 4, 1],
                  [2, 1, 2]])
    # Define the constant vector for the first equation
    b = np.array([2, 12, 10])
    # Solve the equation Ax = b for x using numpy's linalg.solve()
    x = np.linalg.solve(A, b)
    # Round the solution to five significant figures
    x = np.round(x, decimals=5)
    return x

def solve_equation_2():
    """
    Defines, and solves the second equation which is the second matrix.
    Returns:
        numpy.ndarray: Solution vector [x1, x2, x3, x4].
    """
    # Define the coefficient matrix for the second equation
    A = np.array([[1, -10, 2, 4],
                  [3, 1, 4, 12],
                  [9, 2, 3, 4],
                  [-1, 2, 7, 3]])

    # Define the constant vector for the second equation
    b = np.array([2, 12, 21, 37])
    # Solve the equation Ax = b for x using numpy's linalg.solve()
    x = np.linalg.solve(A, b)
    # Round the solution to five significant figures
    x = np.round(x, decimals=5)
    return x

# Main program
if __name__ == "__main__":
    # Call the function to solve the first equation
    solution_1 = solve_equation_1()
    # Print the solution for the first equation
    print("Solution for matrix one:")
    print("x1 =", solution_1[0])
    print("x2 =", solution_1[1])
    print("x3 =", solution_1[2])

    # Call the function to solve the second equation
    solution_2 = solve_equation_2()

    # Print the solution for the second equation
    print("\nSolution for matrix two:")
    print("x1 =", solution_2[0])
    print("x2 =", solution_2[1])
    print("x3 =", solution_2[2])
    print("x4 =", solution_2[3])