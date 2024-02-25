#Chat GPT helped with this code
#Started hw to late and was unable to find a partner

import numpy as np #imports numpy
from scipy.optimize import fsolve #imports scipy

def equation1(x):
    """
    Defines the first equation: x - 3*cos(x) = 0.

    Args:
        x (float): The variable.
    Returns:
        float: The value of the equation at x.
    """
    return x - 3 * np.cos(x)

def equation2(x):
    """
    Defines the second equation: cos(2*x) * x**3 = 0.

    Args:
        x (float): The variable.

    Returns:
        float: The value of the equation at x.
    """
    return np.cos(2 * x) * x ** 3

# Find the roots of the equations
root1 = fsolve(equation1, 0)  # Initial guess for the root
root2 = fsolve(equation2, [1, 2, 3])  # Multiple initial guesses for the root

# Check if the functions intersect and find the intersection point
# Displays whether or not roots intersect
if equation1(root2[0]) * equation1(root2[1]) < 0:
    intersection_point = fsolve(equation1, np.mean(root2))
    print("The functions intersect at x =", intersection_point[0])
else:
    print("The functions do not intersect.")

# Print the roots
print("Roots of equation 1:", root1) # Displays root for eq 1
print("Roots of equation 2:", root2) # Displays root for eq 2