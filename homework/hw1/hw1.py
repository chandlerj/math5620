import numpy as np
import math
def difference_coefficients(k: int, xbar: float, x: np.array) -> np.array:
    # determine size of input points
    n = len(x)
    
    # initialize the Vandermonde matrix with 1s
    A = np.ones((n, n))
    
    # subtract xbar from each element and reshape to a row
    xrow = (np.array(x) - xbar).reshape(1, -1)
    
    # construct the Vandermonde matrix
    for i in range(2, n + 1):
        A[i-1, :] = (xrow ** (i-1)) / math.factorial(i-1)
    
    # initialize the solution set
    b = np.zeros((n, 1))
    b[k] = 1
    
    # solve the linear system to determine the coefficients
    c = np.linalg.solve(A, b)
    print(b)    
    # return the coefficients as an array
    return c.flatten()

def check_coefficients(coefficients: np.array, step: float, x: float) -> np.array:
    n = len(coefficients)
    x_values = np.array([x + i * step for i in range(n)])
    Vandr = np.vander(x_values, increasing=True)
    res = np.linalg.solve(Vandr, x_values)
    return res # should return array of all 0s and 1 one.

def main():
    k = 2  # Order of the derivative
    xbar = 0.0  # Point where the derivative is approximated
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])  # Data points
    
    coefficients = difference_coefficients(k, xbar, x)
    print(f"Finite Difference Coefficients for {k}-th derivative: {coefficients}")
    print(f"Validation: {check_coefficients(coefficients, 0.01, 0.0)}")

main()
