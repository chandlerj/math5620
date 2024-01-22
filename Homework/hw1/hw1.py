import numpy as np

def central_difference(f, x, h):
    return (f(x + h) - f(x - h)) / 2 * h

# u(bar(X)) + u'(bar(x))(x_i - bar{x})
def q2(f, xi, x: np.array) -> np.array:
    results = np.empty(len(x))
    for idx, i in enumerate(x):
        res = f(i) + central_difference(f, i, 0.001) * (i - xi)
        results[idx] = res

    return results

def nth_derivative_at_point(f, x, n, h=1e-5):
    """
    Compute the n-th derivative of a function at a specific point using finite differences.

    Parameters:
    - f: Function to differentiate
    - x: Point at which to compute the derivative
    - n: Order of the derivative
    - h: Step size for numerical differentiation

    Returns:
    - The n-th derivative of f at x
    """
    # Define a list of points for finite difference
    x_points = [x + i * h for i in range(-n//2, n//2 + 1)]

    # Evaluate the function at the specified points
    y_values = [f(point) for point in x_points]

    # Construct the coefficients for the finite difference formula
    coefficients = np.polyfit(x_points, y_values, n)

    # Use numpy's polyder function to differentiate the coefficients
    derivative_coefficients = np.polyder(coefficients)

    # Evaluate the derivative at the given point
    result = np.polyval(derivative_coefficients, x) / (h ** n)

    return result

def x_squared(x):
    return x**2

def main():
    points = np.array([8.1, 8.11, 7.98, 7.5, 7.8])
    print(q2(x_squared, 8, points))
    print(f"first derivative of x^2(2) = {nth_derivative_at_point(x_squared, 2, 1)}")

    print(f"second derivative of x^2(2) = {nth_derivative_at_point(x_squared, 2, 2)}")
main()
