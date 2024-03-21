# std lib
import math
# third party lib
from typing import Callable
from scipy.sparse import diags
import numpy as np
from matplotlib import pyplot as plt


class FiniteDifference:


    def __init__(self, left_endpoint: float, right_endpoint: float, num_points: int,  fx: Callable, exact_function: Callable, h_val = None) -> None:
        # initialize data

        self.function_values = []
        self.interval_points = []
        self.h = 0
        self.approx_solutions = []
        self.matrix = np.array 

        self.left_endpoint = left_endpoint
        self.interval_points.append(left_endpoint)
        self.right_endpoint = right_endpoint
        self.interval_points.append(right_endpoint)
        self.num_points = num_points 

        self.mesh_interval = self.determine_mesh_interval()
        self.create_interval_points()
        
        self.fx = fx 
        self.exact_function = exact_function

        if h_val == None:
            self.h = 1 / (1 + self.mesh_interval)
        else:
            self.h = h_val

    def determine_mesh_interval(self):
        return (self.right_endpoint - self.left_endpoint) / (self.num_points - 1)


    def create_interval_points(self):
        curr_val = self.left_endpoint + self.mesh_interval
        for i in range(1, self.num_points - 1):
            self.interval_points.insert(i, curr_val)
            curr_val += self.mesh_interval
   

    def construct_matrix(self):
        k = [np.ones(self.num_points-1),-2*np.ones(self.num_points),np.ones(self.num_points-1)]
        offset = [-1,0,1]
        self.matrix = diags(k,offset).toarray()


    def create_approximate_solutions(self):
        self.function_values.append(self.left_endpoint)

        # use 2nd order finite difference approx 
        for i in range(1,self.num_points - 1):
            second_difference = (1 / self.h**2) * (self.fx(self.interval_points[i - 1]) - (2 * self.fx(self.interval_points[i])) + self.fx(self.interval_points[i + 1]))
            self.function_values.append(second_difference)
        self.function_values.append(self.right_endpoint)


    def solve_linear_system(self):
        func_vals = np.array(self.function_values).transpose()
        self.approx_solutions = np.linalg.solve(self.matrix, func_vals)


    def solve(self):
        self.construct_matrix()
        self.create_approximate_solutions()
        self.solve_linear_system()


    def compute_global_error(self):   
        exact_solutions = []
        error_at_term =  0
        # find exact solutions
        for point in self.interval_points:
            exact_solutions.append(self.exact_function(point))
        # find error between approx soln and exact soln
        for i, solution in enumerate(exact_solutions):
            error_at_term += (abs(self.approx_solutions[i] - solution)**2)
        # compute l2-norm
        l2_norm = math.sqrt(error_at_term)
        return l2_norm


def display_data(solutions):
    plt.plot(solutions)
    plt.show()


def diff_coeffs(nder: int, a: float, b: float, xbar: float):
    #
    # start by setting the domain and point in the domain where the
    # derivative is to be approximated
    # --------------------------------
    #
    print(a,b,xbar)
    #
    # for this code, the number of points used in the algorithm is going to
    # be fixed to be one more than the degree of the derivative desired.
    # ------------------------------------------------------------------
    #
    mp1 = nder + 1
    m = mp1 - 1
    h = ( b - a ) / m
    #
    # Initialize an array of equally spaced points
    # --------------------------------------------
    #
    xpts = np.zeros(mp1)
    for j in range(mp1):
        xpts[j] = a + j * h
    #
    # Intialize the matrix (Vandermond matrix) for the coefficient matrix and
    # the associated right hand side for the approximation of a given derivative.
    # ---------------------------------------------------------------------------
    #
    amat = np.ones((mp1,mp1))
    for i in range(1,mp1):
        for j in range(mp1):
            amat[i][j] = amat[i-1][j] * ( xpts[j] - xbar )
    #
    # set the right hand side to match the correct derivative
    # -------------------------------------------------------
    #
    rhs = np.zeros(mp1)
    rhs[nder] = 1.0
    #
    # Perform row reduction or Gaussian elimination of the system to an upper
    # triangular matrix.
    # -----------------
    #
    for k in range(m):
        for i in range(k+1,mp1):
            val = amat[i][k] / amat[k][k]
            for j in range(k+1,mp1):
                amat[i][j] = amat[i][j] - val * amat[k][j]
            rhs[i] = rhs[i] - val * rhs[k]
    #
    # Use backsubstitution to get the coefficients.
    # ---------------------------------------------
    #
    coeff = np.zeros(mp1)
    coeff[m] = rhs[m] / amat[m][m]
    for i in range(m-1,-1,-1):
        sum = 0.0
        for j in range(i+1,mp1):
            sum = sum + amat[i][j] * coeff[j]
            coeff[i] = ( rhs[i] - sum ) / amat[i][i]
    #
    # return the coefficients computed above
    # --------------------------------------
    #
    return coeff


def test_finite_solver():
    solution_sets = [] # holds the approximate solutions to each mesh width
    h_vals = [8, 16, 32, 64, 256, 512, 1024]

    exact_function = lambda x : (-1 * (4 * math.cos((3/2) * math.pi * x) ) / (9 * math.pi**2))
    ODE_equation = lambda a : math.cos(((3 * math.pi)/2)*a)
    for val in h_vals:
        solver = FiniteDifference(0, 1, 5, ODE_equation, exact_function,  val)
        solver.solve()
        print(solver.approx_solutions)
        print(solver.compute_global_error())
        solution_sets.append(solver.compute_global_error())
    display_data(solution_sets)

def main(): 
    #test_finite_solver()
    print(diff_coeffs(5, 0.0, 1.0, -1.0))

main()
