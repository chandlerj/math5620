import math
import numpy as np
from scipy.sparse import diags

import matplotlib.pyplot as plt

# the function we want to determine unknowns of
f = lambda x,y: 3 * math.sin(math.pi * x) * math.sin(4 * math.pi * y)


class poissonSolver:


    def __init__(self, f, n):
        self.h = 1/n
        self.f = f
        self.n = n
        
        # create the interval within the bonudary
        self.x = np.arange(0, 1.00001, self.h)
        self.y = np.arange(0, 1.00001, self.h)


    def build_w(self) -> np.array:
        # build the boundary of the matrix
        # recall g(x,y) = 0
        return np.zeros((self.n**2, self.n**2))


    def build_A(self) -> np.array:
        m = self.n - 1
        A = np.array
        k = [np.ones(m*(m-1)), np.ones((m**2) -1), -4*np.ones(m**2), np.ones((m**2)-1), np.ones(m*(m-1))] 
        offset = [-m,-1,0,1,m]
        A = diags(k, offset).toarray()
        A *= (self.h)**2
        return A


    def build_r(self):
        r = np.zeros((self.n - 1)**2)
        for i in range(0, self.n-1):
            for j in range(0, self.n-1):
                r[i+(self.n-1)*j] = f(i+1, j+1)
        return r


    def solve_system_numpy(self):
        # use numpy methods to solve system of equations
        A = self.build_A()
        r = self.build_r()
        A_inv = np.linalg.inv(A)
        C = np.dot(A_inv, r)
        print(f'size of A: {A.shape}, size of r: {r.shape}')
        return C

    def solve_system_jacobi(self):
        # use jacobi iteration to solve system of equations
        def jacobi(A, b, N = 25, x=None):
            # Create an initial guess if needed                                                                                                                                                            
            if x is None:
                x = np.zeros(len(A[0]))

            # Create a vector of the diagonal elements of A                                                                                                                                                
            # and subtract them from A                                                                                                                                                                     
            D = np.diag(A)
            R = A - np.diagflat(D)

            # Iterate for N times                                                                                                                                                                          
            for _ in range(N):
                x = (b - np.dot(R,x)) / D
            return x
        A = self.build_A() 
        r = self.build_r()
        return jacobi(A, r)

    def solve_system_conjugate_gradient(self):
        A = self.build_A()
        r = self.build_r()
        p = r
        x = np.zeros((self.n - 1)**2)
        rsold = np.dot(np.transpose(r), r)
        
        for _ in range(len(r)):
            Ap = np.dot(A, p)
            alpha = rsold / np.dot(np.transpose(p), Ap)
            x = x + np.dot(alpha, p)
            r = r - np.dot(alpha, Ap)
            rsnew = np.dot(np.transpose(r), r)
            if np.sqrt(rsnew) < 1e-8:
                break
            p = r + (rsnew/rsold)*p
            rsold = rsnew
        return x

    def compute_convergence(self):
        results_jacobi = []
        results_numpy = []
        results_conjugate = []
        U = np.zeros((self.n - 1)**2)
        for i in range(0, self.n-1):
            for j in range(0, self.n-1):
                U[i+(self.n-1)*j] = f(i+1, j+1)
        J = self.solve_system_jacobi()
        N = self.solve_system_numpy()
        C = self.solve_system_conjugate_gradient()
        for i, value in enumerate(J):
            results_jacobi.append(abs(U[i] - value))
        for i, value in enumerate(N):
            results_numpy.append(abs(U[i] - value))
        for i, value in enumerate(C):
            results_conjugate.append(abs(U[i] - value))

        return (max(results_jacobi), max(results_numpy), max(results_conjugate))

def display_solver_convergence():
    r_jacobi = []
    r_numpy = []
    r_conjugate = []

    for i in range(2, 7):
        test = poissonSolver(f, 2**i)
        convergence = test.compute_convergence()
        r_jacobi.append(convergence[0])
        r_numpy.append(convergence[1])
        r_conjugate.append(convergence[2])
    plt.plot(r_jacobi, label= 'jacobi convergence')
    plt.plot(r_numpy, label = 'numpy built-in convergence')
    plt.plot(r_conjugate, label = 'conjugate gradient convergence')
    plt.legend()
    plt.show()
    print(r_jacobi, r_numpy, r_conjugate)


display_solver_convergence()



