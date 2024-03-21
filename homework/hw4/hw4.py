import math
import numpy as np
from scipy.sparse import diags

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


    def build_A(self) -> np.array:
        """
        h - value to multiply h by
        m - size of 
        """
        m = self.n-1
        A = np.array
        k = [np.ones(m*(m-1)), np.ones((m**2) -1), -4*np.ones(m**2), np.ones((m**2)-1), np.ones(m*(m-1))] 
        offset = [-m,-1,0,1,m]
        A = diags(k, offset).toarray()
        A *= (self.h)**2
        return A
   

    def build_w(self) -> np.array:
        # build the boundary of the matrix
        # recall g(x,y) = 0
        return np.zeros((self.n, self.n))


    def build_r(self):
        r = np.zeros((self.n - 1)**2)
        for i in range(0, self.n-1):
            for j in range(0, self.n-1):
                r[i+(self.n-1)*j] = f(i+1, j+1)
        return r


    def solve_system(self):
        # use numpy methods to solve system of equations
        A = self.build_A()
        r = self.build_r()
        X,Y = np.meshgrid(self.x, self.y)
        A_inv = np.linalg.inv(A)
        
        C = np.dot(A_inv, r)
        return C


    def solve_system_jacobi(self):
        # use jacobi iteration to solve system of equations

        # TODO: make matrix sparse
        pass


    def solve_system_conjugate_gradient(self):
        pass
test = poissonSolver(f, 10)
zeros = test.build_A()
r = test.build_r()
soln = test.solve_system()
print(soln)
