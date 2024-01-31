import math
from typing import Callable
from scipy.sparse import diags
import numpy as np



class FiniteDifference:
    interval_points = []
    
    approx_solutions = []
    function_values = []
    matrix = np.array 
    def __init__(self, left_endpoint: float, right_endpoint: float, num_points: int, fx: Callable) -> None:

        self.left_endpoint = left_endpoint
        self.interval_points.append(left_endpoint)

        self.right_endpoint = right_endpoint
        self.interval_points.append(right_endpoint)

        self.num_points = num_points 

        self.create_interval_points(num_points)
        
        self.fx = fx 
        print(self.interval_points, len(self.interval_points))

    def determine_mesh_interval(self):
        return (self.right_endpoint - self.left_endpoint) / (self.num_points - 1)

    def create_interval_points(self, num_points):
        value_interval = self.determine_mesh_interval()
        
        curr_val = self.left_endpoint + value_interval 
        for i in range(1, num_points - 1):
            self.interval_points.insert(i, curr_val)
            curr_val += value_interval
        
        # add the m+1st point

        curr_val += value_interval
        self.interval_points.append(curr_val)

    def construct_matrix(self):
        n = 10
        k = [np.ones(n-1),-2*np.ones(n),np.ones(n-1)]
        offset = [-1,0,1]
        self.matrix = diags(k,offset).toarray()

    def evaluate_function(self, x):
        return self.fx(x)

def main(): 
    test = FiniteDifference(0, 1, 8, lambda x: math.sin(x))  
    print(test.evaluate_function(math.radians(90)))
    test.construct_matrix()
    print(test.matrix)
main()
