import math

class FiniteDifference:
    interval_points = []

    def __init__(self, left_endpoint, right_endpoint, num_points, function):
        self.interval_points.append(left_endpoint)
        self.interval_points.append(right_endpoint)
        
        self.create_interval_points(num_points)
        
        print(self.interval_points, len(self.interval_points))


    def create_interval_points(self, num_points):

        left_e = self.interval_points[0]
        right_e = self.interval_points[-1]
        #self.interval_points.remove(right_e)

        value_interval = (right_e - left_e) / (num_points - 1)
        curr_val = left_e + value_interval 
        for i in range(1, num_points - 1):
            self.interval_points.insert(i, curr_val)
            curr_val += value_interval


test = FiniteDifference(5, 20, 8, lambda x: math.sin(x))  
