import numpy as np




def explicit_euler(a, b, h, p0, function):

    time_step = np.arange(0, 100, h)
    P = np.zeros(len(time_step)) # stores state at a given time step
    P[0] = p0
    
    for i in range(0, len(P) -1):
        P[i + 1] = P[i] + h*function(a, b, P[i])

    return P

def implicit_euler(a, b, h, p0, function):
    
    time_step = np.arange(0, 100, h)
    P = np.zeros(len(time_step))
    P[0] = p0
    explicit_results = explicit_euler(a, b, h, p0, function)
    
    for i in range(0, len(P) - 1):
        P[i + 1] = P[i] + (h * function(a, b, explicit_results[i + 1]))

    return P

def trapezoid_euler(a, b, h, p0, function):
    
    time_step = np.arange(0, 100, h)
    P = np.zeros(len(time_step))

    P[0] = p0

    for i in range(0, len(P) - 1):
        predicate = P[i] + h * function(a, b, P[i])
        corrector = P[i] + h/2 * (function(a, b, P[i]) + function(a, b, predicate))
        P[i + 1] = corrector
    return P


def determine_carrying_capacity(a, b, p0):
    return ((a/b) * p0) / p0


function = lambda a, b, P: a * P - (b * P**2)

print(explicit_euler(1.5, 0.001, 0.01, 10, function))
print(explicit_euler(1.5, 0.01, 0.01, 10, function))
print(explicit_euler(0.15, 0.001, 0.01, 10, function))
#print(implicit_euler(1.5, 0.01, 0.01, 10, function))
#print(trapezoid_euler(1.5, 0.01, 0.01, 10, function))
#print(determine_carrying_capacity(1.5, 0.01, 10))
