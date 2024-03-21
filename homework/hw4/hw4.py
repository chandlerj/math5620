import math
import numpy as np
from scipy.sparse import diags

# the function we want to determine unknowns of
f = lambda x,y: 3 * math.sin(math.pi * x) * math.sin(4 * math.pi * y)

def build_A(h:int) -> np.array:
    """
    h - value to multiply h by
    m - size of 
    """
    m = h
    A = np.array
    k = [np.ones(m*(m-1)), np.ones((m**2) -1), -4*np.ones(m**2), np.ones((m**2)-1), np.ones(m*(m-1))] 
    offset = [-m,-1,0,1,m]
    A = diags(k, offset).toarray()
    A *= 1/h**2
    return A

def build_w(n:int) -> np.array:
    pass
print(build_A(5))
