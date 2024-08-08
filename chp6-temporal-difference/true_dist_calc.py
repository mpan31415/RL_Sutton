from numpy.linalg import solve
from numpy import array, eye

A = array([[0,1,0,0,0],
           [1,0,1,0,0],
           [0,1,0,1,0],
           [0,0,1,0,1],
           [0,0,0,1,0]]) - 2*eye(5, dtype=int)

b = array([0,0,0,0,-1])

x = solve(A, b)

print(x)
