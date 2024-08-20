from sudoku.sudoku import is_complete
import numpy as np

arr = np.array([[4, 8, 3, 9, 2, 1, 6, 5, 7],
                [9, 6, 7, 3, 4, 5, 8, 2, 1],
                [2, 5, 1, 8, 7, 6, 4, 9, 3],
                [5, 4, 8, 1, 3, 2, 9, 7, 6],
                [7, 2, 9, 5, 6, 4, 1, 3, 8],
                [1, 3, 6, 7, 9, 8, 2, 4, 5],
                [3, 7, 2, 6, 8, 9, 5, 1, 4],
                [8, 1, 4, 2, 5, 3, 7, 6, 9],
                [6, 9, 5, 4, 1, 7, 3, 8, 2]])
print(arr)

res = is_complete(arr)
print(res)

a = np.diag([3, 3, 3])
print(a)

b = np.array([1, 2, 3])
print(a@b)