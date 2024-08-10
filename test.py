import numpy as np

arr = np.zeros((3, 3, 3))
print(arr)

arr[0, 0, :] = [1, 0, 0]
print(arr)