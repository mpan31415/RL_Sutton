import numpy as np

rng = np.random.default_rng()

n = np.array([rng.uniform(-1, 1)])
print(n)
print(type(n))
print(n.shape)