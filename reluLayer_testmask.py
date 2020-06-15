import numpy as np
from reluLayer import Relu

x = np.array([[1.0, -0.5], [-2.0, 3.0]])
print(x)

mask = (x <=0)
print(mask)