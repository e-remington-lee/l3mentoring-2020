import numpy as np


x = np.array(1)
y = np.array((1,2))

print(x.ndim)
print(y.ndim)

x,y = [{"a" : 1}, {"a" : 1}]
print(x, y)