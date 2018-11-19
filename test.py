import numpy as np
a = np.array([[0,1,0,0],[1,0,0,0],[0,0,0,1]])
b = [np.where(r==1)[0][0] for r in a]
print(b)