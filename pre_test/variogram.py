import numpy as np
import random
import matplotlib.pyplot as plt

S = 1000
N = 900
x = np.linspace(0, 2*np.pi, S)
sx = np.sin(3 * x) * np.sin(10 * x)
density = .8 * abs(np.outer(sx, sx))
density[:, :S//2] += .2
random.seed(10)
points = []

while len(points) < N:
    v, ix, iy = random.random(), random.randint(0, S-1), random.randint(0, S-1)
    if True:
        points.append([ix, iy, density[ix, iy]])

locations = np.array(points).transpose()

L = locations
m = np.array([[np.sqrt((L[0,i]-L[0,j])**2+(L[1,i]-L[1,j])**2), L[2,i], L[2,j]]
                         for i in range(N) for j in range(N) if i>j])



## %% test of matrix
# n = 10
# # t = np.zeros([n, n])
# # print(t)
# t = np.array([[np.sqrt((L[0,i]-L[0,j])**2+(L[1,i]-L[1,j])**2), L[2,i], L[2,j]] for i in range(n) for j in range(n) if i > j])
# print(t)
#%%

# now do the real calculations for the covariogram
#    sort by h and give clear names
i = np.argsort(m[:,0])  # h sorting
h = m[i,0]
zh = m[i,1]
zsh = m[i,2]
zz = zh*zsh

hvals = np.linspace(0,S,1000)  # the values of h to use (S should be in the units of distance, here I just used ints)
ihvals = np.searchsorted(h, hvals)
result = []
for i, ihval in enumerate(ihvals[1:]):
    start, stop = ihvals[i-1], ihval
    N = stop-start
    if N>0:
        mnh = sum(zh[start:stop])/N
        mph = sum(zsh[start:stop])/N
        szz = sum(zz[start:stop])/N
        C = szz-mnh*mph
        result.append([h[ihval], C])
result = np.array(result)
plt.plot(result[:,0], result[:,1])
plt.grid()
plt.show()


#%% Use packages
from skgstat import Variogram
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
np.random.seed(42)








