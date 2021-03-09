import numpy as np
import matplotlib.pyplot as plt

print("hello world")
# grid size
n1 = 25
n = n1 * n1

# define regular grid of locations
vv = np.arange(n1).reshape(-1, 1)
uu = np.ones([n1, 1])
sites1 = vv * uu.T
sites2 = uu * vv.T
# plt.plot(sites1, sites2, 'ko')
# plt.title("grid")
# plt.show()

# vectorise the grid
sites1v = sites1.flatten().reshape(-1, 1)
sites2v = sites2.flatten().reshape(-1, 1)
# true_field = np.sin(.2 * sites2v) + np.cos(.03 * sites1v)
# plt.plot(sites1v, sites2v, 'k.')
# plt.title("grid decomposition")
# plt.show()

# plt.imshow(true_field.reshape(n1, n1))
# plt.title("true_field")
# plt.show()

# prior mean
m = 0
# compute distance
ww = np.ones([n, 1])
ddE = np.dot(sites1v, ww.T) - np.dot(ww, sites1v.T)
dd2E = ddE * ddE
ddN = np.dot(sites2v, ww.T) - np.dot(ww, sites2v.T)
dd2N = ddN * ddN
H = np.sqrt(dd2E + dd2N)
# plt.imshow(H)
# plt.title("distance matrix")
# plt.show()

# compute covariance
phiM = .19
Sigma = (1 + phiM * H) * np.exp(-phiM * H)
# plt.imshow(Sigma)
# plt.title("covariance matrix")
# plt.show()

# compute cholesky
L = np.linalg.cholesky(Sigma)

# sample random part
x = np.dot(L.T, np.random.randn(n).reshape(-1, 1))
prior = np.ones([n, 1]) * m
# prior = true_field.reshape(-1, 1)
x = x + prior
xm = x.reshape(n1, n1)
plt.imshow(xm)
plt.title("prior mean")
plt.colorbar()
plt.show()

# sample from the grid
M = 100
F = np.zeros([M, n])
ind = np.random.randint(n, size = M)
for i in range(M):
    F[i, ind[i]] = True

# measure from the true fieldr
tau = .05
y = np.dot(F, x) + tau * np.random.randn(M).reshape(-1, 1)
# y = np.dot(F, true_field.reshape(-1, 1)) + tau * np.random.randn(M).reshape(-1, 1)

# compute C matrix
C = np.dot(F, np.dot(Sigma, F.T)) + np.diag(np.ones([M, 1]) * tau ** 2)

# compute posterior mean
xhat = prior + np.dot(Sigma, np.dot(F.T, np.linalg.solve(C, (y - np.dot(F, prior)))))
xhatm = xhat.reshape(n1, n1)
plt.imshow(xhatm)
plt.title("posterior mean")
plt.colorbar()
plt.show()

# compute posterior covariance
Vvhat = Sigma - np.dot(Sigma, np.dot(F.T, np.linalg.solve(C, np.dot(F, Sigma))))
vhat = np.diag(Vvhat)
vhatm = vhat.reshape(n1, n1)
plt.imshow(vhatm)
plt.colorbar()
plt.title("posterior variance")
plt.show()
