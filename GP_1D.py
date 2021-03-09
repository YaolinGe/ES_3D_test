import numpy as np
import matplotlib.pyplot as plt

# size
n = 100
sites1v = np.arange(1, n + 1).reshape(-1, 1)

# prior mean
m = 0
# compute east and north distances on grid
ww = np.ones([n, 1])
H = np.abs(sites1v * ww.T - ww * sites1v.T)

# matern kernel
phiM = .19
Sigma = (1 + phiM * H) * np.exp(-phiM * H)

# exponential covariance
# range = 40
# Sigma = np.exp(-(3/range) * H)
# plt.imshow(Sigma)
# plt.title("Sigma, covariance matrix")
# plt.show()


# Compute Cholesky factor
L = np.linalg.cholesky(Sigma)
L_inv = np.linalg.inv(L)
# what does inverse of L tell?

# sample zero mean part
r = np.dot(L, np.random.randn(n).reshape(-1, 1))
# sample by adding mean from prior
r = r + m
plt.plot(r, 'k')
# Sample with design matrix
M = 50 # total number of samples
F = np.zeros([M, n])
des = np.ceil(n * np.random.rand(M)).astype(int) # remember to use astype int
# so matrix index will be integer
for i in range(M):
    F[i, des[i]] = True

# plt.imshow(F)
# plt.scatter(F)
# plt.plot(F)
# plt.show()

# measurement noise addition
tau = .05
# sample data
y = np.dot(F, r) + tau * np.random.rand(M).reshape(-1, 1)
plt.scatter(des, y)
# plt.show()

# Prediction surface from data
C = np.dot(F, np.dot(Sigma, F.T)) + np.diag(tau ** 2 * np.ones([M, 1]))
# plt.imshow(C)
# plt.title("covariance matrix")
# plt.show()

mx = np.ones([n, 1]) * m
rhat = mx + np.dot(Sigma, np.dot(F.T, np.linalg.solve(C, (y - np.dot(F, mx)))))
plt.plot(rhat, 'r:')
# plt.show()

# predication variances
Vvhat = Sigma - np.dot(Sigma, np.dot(F.T, np.linalg.solve(C, np.dot(F, Sigma))))
# plt.imshow(Vvhat)
# plt.show()
vr = np.diag(Vvhat).reshape(-1, 1)

# show the confidence interval
rlow = rhat - 1.28 * np.sqrt(vr)
rupp = rhat + 1.28 * np.sqrt(vr)

plt.plot(rlow, 'g')
plt.plot(rupp, 'g')
plt.show()





#%%
a = np.random.rand(10000)
plt.hist(a)
plt.show()





print("hello world")



#%%
print("hello world")

import numpy as np
import matplotlib.pyplot as plt
n = 100
x = np.arange(n).reshape(-1, 1)
y = np.sin(1 * x)

# plt.plot(x, y)
# plt.show()

# set up prior
prior = np.zeros_like(x)

# set up the distances matrix
H = np.abs(x * np.ones([1, n]) - np.ones([n, 1]) * x.T)
# plt.imshow(H)
# plt.show()

# set up covariance matrix
phiM = .19
Sigma = (1 + phiM * H) * np.exp(-phiM * H)
# plt.imshow(Sigma)
# plt.show()

# compute chol
L = np.linalg.cholesky(Sigma)

# sample with measurement noise
# first design matrix
M = 10
F = np.zeros([M, n])
ind = np.random.randint(n, size = M).reshape(-1, 1)
y_meas = np.zeros([M, 1])
tau = .05
for i in range(M):
    F[i, ind[i]] = True
    y_meas[i] = y[ind[i]]
y_test = np.dot(F, y) + tau * np.random.randn(M, 1)
y_meas = y_meas + tau * np.random.randn(M, 1)

# compute sampling matrix covariance
C = np.dot(F, np.dot(Sigma, F.T)) + tau ** 2 * np.ones([M, M])
# plt.imshow(C)
# plt.show()

# compute posterior mean and covariacne

mu_post = np.dot(Sigma, np.dot(F.T, np.linalg.solve(C, (y_meas - np.dot(F, prior)))))
cov_post = Sigma - np.dot(Sigma, np.dot(F.T, np.linalg.solve(C, np.dot(F, Sigma))))
std_error = np.sqrt(np.diag(cov_post)).reshape(-1, 1)
ci_up = mu_post + 1.28 * std_error
ci_low = mu_post - 1.28 * std_error

plt.plot(x, y, 'k')
plt.plot(ind, y_meas, 'k*')
plt.plot(ind, y_test, 'rs')
plt.plot(mu_post, 'g:')
plt.plot(ci_low, 'y--')
plt.plot(ci_up, 'y--')
plt.show()

plt.imshow(cov_post)
plt.colorbar()
plt.show()











