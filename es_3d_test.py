from es_strategies import *
from usr_defined_func import *

# set up grid
nx = 31
ny = 31
nz = 2

# thresholds
T_thres = 7.16
S_thres = 31.208

# only one parameter is considered
beta_t = [[5.8], [0.098], [0.08]]
beta_s = [[29.0], [0.133], [0.15]]

# correlation paras t & s
sig_st = 0.6

# correlation parameters among depths
sig_d1 = 0.7

# noise
noise_std = 0.05 # noise standard error
noise = np.array([[noise_std ** 2, 0], [0, noise_std ** 2]])

# sal std
sig = 0.025 # system error, random effect
corr_range = 0.3 # 0.8 == 900m

n1 = nx
n2 = ny
n3 = nz
n = n1 * n2 * n3 # total number of grid points
nf = n1 * n2 # only consider the number of grid points in one layer. nf refers to n of flat layer

x = np.arange(1, n1 + 1, 1)
y = np.arange(1, n2 + 1, 1)
z = np.arange(1, n3 + 1, 1)

zz, yy, xx = np.mgrid[1:n3+1, 1:n2+1, 1:n1+1] # since mgrid tends to 1st one as depth layer

# Vectorise the grid
xv = xx.flatten().reshape(-1, 1)
yv = yy.flatten().reshape(-1, 1)
zv = zz.flatten().reshape(-1, 1)

sites = np.hstack((xv, yv, zv)) # so to have to the design matrix, intercept is added later

# covars = np.hstack((np.repeat(1, n).reshape(-1, 1), np.flipud(np.tile(np.arange(1, n1, 1), n1)).reshape(-1, 1)))
xintercept = np.ones([n, 1])
# x1 = np.flipud(np.tile(np.arange(0, n1, 1), n1).reshape(-1, 1)).reshape(-1, 1) # use x1 as the gradually decreasing neighbouring coordinates
# x2 = np.flipud(np.tile(np.arange(0, n3, 1), n1).reshape(-1, 1)).reshape(-1, 1)
covars = np.hstack((xintercept, np.flipud(xv - 1), np.flipud(zv - 1))) # stack the vector to make it as [x0, x1, x2]
## ===== ##
'''Noter
Here covars only takes y direction as the major direction, so that the rest can be arranged as the distance with respect to one point
To have distance decay so to make it as a regression problem
'''

# compute prior & true field
mu_t_prior = np.dot(covars, [[5.8], [0.085], [0.07]])
mu_s_prior = np.dot(covars, [[29.0], [0.136], [0.12]])
mu_prior = np.concatenate((mu_t_prior, mu_s_prior))
prior = np.array(mu_prior)

# true field
mu_t = np.dot(covars, beta_t) # beta_t, is assumed to be the true coeff
mu_s = np.dot(covars, beta_s) # beta_s is assumed to be the true beta_s
mu_stack = np.concatenate((mu_t, mu_s)) # true field

# compute covariance
# Cd = buildcov3d(sites, sites, sig, corr_range, noise_std ** 2)
# C0d = [[1, sig_st], [sig_st, 1]]
# Ct_true_field = np.kron(C0d, Cd)

H, C = buildcov3d(sites, sites, sig, corr_range, noise_std ** 2, noise_true=False)
C0 = [[1, sig_st], [sig_st, 1]] # correlation matrix for salinity & temperature
Ct = np.kron(C0, C) # final matrix considering depth variation & salinity & temperature correlation

plotf(np.copy(Ct), "Covariance matrix for depth & salinity & temperature correlation")
plotf(np.copy(C), "Pure matern matrix without salinty & temp correlation, without kronecker")
plotf(np.copy(H), "Pure euclidean distance matrix") # distance matrix
plotf3d(prior[0:n], xv, yv, zv) # plot the prior in 3D, it worked, but concern may arise when it comes to isomin and isomax

init_trace = np.trace(Ct)

# compute the ES prob for prior estimates
pp = []
for i in range(0, nf):
    Sigma_selected = Ctd[np.ix_([i, nf + i, 2 * nf + i, 3 * nf + i], [i, nf + i, 2 * nf + i, 3 * nf + i])]
    Mxi = [prior[i], prior[nf + i], prior[2 * nf + i], prior[3 * nf + i]]
    # pp.append(mvn.mvnun(np.array([[-np.inf], [-np.inf]]), np.array([[0], [0]]), np.subtract([T_thres, S_thres], np.array(Mxi).ravel()), Sigma_selected)[0])

# plotf(np.array(pp).reshape(nx, ny), 'Prior Exc.Set - T. {} & S. {}'.format(T_thres, S_thres))


#%%
# compute the expected variance
prior_0 = np.copy(prior)
Ct_0 = np.copy(Ct)
Pred_error = None
prior_ev = ExpectedVariance2([T_thres, S_thres], prior, Ct, np.zeros((2 * 1, 2 * nx * ny)), noise * np.eye(2 * 1), np.arange(0, nx * ny))
#EIBV

#%% generate the realisation of the prior
# generate true sal_temp field where it is based on simulated data
true = np.array(mu_prior + np.dot(np.linalg.cholesky(Ct), np.random.randn(Ct .shape[0],1)))
plotf3d(true[0:n], xv, yv, zv)
# plotf3d(prior[n:], xv, yv, zv, "salinity prior")
# plotf(np.copy(true[0:n]).reshape(nx, ny), "true temp")
# plotf(np.copy(true[n:]).reshape(nx, ny), "true salinity")

#%%
excursion_t = true[0:n].copy()
excursion_t[true[0:n]>T_thres] = 1
excursion_t[true[0:n]<T_thres] = 0

excursion_s = true[n:].copy()
excursion_s[true[0:n]>S_thres] = 1
excursion_s[true[0:n]<S_thres] = 0

excursion_st = np.multiply(excursion_t, excursion_s)
excursion_ts = np.multiply(excursion_s, excursion_t)

plotf(excursion_t.reshape(n1, n2), "es_boundary_temp")
plotf(excursion_s.reshape(n1, n2), "es_boundary_sal")
plotf(excursion_st.reshape(n1, n2), "es_boundary_st")

pp = []
for i in range(0, n):
    SS = Ct[np.ix_([i, n + i], [i, n + i])]
    Mxi = [true[i], true[n + i]]
    pp.append(mvn.mvnun(np.array([[-np.inf], [-np.inf]]), np.array([[0], [0]]),
                        np.subtract([T_thres, S_thres], np.array(Mxi).ravel()), SS)[0])

plotf(np.array(pp).reshape(nx, ny), "es_prob")


ym_t = []
ym_s = []
num_of_obs = 0
yp_t = []
yp_s = []
num_of_obs = ny
# define design matrix
G = np.zeros((2 * num_of_obs, 2 * n))

# update the posterior mean and variance
for i in range(ny):
    x_ind = int(np.round(nx / 2))
    y_ind = i
    print(x_ind)
    print(y_ind)
    ind = np.ravel_multi_index((x_ind, y_ind), (nx, ny))
    print(ind)
    G[i, ind] = 1
    ym_t.extend(true[0:n][ind])
    ym_s.extend(true[n:][ind])
    yp_t.extend(prior[0:n][ind])
    yp_s.extend(prior[n:][ind])

ym = np.hstack((ym_t, ym_s))
yp = np.hstack((yp_t, yp_s))

# define noise
R = np.diag(np.repeat(noise[0, 0], num_of_obs).tolist() + np.repeat(noise[1, 1], num_of_obs).tolist())

# compute the conditional distribution
C1 = np.dot(Ct, G.T)
C2 = np.dot(G, np.dot(Ct, G.T)) + R
dev = ym - yp
similarity = np.array(np.linalg.lstsq(C2, dev, rcond = None)[0])
uncertainty_reduction = np.array(np.linalg.lstsq(C2, np.dot(G, Ct), rcond = None)[0])
prior += np.dot(C1, similarity)[:, None]
Sigcond = Ct - np.dot(C1, uncertainty_reduction)


# compute the updated es probabilities
pp = []
for i in range(0, n):
    SS = Sigcond[np.ix_([i, n + i], [i, n + i])]
    Mxi = [prior[i], prior[n + i]]
    pp.append(mvn.mvnun(np.array([[-np.inf], [-np.inf]]), np.array([[0], [0]]),
                        np.subtract([T_thres, S_thres], np.array(Mxi).ravel()), SS)[0])

plotf(np.copy(prior[0:n]).reshape(nx, ny), "posterior temp")
plotf(np.copy(prior[n:]).reshape(nx, ny), "posterior salinity")

plotf(np.array(pp).reshape(nx, ny), "es_prob")

Pred_error1 = np.diag(Sigcond)[0:n]
Pred_error2 = np.diag(Sigcond)[n:]

plotf(np.copy(Pred_error1).reshape(nx, ny), "pred_error_temp")

plotf(np.copy(Pred_error2).reshape(nx, ny), "pred_error_sal")



#%%
# ======== # ========# ========# ========# ========# ========# ========# ========

# another path planning
# update the posterior mean and variance
for i in range(nx):
    x_ind = i
    y_ind = int(np.round(ny / 2))
    print(x_ind)
    print(y_ind)
    ind = np.ravel_multi_index((x_ind, y_ind), (nx, ny))
    print(ind)
    G[i, ind] = 1
    ym_t.extend(true[0:n][ind])
    ym_s.extend(true[n:][ind])
    yp_t.extend(prior[0:n][ind])
    yp_s.extend(prior[n:][ind])

ym = np.hstack((ym_t, ym_s))
yp = np.hstack((yp_t, yp_s))

# define noise
R = np.diag(np.repeat(noise[0, 0], num_of_obs).tolist() + np.repeat(noise[1, 1], num_of_obs).tolist())

# compute the conditional distribution
C1 = np.dot(Ct, G.T)
C2 = np.dot(G, np.dot(Ct, G.T)) + R
dev = ym - yp
similarity = np.array(np.linalg.lstsq(C2, dev, rcond = None)[0])
uncertainty_reduction = np.array(np.linalg.lstsq(C2, np.dot(G, Ct), rcond = None)[0])
prior += np.dot(C1, similarity)[:, None]
Sigcond = Ct - np.dot(C1, uncertainty_reduction)


# compute the updated es probabilities
pp = []
for i in range(0, n):
    SS = Sigcond[np.ix_([i, n + i], [i, n + i])]
    Mxi = [prior[i], prior[n + i]]
    pp.append(mvn.mvnun(np.array([[-np.inf], [-np.inf]]), np.array([[0], [0]]),
                        np.subtract([T_thres, S_thres], np.array(Mxi).ravel()), SS)[0])

plotf(np.copy(prior[0:n]).reshape(nx, ny), "posterior temp")
plotf(np.copy(prior[n:]).reshape(nx, ny), "posterior salinity")

plotf(np.array(pp).reshape(nx, ny), "es_prob")






#%%
# a = np.arange(0, 36, 1).reshape(6, 6)
# n = 3
# print(a)
# for i in range(0, n):
#     print(a[np.ix_([i, n+i], [i, n + i])])
a = np.arange(0, 10, 1).reshape(2, 5)
b = np.ones([2, 2]) * 3
C = np.kron(b, a)
D = np.kron(a, b)
print(a)
print(b)
print(C)
print(D)

#%% test of unravel and ravel
a = np.arange(0,50).reshape(5, 10)
print(a)
row, col = 3, 4
print(row, col)
print(a[row, col])
ind = np.ravel_multi_index((row, col), (5, 10))
b = a.flatten()
print(b)
print(b[ind])
c = a.reshape(-1, 1)
d = c.reshape(5, 10)
print(d)
# print(c)
print(c[ind])

#%% test of euclidean distance among those grid points
a =
h = np.sqrt(scdist.cdist(site_11, site_22, 'sqeuclidean'))
