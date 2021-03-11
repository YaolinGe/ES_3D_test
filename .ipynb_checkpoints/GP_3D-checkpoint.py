import numpy as np
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objects as go

def plotf(Y, string):
    plt.figure(figsize=(5,5))
    plt.imshow(Y)
    plt.title(string)
    plt.colorbar(fraction=0.045, pad=0.04)
    plt.gca().invert_yaxis()
    plt.show()

def plotf3d(val, X, Y, Z):

    # plt.figure()
    # plt.imshow(val)
    # plt.colorbar()
    # plt.ylabel('Northing')
    # plt.xlabel('Easting')
    # plt.title(title_string)
    # plt.tight_layout()
    # plt.show()
    # val /= val.max()
    fig = go.Figure(data=go.Volume(
        x=X, y=Y, z=Z,
        value=val,
        # isomin=-1,
        # isomax=1,
        # opacity=0.4,
        # surface_count=15,
        # colorscale='RdBu',
        # title = string
    ))
    # fig.update_layout(scene_xaxis_showticklabels=False,
    #                   scene_yaxis_showticklabels=False,
    #                   scene_zaxis_showticklabels=False)
    # plotly.offline.plot(fig)
    fig.show()


print("hello world")
# setup grid
n1 = 25
n2 = 25
n3 = 5
n = n1 * n2 * n3

# define regular grid of locations
uu = np.arange(n1).reshape(-1, 1)
vv = np.arange(n2).reshape(-1, 1)
ww = np.arange(n3).reshape(-1, 1)

# vectorise the field
sites1m, sites2m, sites3m = np.meshgrid(uu, vv, ww)
sites1v = sites1m.flatten().reshape(-1, 1)
sites2v = sites2m.flatten().reshape(-1, 1)
sites3v = sites3m.flatten().reshape(-1, 1)

# plot the field
fig = plt.figure(figsize = (10, 10))
ax0 = fig.add_subplot(111, projection = '3d')
ax0.scatter(sites1v, sites2v, sites3v, c = "black")
ax0.set(xlabel = 'easting', ylabel = 'northing')
plt.show()

# prior mean
m = 0

# compute distance
aa = np.ones([n, 1])
ddx = np.dot(sites1v, aa.T) - np.dot(aa, sites1v.T)
dd2x = ddx * ddx
ddy = np.dot(sites2v, aa.T) - np.dot(aa, sites2v.T)
dd2y = ddy * ddy
ddz = np.dot(sites3v, aa.T) - np.dot(aa, sites3v.T)
dd2z = ddz * ddz
H = np.sqrt(dd2x + dd2y + dd2z)

plotf(H, "distance matrix")

# compute covariance
phiM = .8
Sigma = (1 + phiM * H) * np.exp(-phiM * H)

plotf(Sigma, "matern cov")

# compute cholesky
L = np.linalg.cholesky(Sigma)

# sample random part
x = np.dot(L.T, np.random.randn(n).reshape(-1, 1))
prior = np.ones([n, 1]) * m
# prior = true_field.reshape(-1, 1)
x = x + prior
# xm = x.reshape(n1, n2, n3)

plotf3d(x, sites1v, sites2v, sites3v)


#%%

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
