# print fjord data
import netCDF4
import os
from usr_defined_func import *

top_path = os.getcwd()
data_path = '/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Projects_practice/Gradient_ES'
os.chdir(data_path)

fp = 'samples_sub.nc'
nc = netCDF4.Dataset(fp)

# masked numpy arrays
# nc['...'][timestep, z , y, x]
salinity = nc['salinity'][:,:,:,:]
temperature = nc['temperature'][:,:,:,:]
x_current = nc['u_velocity'][:,:,:,:]
y_current = nc['v_velocity'][:,:,:,:]
z_current = nc['w_velocity'][:,:,:,:]
# the grid
x,y,z = np.meshgrid(nc["xc"][:], nc["yc"][:],nc["zc"][:])
# timestamps = [datetime(*x) for x in nc['time'][:,:]]

#%% only to save time
ind_t = 170

# d = np.arange(0, 2)
# xs = np.arange(100, 120)
# ys = np.arange(40, 60)
# x = x_current[ind_t, 0:2, 100:120, 40:60]
# y = y_current[ind_t, 0:2, 100:120, 40:60]
# z = z_current[ind_t, 0:2, 100:120, 40:60]
sal = salinity[ind_t, 0:3, 90:120, 40:60]
# temp = temperature[ind_t, :, :, :]
# sal = salinity[ind_t, 0:5, 80:120, 40:60]
# temp = temperature[ind_t, 0:5, 80:120, 40:60]


# xt = np.ma.filled(x, np.amin(x))
# yt = np.ma.filled(y, np.amin(y))
# zt = np.ma.filled(z, np.amin(z))
# sal_t = np.ma.filled(sal, np.amin(sal))
# temp_t = np.ma.filled(temp, np.amin(temp))
sal_t = np.ma.filled(sal, np.min(sal))
sal_t = (sal_t - np.mean(sal_t)) / np.std(sal_t)
# temp_t = np.ma.filled(temp, 0)

nx = sal_t.shape[1]
ny = sal_t.shape[2]
nz = sal_t.shape[0]
# x = np.arange(0, nx)
# y = np.arange(0, ny)
# z = np.arange(0, nz)

# xt = x_current[ind_t, 0:5, 100:120, 40:60]
# yt = y_current[ind_t, 0:5, 100:120, 40:60]
# zt = z_current[ind_t, 0:5, 100:120, 40:60]
# sal_t = salinity[ind_t, 0:5, 100:120, 40:60]
# temp_t = temperature[ind_t, 0:5, 100:120, 40:60]

zz, yy, xx = np.mgrid[1:nz + 1, 1:ny + 1, 1:nx + 1]

xv = xx.flatten()
yv = yy.flatten()
zv = zz.flatten()


sal_tv = sal_t.flatten()
# temp_tv = temp_t.flatten()

plotf3d(sal_tv, xv, yv, zv)








#%%
## used to change the folder back again
os.chdir(top_path)


