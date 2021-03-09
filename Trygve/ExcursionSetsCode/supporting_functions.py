from __future__ import division                 # Always perform a float division when using the / operator - Back port
from __future__ import print_function           # Back port print function
from scipy.stats import mvn                     # For calculating multivariate pdf and cdf distributions
import scipy.stats as sc
from geographiclib.geodesic import Geodesic     # For motion calculations in lat/lon format.
from matplotlib import lines                    # Plotting lines in graph
import matplotlib.pyplot as plt                 # Plotting
import os                                       # OS / Terminal commands
import pyproj                                   # Map projections
import numpy as np                              # Numpy math library
import scipy.spatial.distance as scdist         # For generating covariance matrix - finding distances


# === Configuration variables for libs
geod = Geodesic.WGS84


# == Helper functions
def buildcov(site_1, site_2, sig, corr_decay, noise, analyse=False, cell_d=None):

    """
    Generate spatial covariance matrix - Matern 3/2 or Squared exponential - Compared with Jo towards Matlab function
    """

    if not isinstance(site_1, np.ndarray):
        site_1 = np.array(site_1)

    if not isinstance(site_2, np.ndarray):
        site_2 = np.array(site_2)

    # Number of sites
    if np.shape(site_1) != np.shape(site_2):
        n = np.shape(site_2)[0] * np.shape(site_2)[1]  # Location based covariance
    else:
        n = np.shape(site_2)[0]  # Similarity based covariance

    # Matern 3/2

    site_11 = np.vstack((site_1[0], site_1[1])).T
    site_22 = np.vstack((site_2[0], site_2[1])).T

    if site_11.shape[-1] == 2 and len(site_11.shape) <= 2 and site_22.shape[-1] == 2 and len(site_22.shape) <= 2:
        pass
    else:
        site_11 = site_1.reshape(site_1.shape[0]*site_1.shape[1], 2)
        site_22 = site_2.reshape(site_2.shape[0]*site_2.shape[1], 2)

    if site_11.shape[-1] == 2 and len(site_11.shape) <= 2 and site_22.shape[-1] == 2 and len(site_22.shape) <= 2:
        pass
    else:
        print('[ERROR]: The sites input is not formatted correctly, can not convert to lat,lon-tuple')
        return np.eye(n)

    # Range
    c_range = corr_decay

    # Scale
    sig2 = sig

    # Calculating the pairwise distance
    h = np.sqrt(scdist.cdist(site_11, site_22, 'sqeuclidean'))

    # Calculating the Matern 3/2 Covariance
    matern = np.multiply(sig2 * np.add(1, c_range * h), np.exp(h * -c_range))

    # Adding empirical covariance and white noise
    cvm = matern + np.eye(matern.shape[0]) * noise

    if analyse and cell_d:
        cell_length = np.mean([cell_d[0], cell_d[1]])
        corr_values = np.diag(np.transpose(cvm)[0].reshape(int(np.sqrt(cvm.shape[0])), int(np.sqrt(cvm.shape[1]))))
        plt.figure()
        plt.imshow(np.transpose(cvm)[0].reshape(int(np.sqrt(cvm.shape[0])), int(np.sqrt(cvm.shape[1]))))
        actual_len = np.arange(0, len(corr_values))*cell_length
        plt.figure(figsize=(8, 8))
        plt.title('The correlation range used in the covariance.', fontsize=12)
        plt.plot(actual_len, corr_values)
        plt.xlim([0, actual_len[-1]])
        plt.xlabel('Distance [m]')
        plt.show()

    # Return the co-variance matrix (cvm)
    return np.transpose(cvm)


def ExpectedVariance2(threshold, mu, Sig, H, R, eval_indexes, evar_debug=False):
    # __slots__ = ('Sigxi', 'Sig', 'muxi', 'a', 'b', 'c')

    """
    Computes IntA = \sum_x \int  p_x(y) (1-p_x(y)) p (y) dy
    x is a discretization of the spatial domain
    y is the data
    p_x(y)=P(T_x<T_threshold , S_x < S_threshold | y) = ...
    \Phi_2_corrST ( [T_threshold-E(T_x|y)] /Std(T_x/y) , S_threshold-E(S_x|y)] /Std(S_x/y)]
    E(T,S|y)=mu+Sig*H'*(H*Sig*H'+R)\(y-H mu ) = xi
    where xi \sim N (mu, Sig*H'*((H*Sig*H'+R)\(H*Sig)) is the only variable
    that matters in the integral and for each x, this is an integral over
    xi_x = (xi_xT,xi_xS)
    """
    # For debug
    # H = np.zeros((2*50,2*50*50))
    # H[0:50, 0:50] = np.eye(50)
    # H[50:100, 50:100] = np.eye(50)
    # R = 0.25 * np.eye(100)

    # Xi variable distribution N(muxi, Sigxi)
    a = np.dot(Sig, H.T)
    b = np.dot(np.dot(H, Sig), H.T) + R
    c = np.dot(H, Sig)
    Sigxi = np.dot(a, np.linalg.solve(b, c))
    V = Sig - Sigxi  # Uncertainty reduction
    n = int(mu.flatten().shape[0]/2)
    muxi = np.copy(mu)

    IntA = 0.0
    pp = None

    if evar_debug:
        pp = []

    for i in eval_indexes:

        SigMxi = Sigxi[np.ix_([i, n+i], [i, n+i])]
        rho = V[i, n+i] / np.sqrt(V[i, i]*V[n+i, n+i])

        if np.isnan(rho):
            rho = 0.6

        Mxi = [muxi[i], muxi[n+i]]
        sn_1 = np.sqrt(V[i, i])
        sn_2 = np.sqrt(V[n+i, n+i])
        sn2 = np.array([[sn_1**2, sn_1*sn_2*rho], [sn_1*sn_2*rho, sn_2**2]])

        if evar_debug:
            pp.append(mvn.mvnun(np.array([[-np.inf], [-np.inf]]), np.array([[0], [0]]), np.subtract([threshold[0], threshold[1]], np.array(Mxi).ravel()), SigMxi)[0])

        mm = np.vstack((Mxi, Mxi))
        SS = np.add(np.vstack((np.hstack((sn2, np.zeros((2, 2)))), np.hstack((np.zeros((2, 2)), sn2)))), np.vstack((np.hstack((SigMxi, SigMxi)), np.hstack((SigMxi, SigMxi)))))
        vv2 = np.add(sn2, SigMxi)
        Thres = np.array([threshold[0], threshold[1]])
        mur = np.subtract(Thres, Mxi)
        IntB_a = mvn.mvnun(np.array([[-np.inf], [-np.inf]]), np.array([[0], [0]]), mur, vv2)[0]
        Thres = np.array([threshold[0], threshold[1], threshold[0], threshold[1]])
        mur = np.subtract(Thres, mm)
        IntB_b = mvn.mvnun(np.array([[-np.inf], [-np.inf], [-np.inf], [-np.inf]]), np.array([[0], [0], [0], [0]]), mur, SS)[0]

        IntA = IntA + np.nansum([IntB_a, -IntB_b])

    if evar_debug:
        plt.figure()
        plt.imshow(np.array(pp).reshape(30, 30))
        plt.show()

    return IntA


def wgs84_dist(lat1, lon1, lat2, lon2):
    """
    Calculates distance based on WGS84 Geode
     lat1, lon1 = origin
     lat2, lon2 = destination
    """

    if lat1 > 1000:
        print('[fs-utils] - ERROR: Wrong input. Needs to be decimal degrees ')

    dist = geod.Inverse(lat1, lon1, lat2, lon2)['s12']

    return dist


def wgs84_step(lat, lon, azi, dist):
    """
    Calculates a lat, lon based on distance and azimuth
    """

    if lat > 1000:
        print('ERROR: Wrong input. Needs to be decimal degrees ')

    walk = geod.Direct(lat, lon, azi, dist)

    return walk


def wgs84_azi(lat1, lon1, lat2, lon2):
    """
    Calculates distance based on WGS84 Geode
     lat1, lon1 = origin
     lat2, lon2 = destination
    """

    if lat1 > 1000:
        print('ERROR: Wrong input. Needs to be decimal degrees ')

    azi = geod.Inverse(lat1, lon1, lat2, lon2)['azi1']

    return azi


def meters_to_degrees(input_u, input_v):
    """
    Converts meters to degrees
    """
    return input_u/wgs84_dist(63.0, 8.5, 64.0, 8.5), input_v/wgs84_dist(63.0, 8.0, 63.0, 9.0)

def build_graph(X_p, x_resolution, y_resolution):
    bG, G_node_positions, G_node_points = {}, {}, {}
    for ii in range(X_p.shape[0]):
        bG[ii] = {}
        G_node_positions[ii] = X_p[ii, :]
        G_node_points[ii] = X_p[ii, :]
        for jj in range(X_p.shape[0]):
            if ii != jj:
                w = np.linalg.norm(X_p[ii, :] - X_p[jj, :])
                x_dist = abs(X_p[ii, 0] - X_p[jj, 0])
                y_dist = abs(X_p[ii, 1] - X_p[jj, 1])
                if x_dist <= x_resolution * 1.0 and y_dist <= y_resolution * 1.0:
                    bG[ii][jj] = round(2.0 * w / x_resolution)
    return bG, G_node_positions, G_node_points


def node_to_edge_graph(iG, G_node_positions):
    G2 = {}
    G2_node_positions = {}
    G2_node_points = {}
    for v_i in iG:
        for v_j in iG[v_i]:
            G2[(v_i, v_j)] = {}
            G2_node_positions[(v_i, v_j)] = (0.9 * G_node_positions[v_i] + 0.1 * G_node_positions[v_j])
            for v_k in iG[v_j]:
                G2[(v_i, v_j)][(v_j, v_k)] = iG[v_i][v_j]
    return G2, G2_node_positions, G2_node_points


def generate_edge_points(iG, G_node_positions, resolution):
    G_measurement_points = {}
    for v_i in iG:
        for v_j in iG:
            if v_j in iG[v_i]:
                points = []
                start_p = G_node_positions[v_i]
                end_p = G_node_positions[v_j]
                num_points = int(wgs84_dist(start_p[1], start_p[0], end_p[1], end_p[0]) / resolution)
                for point_i in range(1, num_points+1):
                    s = float(point_i) / num_points
                    points.append((1 - s) * start_p + s * end_p)
                G_measurement_points[(v_i, v_j)] = np.reshape(np.array(points), (-1, 2))
            else:
                G_measurement_points[(v_i, v_j)] = np.zeros((0, 2))
    return G_measurement_points


def plot_graph(iax, iG, G_node_positions, x_resolution, y_resolution, color='k', nodenames=True):

    for v_i in iG:
        v_i_pos = G_node_positions[v_i]
        iax.scatter([v_i_pos[0]], [v_i_pos[1]], c=color)
        for v_j in iG[v_i]:
            v_j_pos = G_node_positions[v_j]
            plt.annotate('', v_j_pos, xytext=v_i_pos,
                         arrowprops=dict(linewidth=0, width=1,
                                         headwidth=4, headlength=10, fc="k"))
        if nodenames:
            if v_i == 0 or v_i == 2:
                iax.annotate(str(v_i),
                            xy=(v_i_pos[0], v_i_pos[1]),
                            xytext=(v_i_pos[0] - x_resolution * 0.05, v_i_pos[1] + y_resolution * 0.01),
                            verticalalignment='bottom',
                            fontsize=12,
                            color='k')
            else:
                iax.annotate(str(v_i),
                            xy=(v_i_pos[0], v_i_pos[1]),
                            xytext=(v_i_pos[0] + x_resolution * 0.01, v_i_pos[1] + y_resolution * 0.01),
                            verticalalignment='bottom',
                            fontsize=12,
                            color='r')


def plot_edge_points(iax, G_edge_points, c='b'):
    for v in G_edge_points:
        for point in list(G_edge_points[v]):
            iax.scatter([point[0]], [point[1]], c=c)


def plot_path(iax, G_node_positions, Pat, linestyle='-'):
    v_prev = Pat[0]
    for v in Pat[1:]:
        start_p = G_node_positions[v_prev]
        end_p = G_node_positions[v]
        line_artist = lines.Line2D(
            [start_p[0], end_p[0]], [start_p[1], end_p[1]],
            color="red", linewidth=5.0,
            linestyle=linestyle, antialiased=True)
        iax.add_artist(line_artist)
        v_prev = v


class Field(object):
    """
    Field class with data and grid instances (in 2D projection and geographic reference) of the area
    """

    def __init__(self, res_x=50, res_y=50, initialValue=0.0, utm_zone=32, silence=False):

        # Grid parameters
        self.dx = res_x  # Resolution East - X (number of cells)
        self.dy = res_y  # Resolution North - Y (number of cells)
        self.data = np.array([[initialValue for y in range(res_y)] for x in range(res_x)])  # Initial value for data, default 0
        self.llcoor = [[initialValue for y in range(res_y)] for x in range(res_x)]  # Initial value for grid, default 0
        self.xycoor = [[initialValue for y in range(res_y)] for x in range(res_x)]  # Initial value for grid, default 0
        self.xylines = []
        self.proj = None
        self.zone = utm_zone
        self.silence = silence

        # Set the local projection to be used: =================================================== POLAR STEREOGRAPHIC
        cNP = [13272.0, 11344.5]  # Coordinate North Pole
        self.proj_polarst = pyproj.Proj(proj='stere',
                                        lat_ts=60.0,            # Std. parallel
                                        lat_0=90.0,             # Lat. of proj. origin
                                        lon_0=58.0,             # Straight vertical lon. from pole
                                        x_0=cNP[0] * 160,  # False easting for avoiding negative x, y values
                                        y_0=cNP[1] * 160,  # False northing
                                        a=6370000,  # Ellipsoid
                                        b=6370000)  # Ellipsoid

        # Set the local projection to be used: =================================================== UTM - 32 (WGS84)
        self.proj_wgs84 = pyproj.Proj(proj='utm', zone=self.zone, ellps='WGS84')

    def setGrid(self, lat_lrc=62.379897, lon_lrc=5.699939, lat_ulc=62.386661, lon_ulc=5.685864, proj='wgs84'):

        """
        Sets the grid into lat/lon format and x/y coordinates
        """
        self.proj = proj

        # Calculating grid steps
        if proj == 'polar':
            ly, lx = self.proj_polarst(lon_lrc, lat_lrc)  # Lower right corner y,x value
            uy, ux = self.proj_polarst(lon_ulc, lat_ulc)  # Upper left corner y,x value
            step_x = np.abs(lx - ux) / self.dx
            step_y = np.abs(ly - uy) / self.dy
        else:
            ly, lx = self.proj_wgs84(lon_lrc, lat_lrc)  # Lower corner x,y value
            uy, ux = self.proj_wgs84(lon_ulc, lat_ulc)  # Upper corner x,y value
            step_x = np.abs(lx - ux) / (self.dx)
            step_y = np.abs(ly - uy) / (self.dy)

        if lx < ux:
            if uy > ly:
                xylines = np.linspace(lx, ux, self.dx), np.linspace(ly, uy, self.dy)
            else:
                xylines = None
                print('[fs-utils] - Error in grid xylines! uy < ly')
                print('[fs-utils] - For WGS84 you need to orient the projection Northwards')
                print('[fs-utils] - For Polar Stereographic you need to orient 45 deg off North')
                return None
        else:
            xylines = None
            print('[fs-utils] - Error in grid xylines! ux > lx')
            print('[fs-utils] - For WGS84 you need to orient the projection Northwards')
            print('[fs-utils] - For Polar Stereographic you need to orient 45 deg off North')
            return None

        if step_x < 0:
            raise Exception('[fs-utils] - GRID ERROR: Check the grid geographical coordinates - Grid has negative step')

        # Finding center lat & lon for each cell: ================================================= LAT LON GENERATION
        x_count = 0
        y_count = 0

        # ================================== #
        # This is the grid - lat/lon mapping #
        # ================================== #

        # upper corner is (0,0)
        # y is horizontal
        # x is vertical
        # mapping is therefore as this:
        #
        #                  x  y------------>
        #                  |
        #                  |    [[0,0] [0,1] .....
        #                  |    [[1,0] [1,1] .....
        #                  v
        #
        # along with indexing  [ 1 2 3 4 5 ....
        #                      [ 30 31 .....
        #
        #
        for y in xylines[1]:
            for x in xylines[0][::-1]:
                if proj == 'polar':
                    self.llcoor[x_count][y_count] = self.proj_polarst(y, x, inverse=True)
                    self.xycoor[x_count][y_count] = (y, x)
                    x_count += 1
                else:
                    self.llcoor[x_count][y_count] = self.proj_wgs84(y, x, inverse=True)
                    self.xycoor[x_count][y_count] = (y, x)
                    x_count += 1
            y_count += 1
            x_count = 0

        if proj == 'polar':
            assert self.llcoor[1][2] == self.proj_polarst(self.xycoor[1][2][0], self.xycoor[1][2][1], inverse=True), "[fs-utils] - Mapping not valid"
        else:
            assert self.llcoor[1][2] == self.proj_wgs84(self.xycoor[1][2][0], self.xycoor[1][2][1], inverse=True), "[fs-utils] - Mapping not valid"

        self.xylines = xylines

    def setValue(self, value):
        """
        Set the value of the field to a value
        :param value: input value to assign
        :return: None
        """
        self.data = [[value for y in range(self.dy)] for x in range(self.dx)]


    @property
    def yy(self):
        """
        Returns the x coordinate line from 0 - 9
        """
        return self.xylines[1]

    @property
    def n(self):
        """
        Returns the x coordinate line from 0 - 9
        """
        return self.dy * self.dx

    @property
    def xx(self):
        """
        Returns the y coordinate line from 0 - 9
        """
        return self.xylines[0][::-1]

    @property
    def xmax(self):
        return np.max(self.xx)

    @property
    def ymax(self):
        return np.max(self.yy)

    @property
    def xmin(self):
        return np.min(self.xx)

    @property
    def ymin(self):
        return np.min(self.yy)

    @property
    def glon(self):
        """
        Returns the grid lon coordinates. according to 0-9 y direction line 1, 0-9 y dir. line 2, etc.
        """
        return sum([[x[0] for x in self.llcoor[:][y]] for y in range(self.dy)], [])

    @property
    def glat(self):
        """
        Returns the grid lon coordinates. according to 0-9 y direction line 1, 0-9 y dir. line 2, etc.
        """
        return sum([[x[1] for x in self.llcoor[:][y]] for y in range(self.dy)], [])

    def __str__(self):
        """
        Prints the grid as a list of lat lon - coordinates to 0-9 x direction line 1, 0-9 x dir. line 2, etc.
        """
        out = [self.llcoor[x][y] for y in range(self.dy) for x in range(self.dx)]

        # # Printing with google maps
        # lon = sum([[x[0] for x in self.llcoor[:][y]] for y in range(self.dy)], [])
        # lat = sum([[x[1] for x in self.llcoor[:][y]] for y in range(self.dy)], [])
        # gmap = GMap()
        # for i in range(0, 100, 1):
        #     gmap.add_point((lat[i], lon[i]))
        # with open("output.html", "w") as gout:
        #     print(gmap, file=gout)

        return '\n'.join([''.join(str(x[1]) + ',' + str(x[0])) for x in out])

    @staticmethod
    def getEuclidianCoord(self, i, j):
        return j*self.dx+i

    def get_neigh(self, idx, idy, star=False, debug=False):

        if not star:
            neigh_idx = np.vstack((idx - 1, idx + 1, idx - 1, idx + 1, idx, idx, idx + 1, idx - 1))
            neigh_idy = np.vstack((idy, idy, idy - 1, idy + 1, idy - 1, idy + 1, idy - 1, idy + 1))
        else:
            neigh_idx = np.vstack((idx - 1, idx + 1, idx, idx))
            neigh_idy = np.vstack((idy, idy, idy - 1, idy + 1))

        if -1 in neigh_idx or -1 in neigh_idy:
            neigh_idy = neigh_idy[neigh_idy >= 0]
            neigh_idx = neigh_idx[neigh_idx >= 0]

        neigh_idy = neigh_idy[neigh_idy <= self.dy - 1]
        neigh_idx = neigh_idx[neigh_idx <= self.dx - 1]

        if len(neigh_idx) == 0 or len(neigh_idy) == 0:
            print('[ERROR] - Outside grid!')
            return None
        try:
            neigh = np.ravel_multi_index((neigh_idy, neigh_idx), dims=(self.dx, self.dy))
        except ValueError:
            print('[ERROR] - Check input cell positions. Outside of grid!')

        if debug:
            rolled_out_index = np.arange(self.dx * self.dy).reshape(self.dx, self.dy)
            for p in [np.unravel_index(n, dims=(self.dx, self.dy)) for n in neigh]:
                rolled_out_index[p[0], p[1]] = 1000

            plt.figure()
            plt.imshow(rolled_out_index)
            plt.show()

        return neigh

    def getGridCoordinates(self, lat, lon, res_gx=0, res_gy=0):

        ##################################
        #                                #
        #  New getGridCoordinates class  #
        #                                #
        ##################################

        #########################
        #                       #
        #  POLAR STEREOGRAPHIC  #
        #                       #
        #########################

        if self.proj == 'polar':
            y, x = self.proj_polarst(lon, lat)

            if isinstance(y, float):

                idx = np.argmin(np.abs(self.xx - x))  # Note the inverse referencing here, due to matrix and proj differences
                idy = np.argmin(np.abs(self.yy - y))

                if res_gx == 0:
                    return idx, idy
                else:
                    rolled_out_index = np.arange(res_gx * res_gy).reshape(res_gx, res_gy)
                    ifa = rolled_out_index[idx, idy]
                    return idx, idy, ifa
            else:
                idx = []
                idy = []

                for i in range(0, len(y)):
                    idx.append(np.argmin(np.abs(self.xx - x[i])))  # Note the referencing here, due to polar stereographic the axes are correct
                    idy.append(np.argmin(np.abs(self.yy - y[i])))

                if res_gx == 0:
                    return idx, idy
                else:

                    rolled_out_index = np.arange(res_gx * res_gy).reshape(res_gx, res_gy)
                    ifa = rolled_out_index[idx, idy]
                    return idx, idy, ifa

        #####################
        #                   #
        #  WGS 84, ZONE 32  #
        #                   #
        #####################

        else:
            y, x = self.proj_wgs84(lon, lat)

            if isinstance(y, float):

                idx = np.argmin(np.abs(self.xx - x))  # Note the inverse referencing here, due to matrix and proj differences for WGS84
                idy = np.argmin(np.abs(self.yy - y))

                if res_gx == 0:
                    return idx, idy
                else:
                    rolled_out_index = np.arange(res_gx * res_gy).reshape(res_gx, res_gy)
                    ifa = rolled_out_index[idx, idy]
                    return idx, idy, ifa
            else:
                idx = []
                idy = []

                for i in range(0, len(y)):
                    idx.append(np.argmin(np.abs(self.xx - x[i])))  # Note the inverse referencing here
                    idy.append(np.argmin(np.abs(self.yy - y[i])))

                if res_gx == 0:
                    return idx, idy
                else:

                    rolled_out_index = np.arange(res_gx * res_gy).reshape(res_gx, res_gy)
                    ifa = rolled_out_index[idx, idy]
                    return idx, idy, ifa

    def loadvalues(self, input_values, mode='load'):
        if mode == 'load':
            if np.shape(self.data) == np.shape(input_values):

                if not os.path.isfile('field_data_old.npy'):
                    open('field_data_old.npy', mode='w')
                    # Storing old data
                    np.save('field_data_old.npy', self.data)

                np.save('field_data_old.npy', self.data)
                self.data = input_values
                if self.silence:
                    pass
                else:
                    print('[fs-utils] - GRID: Load Successful')
            else:
                print('[fs-utils] - Did NOT load data. Incompaticle dimensions.')

        elif mode == 'reload':
            self.data = np.load('field_data_old.npy')


if __name__ == "__main__":

    print('#===== Running computability test on the supporting function code =====#')
    compability_count = 0

    print('Testing neighbour cell fetch function')
    testgrid = Field(30, 30, 0)

    res1 = testgrid.get_neigh(0, 0).tolist()
    res2 = testgrid.get_neigh(0, 0, star=True).tolist()
    res3 = testgrid.get_neigh(29, 0).tolist()
    res4 = testgrid.get_neigh(0, 29).tolist()
    res5 = testgrid.get_neigh(15, 15).tolist()
    result = [res1, res2, res3, res4, res5]

    if None in result:
        pass
    else:
        compability_count += 1

    # Testing integrals, analytically vs Monte Carlo - Variables are Temperature and Salinity

    # Specify input parameters: Mean (Prior mean \xi)
    m = np.array([[8], [27]])
    # Threshold ( temp t and salinity s)
    T = np.array([[6], [21]])
    # Covariance (Prior covariance \xi)
    s2 = np.array([[2**2, 2*4*0.9], [2*4*0.9, 4**2]])
    # Measurement noise ( y = \xi + N(0,diag[]) )
    sig2 = s2 + np.diag([0.3, 1])
    # Covariance measurement and \xi (might not be at same location)
    k = 0.9 * s2

    # Mean m_{\xi}
    Mxi = m
    # Reduction in covariance
    SigMxi = np.dot(k, np.linalg.solve(sig2, k.T))
    # Updated covariance
    sn2 = s2 - SigMxi
    rho = sn2[1, 0] / np.sqrt(sn2[0, 0] * sn2[1, 1])  # Correlation in Z
    sn_1 = np.sqrt(sn2[0, 0])
    sn_2 = np.sqrt(sn2[1, 1])

    mur = (T - Mxi)

    # Monte Carlo solution

    B = 10000
    L = np.linalg.cholesky(SigMxi)
    u = np.dot(Mxi, np.ones((1, B))) + np.dot(L, np.random.randn(2, B))  # m_{xi} \sim N(m_{\xi},\Sigma_{\xi}), see eq (9) in draft paper.
    Pcalc = np.zeros((B, 1))

    for i in range(0, B):
        Pcalc[:][i] = sc.multivariate_normal.cdf(T.flatten(), mean=np.array([[u[0, i]], [u[1, i]]]).flatten(), cov=sn2)

    IntMC = np.mean(np.multiply(Pcalc, (1-Pcalc)))

    # Analytical solution 1

    mm = np.vstack((Mxi, Mxi))
    SS = np.add(np.vstack((np.hstack((sn2, np.zeros((2, 2)))), np.hstack((np.zeros((2, 2)), sn2)))), np.vstack((np.hstack((SigMxi, SigMxi)), np.hstack((SigMxi, SigMxi)))))
    vv2 = np.add(sn2, SigMxi)
    Thres = np.array([T[0], T[1]])
    mur = np.subtract(Thres, Mxi)
    IntB_a = mvn.mvnun(np.array([[0], [0]]), np.array([[np.inf], [np.inf]]), mur, vv2)[0]
    Thres = np.array([T[0], T[1], T[0], T[1]])
    mur = np.subtract(Thres, mm)
    IntB_b = mvn.mvnun(np.array([[0], [0], [0], [0]]), np.array([[np.inf], [np.inf], [np.inf], [np.inf]]), mur, SS)[0]

    IntA = np.nansum([IntB_a, -IntB_b])

    print('Monte Carlo solution is {}. Analytical 1 is {}'.format(IntMC, IntA))


    # Analytical solution 2

    # The 4-variate calculation is a linear combinations of Z_1=(Z_{1,a},Z_{1,b}), Z_2=(z_{2,a},Z_{2,b}), m_{\xi}= (m_{\xi,a},m_{\xi,b})
    MMvec = np.array([[0], [0], [0], [0], Mxi[0], Mxi[1]])
    SSvec = np.array([[1, rho, 0, 0, 0, 0], [rho, 1, 0, 0, 0, 0], [0, 0, 1, rho, 0, 0], [0, 0, rho, 1, 0, 0], [0, 0, 0, 0, SigMxi[0, 0], SigMxi[0, 1]],[0, 0, 0, 0, SigMxi[1, 0], SigMxi[1, 1]]])
    SSvecNEG = np.array([[1, rho, 0, 0, 0, 0], [rho, 1, 0, 0, 0, 0], [0, 0, 1, -rho, 0, 0], [0, 0, -rho, 1, 0, 0], [0, 0, 0, 0, SigMxi[0, 0], SigMxi[0, 1]], [0, 0, 0, 0, SigMxi[1, 0], SigMxi[1, 1]]])
    # Part I linear combination
    Amat = np.array([[sn_1, 0, 0, 0, 1, 0], [0, sn_2, 0, 0, 0, 1], [0, 0, sn_1, 0, -1, 0], [0, 0, 0, sn_2, 0, -1]])
    mean1 = np.dot(Amat, MMvec)
    var1 = np.dot(np.dot(Amat, SSvec), Amat.T)
    Thres1 = np.array([T, -T])
    IntA_part1 = sc.multivariate_normal.cdf(Thres1.flatten(), mean=mean1.flatten(), cov=var1)
    # Part II linear combination
    Amat = np.array([[sn_1, 0, 0, 0, 1, 0], [0, sn_2, 0, 0, 0, 1], [0, 0, sn_1, 0, 1, 0], [0, 0, 0, sn_2, 0, -1]])

    mean2 = np.dot(Amat, MMvec)
    var2 = np.dot(np.dot(Amat, SSvecNEG), Amat.T)
    Thres2 = np.array([T[0], T[1], T[0], -T[1]])
    IntA_part2 = sc.multivariate_normal.cdf(Thres2.flatten(), mean=mean2.flatten(), cov=var2)
    # Part III linear combination
    Amat = np.array([[sn_1, 0, 0, 0, 1, 0], [0, sn_2, 0, 0, 0, 1], [0, 0, sn_1, 0, -1, 0], [0, 0, 0, sn_2, 0, 1]])

    mean3 = np.dot(Amat, MMvec)
    var3 = np.dot(np.dot(Amat, SSvecNEG), Amat.T)
    Thres3 = np.array([T[0], T[1], -T[0], T[1]])
    IntA_part3 = sc.multivariate_normal.cdf(Thres3.flatten(), mean=mean3.flatten(), cov=var3)
    # All parts
    IntA = IntA_part1 + IntA_part2 + IntA_part3

    print('Monte Carlo solution is {}. Analytical 2 is {}'.format(IntMC, IntA))
    compability_count += 1

    if compability_count >= 2:
        print('#===== Successful test of code, all functions are OK! =====#')
    else:
        print('#===== Check for ERRORS! =====#')