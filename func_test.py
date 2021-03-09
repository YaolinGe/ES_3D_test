from __future__ import print_function
from __future__ import division
import sys
import pickle
from es_strategies import *
from es_cases import *
import seaborn as sns

# == Plotting
# import matplotlib
# matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['ps.fonttype'] = 42
from matplotlib import pyplot as plt
plt.ioff()  # Running plt.ioff() - plots are kept in background, plt.ion() plots are shown as they are generated
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 27})
plt.rcParams.update({'font.style': 'oblique'})

my_path = os.path.dirname(os.path.abspath(__file__))


# Simulation study


# == Script mode (sim or view old results)
sim = True

# == Timer
t_init = time.time()

# ===================================== #
# ========= SCRIPT PARAMETERS ========= #
# ===================================== #

# ==== Decimal degree resolution and max, min, these will decide how many nodes there will be
grid_cell_dimension = (87, 87)

# To make a square area with correct coordinates - a projection down to utm has to be made
proj_wgs84 = pyproj.Proj(proj='utm', zone=32, ellps='WGS84')

# Lat / Lon resolution
x_resolution = 0.025 / (2.5 * 1.5)  # Legacy = 0.05
y_resolution = 0.005 / (1.0 * 1.5)  # Legacy = 0.02

# ==== Set the maximum and minimum values for the graph - Not fixed for east/north max! These will change acc. to mgrid!
grid_east_min, grid_east_max = 8.54927, 8.60618
grid_east_diff = grid_east_max - grid_east_min

grid_north_min, grid_north_max = 63.98, 64.005
grid_north_diff = grid_north_max - grid_north_min

dist_east = wgs84_dist(grid_north_min, grid_east_min, grid_north_min, grid_east_max)
dist_north = wgs84_dist(grid_north_min, grid_east_min, grid_north_max, grid_east_min)

# ==== The distance between points to sample on the grid graph
sample_distance = 80

# ==== Grid resolution in x direction & Grid resolution in y direction
nx = 31
ny = 31

# ==== Source location (river)
sitesSOURCE = [[31], [31]] #


# ==== Excursion set threshold
Tthreshold = 7.16
Sthreshold = 31.208
Th = [Tthreshold, Sthreshold]

print("hello world")

# ==== Trend parameters
beta1 = [[5.8], [0.098]]  # Start_temp, temperature_trend_factor, increase to west
beta2 = [[29.0], [0.133]]  # Start_sal, salinity_trend_factor

# ==== Correlation T-S
sig_st = 0.6

# ==== Noise
nugg = 0.05
noise = np.array([[nugg ** 2, 0], [0, nugg ** 2]])

# ===== Spatial correlation
# sig = 1
# corr_range = 0.9  # 0.9 ~ 600m
# noise = 0.005**2
sig = 0.25 # std in temp and sal
corr_range = 0.3  # 0.8 ~ 900m

# ==== Sampling strategies
strategies = ['naive', 'myopic', 'look-ahead', 'static_north', 'static_east', 'static_zigzag']
#strategies = ['static_north']
static_north_nodes_0 = [54, 55, 56, 57, 58, 59, 60, 61, 62, 61]
static_east_nodes_0 = [16, 27, 38, 49, 60, 71, 82, 93, 104, 115]
static_zigzag_nodes_0 = [15, 26, 36, 47, 57, 68, 78, 89, 99, 110]
strategies_colors = ['r', 'b', 'k', 'g', 'y', 'm', 'c']

# ==== Simulation steps
timesteps = 10
replicates = 1  # Monte Carlo simulation steps
plot_excursion = True
plot_excursion_prob = True
plot_covariance = False
plot_priors = True
plot_wpgraph = True
plot_evolution = True

# ==== Start point
curr_point0, curr_point = 53, 53

# ==== The AUV route
route = []

# ==== Toggle to generate plots
plotting = True
fig_height = 7.0
fig_width = 8.5
t_min = 5.8
t_max = 8.435
s_min = 29.0
s_max = 33.185
p_min = 0
p_max = 0.2525

if sim:

    print("Simulation is true, coming to the loop")
    # ================================================== #
    # ========== GRAPH AND WAYPOINT GENERATION ========= #
    # ================================================== #

    ly, lx = proj_wgs84(grid_east_min, grid_north_min)  # Lower corner x,y value
    uy, ux = proj_wgs84(grid_east_max, grid_north_max)  # Upper corner x,y value
    graph_spatial_points = []
    nrows, ncols = 11, 11
    row_spacing = np.abs(lx - ux) / 11
    col_spacing = row_spacing * 0.5 * np.sqrt(3)  # Altitude for an equilateral triangle

    for col_i in range(ncols):
        p1 = col_i * col_spacing + ly
        if col_i % 2 == 1:
            offset = 0.5 * row_spacing
            nrows_thiscol = nrows - 1
        else:
            offset = 0.0
            nrows_thiscol = nrows
        for row_i in range(nrows_thiscol):
            p2 = offset + row_spacing * row_i + lx
            graph_spatial_points.append((proj_wgs84(p1, p2, inverse=True)))

    graph_spatial_points = np.array(graph_spatial_points)
    east_min, east_max = min(graph_spatial_points[:, 0]), max(graph_spatial_points[:, 0])
    north_min, north_max = min(graph_spatial_points[:, 1]), max(graph_spatial_points[:, 1])

    dist_east1 = wgs84_dist(north_min, east_min, north_min, east_max)
    dist_nort1 = wgs84_dist(north_min, east_min, north_max, east_min)

    # ==== Build graph
    graph, graph_node_pos, graph_node_points = build_graph(graph_spatial_points, x_resolution, y_resolution)
    # with open('graph_node_pos.pickle'.format(simname), 'wb') as handle:  # Saving current simulation data (overwriting)
    #     pickle.dump(graph_node_pos, handle)
    graph_meas_nodes = generate_edge_points(graph, graph_node_pos, sample_distance)  # Legacy 360

    print('Travel graph generated.')

    # ==== Description
    # graph = the graph with the logical connection to the node numbers
    # graph_node_pos = contains the lat/lon of the nodes
    # graph_node_points = contains the lat/lon of the nodes

    # =============================================== #
    # ======= GENERATE GRID FOR GP AND COV  ========= #
    # =============================================== #

    # ==== Generate and establish grid to investigate empirical covariance
    grid = Field(nx, ny, 0)  # the grid class instance of the Field class - for holding the estimated data


    print('STOP here +++++++++++++++++++++++++++++++++++++++++')

    # == Set to the same extent as the graph! This way it is possible to relate the grid to the graph
    counter = 1
    nb_of_nodes_in_east = 0
    while graph_node_pos[counter][0] > graph_node_pos[counter - 1][0]:
        nb_of_nodes_in_east += 1
        counter += 1
        if counter == 500:
            print('Error in finding lower corner node, due to orientation. Is the area oriented towards north?')

    counter = nb_of_nodes_in_east + 1

    lat_lc = graph_node_pos[nb_of_nodes_in_east][1]
    lon_lc = graph_node_pos[nb_of_nodes_in_east][0]

    lat_uc = graph_node_pos[nb_of_nodes_in_east + 1][1]
    lon_uc = graph_node_pos[nb_of_nodes_in_east + 1][0]

    nb_of_nodes_in_north = nb_of_nodes_in_east + 1
    new_row = 1

    for i in range(0, 500):

        try:
            lat_uc = graph_node_pos[counter + nb_of_nodes_in_east + 1][0]
            lon_uc = graph_node_pos[counter + nb_of_nodes_in_east + 1][0]
        except KeyError:
            lat_uc = graph_node_pos[counter][1]
            lon_uc = graph_node_pos[counter][0]
            break

        counter += nb_of_nodes_in_east + 1

    # == Requesting the extent and generating the grid
    grid.setGrid(lat_lc, lon_lc, lat_uc, lon_uc, proj='wgs84')

    # == Extracting the lat/lon from the newly generated grid
    glat = grid.glat
    glon = grid.glon

    # == Finding the spatial extent of the new grid
    width = wgs84_dist(grid.llcoor[nx - 1][0][1], grid.llcoor[nx - 1][0][0], grid.llcoor[nx - 1][ny - 1][1], grid.llcoor[nx - 1][ny - 1][0]) * 0.002
    height = wgs84_dist(grid.llcoor[0][0][1], grid.llcoor[0][0][0], grid.llcoor[nx - 1][0][1], grid.llcoor[nx - 1][0][0]) * 0.002

    # == Collecting the lat/lon coordinates from the new grid class into a new variable "coord"
    coord = zip(glat, glon)  # Set the lat lon pairs into the coord variable (will be used in for-loop for retrieving data

    a, b = grid.getGridCoordinates(grid.llcoor[nx - 1][ny - 1][1], grid.llcoor[nx - 1][ny - 1][0])
    assert a == nx - 1, "ASSERT ERROR: getGridCoordinates reports wrong value"
    assert b == ny - 1, "ASSERT ERROR: getGridCoordinates reports wrong value"

    sgrid = Field(nx, ny, 0)
    sgrid.setGrid(lat_lc, lon_lc, lat_uc, lon_uc, proj='wgs84')

    # ==== Plotting the guide graph with the correct width and height
    plot_wpgraph = False
    # plot_wpgraph = True
    if plot_wpgraph:
        from matplotlib.ticker import FormatStrFormatter
        fig = plt.figure(figsize=(8.5, 8.5))
        ax = plt.gca()
        plot_graph(ax, graph, graph_node_pos, x_resolution, y_resolution)
        plot_edge_points(ax, graph_meas_nodes, c='c')
        for c in coord:
            ax.scatter(c[1], c[0], marker='s', facecolors='none', s=15*11.2, edgecolors='b', linewidth=0.1)
        fname = str(my_path + '/Evaluation_results/plots/wp_graph.pdf')
        ax.set_xlim(east_min - x_resolution * 0.2, east_max + x_resolution * 0.2)
        ax.set_ylim(north_min - y_resolution * 0.2, north_max + y_resolution * 0.2)
        ax.get_yaxis().get_major_formatter().set_useOffset(False)
        ax.get_yaxis().set_major_formatter(FormatStrFormatter('%.2f'))
        plt.title('Travel graph w. cell and node locations shown', fontsize=12)
        plt.xlabel('East [Degrees]')
        plt.ylabel('North [Degrees]')
        plt.tight_layout()
        # fig.savefig(fname)
        plt.show()

    print("test end")

    a, b = grid.getGridCoordinates(grid.llcoor[nx - 1][ny - 1][1], grid.llcoor[nx - 1][ny - 1][0])
    assert a == nx - 1, "ASSERT ERROR: getGridCoordinates reports wrong value"
    assert b == ny - 1, "ASSERT ERROR: getGridCoordinates reports wrong value"

    sgrid = Field(nx, ny, 0)
    sgrid.setGrid(lat_lc, lon_lc, lat_uc, lon_uc, proj='wgs84')

    print('Grids generated.')

    n1 = nx  # size in east
    n2 = ny  # size in north
    n = n1 * n2  # total size
    xgv = np.arange(1 / (2 * n1), 1, (1 / n1))

    sites1 = np.tile(np.arange(1, n1 + 1, 1), n1)
    sites2 = np.repeat(np.arange(1, n2 + 1, 1), n2)
    sites = [sites1, sites2]

    # Covariates (vectorised x variables)
    covars = np.array([np.repeat(1, n), np.sqrt((sites[0] - np.repeat(1, n) * sitesSOURCE) ** 2)[0, :]]).T # why not use \
                                                        # fliplr or flipud to flip the array up and down or left and right
    #covars2 = np.array([np.repeat(1, n), np.sqrt(np.sum((sites[0] - np.repeat(1, n) * sitesSOURCE) ** 2, axis=0))]).T

# print("can you stop here?")
    # Add trend (Assumes middle centered)
    mu1 = np.dot(covars, [[5.8], [0.085]])  # since the grid is reversed, therefore, no need to add -0.085 in the second element
    mu2 = np.dot(covars, [[29.0], [0.136]]) # otherwise, it is one type of the regression , it is one y = beta0 - beta1 * x
    mu_0 = np.concatenate((mu1, mu2))

    print('Generating a simulated sal-temp field (prior-environment)')
    x_prior = np.array(mu_0)

    # True trend (off-centre)
    mu1 = np.dot(covars, beta1)
    mu2 = np.dot(covars, beta2)
    print('True Environment Trend used T:{}, S:{}'.format(beta1, beta2))
    mu_0 = np.concatenate((mu1, mu2))

    print('Generating spatial covariance. \n sigma: {} \n range: {} \n noise: {} \n cell.dim {}'.format(sig, corr_range, noise, grid_cell_dimension))
    Cd = buildcov(sites, sites, sig, corr_range, nugg**2, analyse=False, cell_d=grid_cell_dimension)
    C0d = [[1, sig_st], [sig_st, 1]]
    Ct_true_field = np.kron(C0d, Cd)

    # The estimated spatial covariance
    print('Generating the spatial covariance matrix')
    C = buildcov(sites, sites, sig, corr_range, nugg**2, analyse=False, cell_d=grid_cell_dimension)
    C0 = [[1, sig_st], [sig_st, 1]]
    Ct = np.kron(C0, C)

    init_trace = np.trace(Ct)
    print('Initial TRACE: {}'.format(init_trace))

    plot_priors = True
    if plot_priors:
        print('Plotting priors')
        fname = str(my_path + '/Evaluation_results/plots/[prior]_temp.pdf')
        fig_prior_t = plt.figure(figsize=(fig_width, fig_height))
        plt.imshow(np.copy(x_prior[0:n]).reshape(nx, ny), vmin=t_min, vmax=t_max)
        plt.colorbar()
        plt.ylabel('Northing')
        plt.xlabel('Easting')
        plt.title('Temperature', fontsize=12)
        plt.tight_layout()
        plt.show()
        # plt.savefig(fname)

        fname = str(my_path + '/Evaluation_results/plots/[prior]_sal.pdf')
        fig_prior_s = plt.figure(figsize=(fig_width, fig_height))
        plt.imshow(np.copy(x_prior[n:]).reshape(nx, ny), vmin=s_min, vmax=s_max)
        plt.colorbar()
        plt.ylabel('Northing')
        plt.xlabel('Easting')
        plt.title('Salinity', fontsize=12)
        plt.tight_layout()
        plt.show()
        # plt.savefig(fname)


        print('Calculating excursion probabilities - for prior estimate')
        pp = []
        for i in range(0, n):
            SS = Ct[np.ix_([i, n+i], [i, n+i])]
            Mxi = [x_prior[i], x_prior[n + i]]
            # the following codes compute the integral of the pdf
# ==========================================================================================
# value,inform = mvnun(lower,upper,means,covar,[maxpts,abseps,releps])
#
# Wrapper for ``mvnun``.
#
# Parameters
# ----------
# lower : input rank-1 array('d') with bounds (d)
# upper : input rank-1 array('d') with bounds (d)
# means : input rank-2 array('d') with bounds (d,n)
# covar : input rank-2 array('d') with bounds (d,d)
#
# Other Parameters
# ----------------
# maxpts : input int, optional
#     Default: d*1000
# abseps : input float, optional
#     Default: 1e-06
# releps : input float, optional
#     Default: 1e-06
#
# Returns
# -------
# value : float
# inform : int
# ==========================================================================================
            pp.append(mvn.mvnun(np.array([[-np.inf], [-np.inf]]), np.array([[0], [0]]), np.subtract([Th[0], Th[1]], np.array(Mxi).ravel()), SS)[0])


        print('Plotting prior excursion probabilities')
        fname = str(my_path + '/Evaluation_results/plots/[prior]_excursion_prob.pdf')
        plt.figure(66, figsize=(fig_width, fig_height))
        plt.imshow(np.array(pp).reshape(nx, ny))
        plt.colorbar()
        plt.ylabel('Northing')
        plt.xlabel('Easting')
        plt.title('Prior Exc.Set - T. {} & S. {}'.format(Tthreshold, Sthreshold), fontsize=12)
        plt.tight_layout()
        plt.show()
        # plt.savefig(fname)

    print("finishing calcuating the probabilities")

    x_prior_0 = np.copy(x_prior)  # Copy this array, the x_prior will be mutated inside the next loop.
    Ct_0 = np.copy(Ct)            # Copy this array, the Ct will be mutated inside the next loop.
    Pred_error = None
    # Initial expected variance using the prior
    prior_ev = ExpectedVariance2(Th, x_prior, Ct_0, np.zeros((2*1, 2*nx*ny)), noise * np.eye(2 * 1), np.arange(0, nx*ny))

    # == Create structure for holding simulation data
    sim_data = {'true_env': [], 'x_prior_0': x_prior_0, 'Ct_0': Ct_0, 'prior_ev': prior_ev}
    for s in strategies:
        sim_data[s] = {}

    print("end of testing")

    for r in range(0, replicates):

        # Draw a "true" field
        print('Generating a simulated sal-temp field (true-environment)')
        x_true = np.array(mu_0 + np.dot(np.linalg.cholesky(Ct_true_field), np.random.randn(Ct_true_field.shape[0], 1)))
        sim_data['true_env'].append(x_true)

        print('Plotting trues')
        fname = str(my_path + '/Evaluation_results/plots/[true]_temp_{}.pdf'.format(r))
        fig_prior_t = plt.figure(figsize=(fig_width, fig_height))
        plt.imshow(np.copy(x_true[0:n]).reshape(nx, ny), extent=[east_min, east_max, north_min, north_max], vmin=t_min, vmax=t_max)
        plt.colorbar()
        ax = plt.gca()
        ax.get_yaxis().get_major_formatter().set_useOffset(False)
        #ax.get_yaxis().set_major_formatter(FormatStrFormatter('%.2f'))
        plt.ylabel('Northing')
        plt.xlabel('Easting')
        plt.title('Temperature', fontsize=12)
        plt.tight_layout()
        plt.show()
        # plt.savefig(fname)

        fname = str(my_path + '/Evaluation_results/plots/[true]_sal_{}.pdf'.format(r))
        fig_prior_s = plt.figure(figsize=(fig_width, fig_height))
        plt.imshow(np.copy(x_true[n:]).reshape(nx, ny), vmin=s_min, vmax=s_max)
        plt.colorbar()
        plt.ylabel('Northing')
        plt.xlabel('Easting')
        plt.title('Salinity', fontsize=12)
        plt.tight_layout()
        # plt.savefig(fname)
        plt.show()

        strat_count = 0
        print("end of ++++================================")
        print(r)
        print("end of ++++================================")

        # count = 0
        # for s in strategies:
        #     print(s)
        #     print(count)
        #     count = count + 1

        s = strategies[3] # only use this as a demonstration

        print('Load the prior into the grid structures')
        x_prior = np.copy(x_prior_0)  # Start with the same unaltered prior
        grid.data = x_prior[0:n].reshape(n1, n2)
        sgrid.data = x_prior[n:].reshape(n1, n2)

        # Excursion set in realization
        excursion_t = x_true[0:n].copy()
        excursion_t[x_true[0:n] > Tthreshold] = 1
        excursion_t[x_true[0:n] < Tthreshold] = 0

        excursion_s = x_true[n:].copy()
        excursion_s[x_true[n:] > Sthreshold] = 1
        excursion_s[x_true[n:] < Sthreshold] = 0

        excursion_st = np.multiply(excursion_t, excursion_s)

        # Start with initial parameters
        Ct = np.copy(Ct_0)
        static_north_nodes = list(static_north_nodes_0)
        static_east_nodes = list(static_east_nodes_0)
        static_zigzag_nodes = list(static_zigzag_nodes_0)
        route = []

        print("please end here")

        sim_data[s][r] = {}
        for var in ['Ct', 'x_prior', 'route', 'RMSE', 'R2', 'EV', 'run_time']:
            sim_data[s][r][var] = []

        sim_data[s][r]['RMSE'].append(np.sqrt((((x_true - x_prior) ** 2).sum()) / n))
        sim_data[s][r]['R2'].append(100 * (1 - (np.trace(Ct)) / init_trace))
        sim_data[s][r]['EV'].append(prior_ev)

        # ==== Start point
        if s == 'static_east' or s == 'static_zigzag':
            curr_point = 5
        else:
            curr_point = curr_point0

        plot_excursion = True
        if plot_excursion:
            print('Plotting true excursion sets')
            fname = str(my_path + '/Evaluation_results/plots/{}/[{}]_excursion_t_sim_{}.pdf'.format(s, s, r))
            plt.figure(figsize=(fig_width, fig_height))
            plt.imshow(excursion_t.reshape(n1, n2))
            plt.colorbar()
            plt.ylabel('Northing')
            plt.xlabel('Easting')
            plt.title('Excursion Boundary - Temp {} C'.format(Tthreshold), fontsize=12)
            plt.tight_layout()
            # plt.savefig(fname)

            fname = str(my_path + '/Evaluation_results/plots/{}/[{}]_excursion_s_sim_{}.pdf'.format(s, s, r))
            plt.figure(figsize=(fig_width, fig_height))
            plt.imshow(excursion_s.reshape(n1, n2))
            plt.colorbar()
            plt.ylabel('Northing')
            plt.xlabel('Easting')
            plt.title('Excursion Boundary - Sal. {} g/kg'.format(Sthreshold), fontsize=12)
            plt.tight_layout()
            # plt.savefig(fname)

            fname = str(my_path + '/Evaluation_results/plots/{}/[{}]_excursion_ts_sim_{}.pdf'.format(s, s, r))
            plt.figure(figsize=(fig_width, fig_height))
            plt.imshow(excursion_st.reshape(n1, n2))
            plt.colorbar()
            plt.ylabel('Northing')
            plt.xlabel('Easting')
            plt.title('Excursion Boundary - Temp {} C & Sal. {} g/kg'.format(Tthreshold, Sthreshold), fontsize=12)
            plt.tight_layout()
            # plt.savefig(fname)

        plot_excursion_prob = True
        if plot_excursion_prob:

            print('Calculating excursion probabilities - for true field')
            pp = []
            for i in range(0, n):
                SS = Ct[np.ix_([i, n + i], [i, n + i])]
                Mxi = [x_true[i], x_true[n + i]]
                pp.append(mvn.mvnun(np.array([[-np.inf], [-np.inf]]), np.array([[0], [0]]), np.subtract([Th[0], Th[1]], np.array(Mxi).ravel()), SS)[0])

            print('Plotting prior excursion probabilities')
            fname = str(my_path + '/Evaluation_results/plots/{}/[{}]_excursion_prob_sim_{}.pdf'.format(s, s, r))
            plt.figure(67, figsize=(fig_width, fig_height))
            plt.imshow(np.array(pp).reshape(nx, ny))
            plt.ylabel('Northing')
            plt.xlabel('Easting')
            plt.title('Exc.prob TRUE - T. {} & S. {} '.format(Tthreshold, Sthreshold), fontsize=12)
            plt.tight_layout()
            # plt.savefig(fname)

# print("it will end here")
# #%%
        current_strategy = s
        print('\n')
        print('**** Running {} strategy ****'.format(s))
        print('\n')

        run_init = time.time()

        for t in range(0, timesteps):

            print('\n')
            print('**** Strategy: {} - Replicate: {}/{} - Simulation step: {}/{} ****'.format(s, r, replicates, t, timesteps))
            print('\n')

            # Pre-allocate for new measurements for the coming leg
            y_measurement_t = []
            y_measurement_s = []
            pred_yt = []
            pred_ys = []

            # Use different strategies to find next point to sample
            if s == 'naive':
                next_point = naive(curr_point, grid, graph, Th, Ct, graph_node_points, graph_meas_nodes, x_prior, nx, ny)
            elif s == 'myopic':
                next_point = myopic(curr_point, grid, graph, Th, Ct, noise, graph_node_points, graph_meas_nodes, x_prior, nx, ny, sim_data[s][r]['EV'][-1])
            elif s == 'look-ahead':
                next_point = lookahead(curr_point, grid, graph, Th, Ct, noise, graph_node_points, graph_meas_nodes, x_prior, nx, ny, sim_data[s][r]['EV'][-1])
            elif s == 'static_north':
                next_point = static_north_nodes[0]
                static_north_nodes.pop(0)
            elif s == 'static_east':
                next_point = static_east_nodes[0]
                static_east_nodes.pop(0)
            elif s == 'static_zigzag':
                next_point = static_zigzag_nodes[0]
                static_zigzag_nodes.pop(0)
            else:
                print('Strategy not recognized - Skipping')
                continue

            route.append(next_point)
            print('AUV chooses node {} for next waypoint.'.format(next_point))
            print('\n')

            # Observation matrix for fist leg
            num_of_obs = graph_meas_nodes[(curr_point, next_point)].shape[0]
            G = np.zeros((2 * num_of_obs, 2 * nx * ny))  # Rows = # of samples, Col = ls_nx*ls_ny
            counter = 0

            # == Adding the intermediate samples
            print('Gathering data along chosen design')

            for sample_loc in graph_meas_nodes[(curr_point, next_point)]:
                # Observations
                sample_lat = sample_loc[1]
                sample_lon = sample_loc[0]
                gx, gy, gi = grid.getGridCoordinates(sample_lat, sample_lon, res_gx=nx, res_gy=ny)
                G[counter, gi] = 1
                G[num_of_obs + counter, (nx * ny) + gi] = 1
                counter += 1
                print("gi is ", gi)

                # Measurement
                y_measurement_t.extend(x_true[0:n][gi])
                y_measurement_s.extend(x_true[n:][gi])

                # Predicted measurement
                pred_yt.append(grid.data[gx, gy])
                pred_ys.append(sgrid.data[gx, gy])

            y_measurement = np.hstack((y_measurement_t, y_measurement_s))
            pred_y = np.hstack((pred_yt, pred_ys))

            # Measurement noise
            R = np.diag(np.repeat(noise[0, 0], num_of_obs).tolist() + np.repeat(noise[1, 1], num_of_obs).tolist())

            # Conditional distribution
            C1 = np.dot(Ct, G.T)
            C2 = np.dot(G, np.dot(Ct, G.T)) + R
            innovation = y_measurement - pred_y

print("end")
#%%
            similarity = np.array(np.linalg.lstsq(C2, innovation, rcond=None)[0])
            uncertainty_reduction = np.array(np.linalg.lstsq(C2, np.dot(G, Ct), rcond=None)[0])

            # Print the initial distributions
            if t == 0:

                # Prediction error 0
                Initial_prediction_error = np.copy(np.diag(Ct))
                Initial_prediction_error = Initial_prediction_error[0:nx * ny].reshape(nx, ny)

                # Initial value plots for mean
                if plot_evolution:
                    plt.figure(10, figsize=(fig_width * timesteps, fig_height))
                    plt.subplot(1, timesteps + 1, 1)
                    plt.imshow(grid.data, aspect='auto', vmin=t_min, vmax=t_max)
                    plt.xlabel('East Direction (grid cell)')
                    plt.ylabel('North Direction (grid cell)')
                    plt.title('At start', fontsize=12)
                    plt.colorbar()
                    plt.show()

                    print('Calculating excursion probabilities - for est field')
                    pp = []
                    for i in range(0, n):
                        SS = Ct[np.ix_([i, n+i], [i, n+i])]
                        Mxi = [x_prior[i], x_prior[n + i]]
                        pp.append(mvn.mvnun(np.array([[-np.inf], [-np.inf]]), np.array([[0], [0]]), np.subtract([Th[0], Th[1]], np.array(Mxi).ravel()), SS)[0])

                    print('Plotting est excursion probabilities')
                    plt.figure(13, figsize=(fig_width*timesteps, fig_height))
                    plt.subplot(1, timesteps+1, 1)
                    plt.imshow(np.array(pp).reshape(nx, ny), aspect='auto')
                    plt.xlabel('East Direction (grid cell)')
                    plt.ylabel('North Direction (grid cell)')
                    plt.title('Exc.prob at start '.format(Tthreshold, Sthreshold), fontsize=12)
                    plt.colorbar()
                    plt.show()

                    plt.figure(14, figsize=(fig_width, fig_height))
                    plt.title('Start')
                    plt.imshow(np.array(pp).reshape(nx, ny), aspect='auto')
                    # plt.savefig(str(my_path + '/Evaluation_results/plots/anim/{}_{}.png'.format(s, t)), dpi=200)
                    plt.show()

                    # Initial value plots for variance
                    plt.figure(11, figsize=(fig_width*timesteps, fig_height))
                    plt.subplot(1, timesteps+1, 1)
                    plt.imshow(Initial_prediction_error, aspect='auto', vmin=p_min, vmax=p_max)
                    plt.xlabel('East Direction (grid cell)')
                    plt.ylabel('North Direction (grid cell)')
                    plt.title('At start', fontsize=12)
                    plt.colorbar()
                    plt.show()

print("please stop here")
#%%
            # Load the new measurement into the predicted environment
            print('Update the prior with measurements')
            x_prior += np.dot(C1, similarity)[:, None]  # Update the predicted environment
            sim_data[s][r]['x_prior'].append(x_prior)
            grid.data = x_prior[0:n].reshape(nx, ny)
            sgrid.data = x_prior[n:].reshape(nx, ny)

            # Update the new uncertainty
            print('Update covariance matrix')
            Sigcond = Ct - np.dot(C1, uncertainty_reduction)

            if plot_evolution:
                plt.figure(10, figsize=(fig_width*timesteps, fig_height))
                plt.subplot(1, timesteps+1, t+2)
                plt.title('Route: {}'.format(route[-1]))
                plt.imshow(grid.data, aspect='auto', vmin=t_min, vmax=t_max)
                plt.xticks(())
                plt.yticks(())

            if t == timesteps - 1:
                try:
                    fname = str(my_path + '/Evaluation_results/plots/{}/[{}]_est-envir_{}.pdf'.format(s, s, r))
                    plt.subplots_adjust(left=0.03, bottom=0.15, right=0.97, top=0.9, wspace=0.30, hspace=0.2)
                    # plt.savefig(fname)
                    plt.close()
                except Exception as e:
                    print('Figure Error: Can save figure. Error message: {}'.format(e))
                    pass

            if plot_evolution:
                print('Calculating excursion probabilities - for est field')
                pp = []
                for i in range(0, n):
                    SS = Sigcond[np.ix_([i, n+i], [i, n+i])]
                    Mxi = [x_prior[i], x_prior[n + i]]
                    pp.append(mvn.mvnun(np.array([[-np.inf], [-np.inf]]), np.array([[0], [0]]), np.subtract([Th[0], Th[1]], np.array(Mxi).ravel()), SS)[0])

                print('Plotting est excursion probabilities')
                plt.figure(13, figsize=(fig_width*timesteps, fig_height))
                plt.subplot(1, timesteps+1, t+2)
                plt.title('Route: {}'.format(route[-1]), fontsize=12)
                plt.imshow(np.array(pp).reshape(nx, ny), aspect='auto')

                plt.figure(14, figsize=(fig_width, fig_height))
                plt.title('Strategy: {} Route: {}'.format(s, route[-1]), fontsize=12)
                plt.imshow(np.array(pp).reshape(nx, ny), aspect='auto')
                if s == 'static_east' or s == 'static_zigzag':
                    points_x = [int(nx/2)-1]
                    points_y = [0]
                else:
                    points_x = [nx-1]
                    points_y = [int(ny/2)-1]
                for place in route:
                    lat = graph_node_points[place][1]
                    lon = graph_node_points[place][0]
                    point_x, point_y, point_rolled = grid.getGridCoordinates(lat, lon, nx, ny)
                    points_x.append(point_x)
                    points_y.append(point_y)
                    plt.scatter(point_y, point_x, s=40, c=strategies_colors[strat_count])
                plt.plot(points_y, points_x, strategies_colors[strat_count], label='AUV Route')

                plt.xticks(())
                plt.yticks(())
                # plt.savefig(str(my_path + '/Evaluation_results/plots/anim/{}_{}_{}.png'.format(s, r, t)), dpi=200)

            if t == timesteps - 1:
                try:
                    fname = str(my_path + '/Evaluation_results/plots/{}/[{}]_est-excur_{}.pdf'.format(s, s, r))
                    plt.subplots_adjust(left=0.03, bottom=0.15, right=0.97, top=0.9, wspace=0.30, hspace=0.2)
                    # plt.savefig(fname)
                    plt.close()
                except Exception as e:
                    print('Figure Error: Can save figure. Error message: {}'.format(e))
                    pass

            Pred_error = np.diag(Sigcond)[0:n]
            sim_data[s][r]['Ct'].append(Pred_error)

            # == Plotting the variance
            if plot_evolution:
                plt.figure(11, figsize=(fig_width*timesteps, fig_height))
                plt.subplot(1, timesteps+1, t+2)
                plt.title('Route: {}'.format(route[-1]), fontsize=12)
                plt.imshow(Pred_error.reshape(nx, ny), aspect='auto', vmin=p_min, vmax=p_max)
                plt.xticks(())
                plt.yticks(())

            if t == timesteps - 1:
                try:
                    fname = str(my_path + '/Evaluation_results/plots/{}/[{}]_est-uncert_{}.pdf'.format(s, s, r))
                    plt.subplots_adjust(left=0.03, bottom=0.15, right=0.97, top=0.9, wspace=0.30, hspace=0.2)
                    # plt.savefig(fname)
                    plt.close()
                except Exception as e:
                    print('Figure Error: Can save figure. Error message: {}'.format(e))
                    pass

            # Simulate arrival
            curr_point = next_point
            Ct = Sigcond

            #Save data
            sim_data[s][r]['RMSE'].append(np.sqrt((((x_true-x_prior)**2).sum())/n))
            sim_data[s][r]['R2'].append(100 * (1 - (np.trace(Ct))/(init_trace)))
            sim_data[s][r]['EV'].append(ExpectedVariance2(Th, x_prior, Ct, np.zeros((2*1, 2*nx*ny)), noise * np.eye(2 * 1), np.arange(0, nx*ny)))
            sim_data[s][r]['run_time'].append(time.time() - run_init)

            print('RMSE {}'.format(sim_data[s][r]['RMSE'][-1]))
            print('TRACE: {}'.format(np.trace(Ct)))
            print('R-statistic: {}'.format(sim_data[s][r]['R2'][-1]))
            print('EV: {}'.format(sim_data[s][r]['EV'][-1]))
            print('\n')

        sim_data[s][r]['route'].append(route)
        sim_data[s][r]['run_time'].append(time.time() - run_init)

        fname = str(my_path + '/Evaluation_results/plots/{}/[{}]_excursion_set_w_route_{}.pdf'.format(s, s, r))
        plt.figure(20, figsize=(fig_width*3, fig_height))
        plt.subplot(1, 3, 1)
        plt.imshow(excursion_st.reshape(n1, n2))
        x_true_dummy = np.arange(0, nx, 1)
        y = np.arange(0, ny, 1)
        X, Y = np.meshgrid(x_true_dummy, y)
        plt.contour(X, Y, Pred_error.reshape(nx, ny), linestyles='-')
        plt.ylabel('Northing')
        plt.xlabel('Easting')
        if s == 'static_east' or s == 'static_zigzag':
            points_x = [int(nx/2)-1]
            points_y = [0]
        else:
            points_x = [nx-1]
            points_y = [int(ny/2)-1]
        for place in route:
            lat = graph_node_points[place][1]
            lon = graph_node_points[place][0]
            point_x, point_y, point_rolled = grid.getGridCoordinates(lat, lon, nx, ny)
            points_x.append(point_x)
            points_y.append(point_y)
            plt.scatter(point_y, point_x, s=40, c=strategies_colors[strat_count])
        plt.plot(points_y, points_x, strategies_colors[strat_count], label='AUV Route')
        plt.title('Strategy: {} - Steps: {}'.format(s, t), fontsize=12)
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.imshow(excursion_t.reshape(n1, n2))
        plt.ylabel('Northing')
        plt.xlabel('Easting')
        plt.title('Excursion Boundary - Temp {} C'.format(Tthreshold), fontsize=12)
        plt.tight_layout()

        plt.subplot(1, 3, 3)
        plt.imshow(excursion_s.reshape(n1, n2))
        plt.ylabel('Northing')
        plt.xlabel('Easting')
        plt.title('Excursion Boundary - Sal. {} g/kg'.format(Sthreshold), fontsize=12)
        plt.tight_layout()

        plt.savefig(fname)

        print('Calculating excursion probabilities - for final estimate')

        pp = []
        for i in range(0, n):
            SS = Ct[np.ix_([i, n+i], [i, n+i])]
            Mxi = [x_prior[i], x_prior[n + i]]
            pp.append(mvn.mvnun(np.array([[-np.inf], [-np.inf]]), np.array([[0], [0]]), np.subtract([Th[0], Th[1]], np.array(Mxi).ravel()), SS)[0])

        print('Plotting final estimate of excursion probabilities')
        plt.figure(66, figsize=(fig_width, fig_height))
        plt.imshow(np.array(pp).reshape(nx, ny))
        plt.ylabel('Northing')
        plt.xlabel('Easting')
        plt.title('Posterior Exc.Set - T. {} & S. {}'.format(Tthreshold, Sthreshold), fontsize=12)
        plt.tight_layout()

        fname = str(my_path + '/Evaluation_results/plots/{}/[{}][posterior]_excursion_prob_w_route_{}.pdf'.format(s, s, r))
        plt.figure(66, figsize=(fig_width, fig_height))
        if s == 'static_east' or s == 'static_zigzag':
            points_x = [int(nx/2)-1]
            points_y = [0]
        else:
            points_x = [nx-1]
            points_y = [int(ny/2)-1]
        for place in route:
            lat = graph_node_points[place][1]
            lon = graph_node_points[place][0]
            point_x, point_y, point_rolled = grid.getGridCoordinates(lat, lon, nx, ny)
            points_x.append(point_x)
            points_y.append(point_y)

            plt.scatter(point_y, point_x, s=40, c=strategies_colors[strat_count])
        plt.plot(points_y, points_x, strategies_colors[strat_count], label='{}'.format(s))
        plt.legend()

        plt.savefig(fname)

        fname = str(my_path + '/Evaluation_results/plots/{}/[{}]_excursion_true_w_route_{}.pdf'.format(s, s, r))
        plt.figure(67, figsize=(fig_width, fig_height))
        if s == 'static_east' or s == 'static_zigzag':
            points_x = [int(nx/2)-1]
            points_y = [0]
        else:
            points_x = [nx-1]
            points_y = [int(ny/2)-1]
        for place in route:
            lat = graph_node_points[place][1]
            lon = graph_node_points[place][0]
            point_x, point_y, point_rolled = grid.getGridCoordinates(lat, lon, nx, ny)
            points_x.append(point_x)
            points_y.append(point_y)

            plt.scatter(point_y, point_x, s=40, c=strategies_colors[strat_count])
        plt.plot(points_y, points_x, strategies_colors[strat_count], label='{}'.format(s))
        plt.legend()

        plt.savefig(fname)

        strat_count += 1

        with open('simdata.pickle', 'wb') as handle:  # Saving current simulation data
            pickle.dump(sim_data, handle)

        plt.close('all')  # Close all plots rel. to this sim

    with open('simdata.pickle', 'wb') as handle:  # Saving current simulation data (overwriting)
        pickle.dump(sim_data, handle)

    plt.close('all')  # Close all plots rel. to this sim

else:

    strategies = ['look-ahead', 'myopic']

    clrs = sns.color_palette("jet_r", len(strategies))

    print('Unloading previous simulation and plotting data to files')

    results = pickle.load(open("simdata.pickle", "rb"), encoding='latin1')

    plot_mode = 'avg'  # choose bet. 'avg' or 'individual'

    if plot_mode is 'avg':
        fname = str(my_path + '/Evaluation_results/plots/RMSE_avg.pdf')
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        clr_cnt = 0
        for s in strategies:
            s_mean = np.mean([results[s][r]['RMSE'] for r in range(0, replicates)], axis=0)
            s_std = np.std([results[s][r]['RMSE'] for r in range(0, replicates)], axis=0)
            ax.plot(s_mean, c=clrs[clr_cnt], label='{}'.format(s))
            #ax.fill_between(np.array(range(0, len(s_mean))), s_mean - s_std, s_mean + s_std, alpha=0.3, edgecolor='k', facecolor=clrs[clr_cnt])
            ax.errorbar(np.array(range(0, len(s_mean))), s_mean, yerr=s_std, fmt='o', c=clrs[clr_cnt])
            clr_cnt += 1
        ax.legend(loc=3, fontsize=20)
        plt.title('RMSE - Avg. of {} Replicates'.format(replicates), fontsize=20)
        plt.savefig(fname)

        fname = str(my_path + '/Evaluation_results/plots/EV_avg.pdf')
        fig2, ax = plt.subplots(figsize=(fig_width, fig_height))
        clr_cnt = 0
        for s in strategies:
            s_mean = np.mean([results[s][r]['EV'] for r in range(0, replicates)], axis=0)
            s_std = np.std([results[s][r]['EV'] for r in range(0, replicates)], axis=0)
            ax.plot(s_mean, c=clrs[clr_cnt], label='{}'.format(s))
            #ax.fill_between(np.array(range(0, len(s_mean))), s_mean - s_std, s_mean + s_std, alpha=0.3, edgecolor='k', facecolor=clrs[clr_cnt])
            ax.errorbar(np.array(range(0, len(s_mean))), s_mean, yerr=s_std, fmt='o', c=clrs[clr_cnt])
            clr_cnt += 1
        ax.legend(loc=3, fontsize=20)
        plt.title('Reduction in excursion set - Avg. of {} Replicates'.format(replicates), fontsize=20)
        plt.savefig(fname)

        fname = str(my_path + '/Evaluation_results/plots/RunTimes_avg.pdf')
        plt.figure(figsize=(fig_width, fig_height))
        fig3, ax = plt.subplots(figsize=(fig_width, fig_height))
        clr_cnt = 0
        for s in strategies:
            s_mean = np.mean([results[s][r]['run_time'] for r in range(0, replicates)], axis=0)
            s_std = np.std([results[s][r]['run_time'] for r in range(0, replicates)], axis=0)
            ax.plot(s_mean, c=clrs[clr_cnt], label='{}'.format(s))
            #ax.fill_between(np.array(range(0, len(s_mean))), s_mean - s_std, s_mean + s_std, alpha=0.3, edgecolor='k', facecolor=clrs[clr_cnt])
            ax.errorbar(np.array(range(0, len(s_mean))), s_mean, yerr=s_std, fmt='o', c=clrs[clr_cnt])
            clr_cnt += 1
        ax.legend(loc=2, fontsize=20)
        plt.title('Run times for strategies - Avg. of {} Replicates'.format(replicates), fontsize=20)
        plt.savefig(fname)

        fname = str(my_path + '/Evaluation_results/plots/R2_avg.pdf')
        fig4, ax = plt.subplots(figsize=(fig_width, fig_height))
        clr_cnt = 0
        for s in strategies:
            s_mean = np.mean([results[s][r]['R2'] for r in range(0, replicates)], axis=0)
            s_std = np.std([results[s][r]['R2'] for r in range(0, replicates)], axis=0)
            ax.plot(s_mean, c=clrs[clr_cnt], label='{}'.format(s))
            #ax.fill_between(np.array(range(0, len(s_mean))), s_mean - s_std, s_mean + s_std, alpha=0.3, edgecolor='k', facecolor=clrs[clr_cnt])
            ax.errorbar(np.array(range(0, len(s_mean))), s_mean, yerr=s_std, fmt='o', c=clrs[clr_cnt])
            clr_cnt += 1
        ax.legend(loc=2, fontsize=20)
        plt.title('R2 (explained variance) - Avg. of {} Replicates'.format(replicates), fontsize=20)
        plt.savefig(fname)

    elif plot_mode is 'individual':

        for r in range(0, 7):
            fname = str(my_path + '/Evaluation_results/plots/RMSE_{}.pdf'.format(r))
            plt.figure(figsize=(fig_width, fig_height))
            for s in strategies:
                plt.plot(results[s][r]['RMSE'], label='{}'.format(s))
            plt.legend()
            plt.title('RMSE')
            plt.savefig(fname)

            fname = str(my_path + '/Evaluation_results/plots/EV_{}.pdf'.format(r))
            plt.figure(figsize=(fig_width, fig_height))
            for s in strategies:
                plt.plot(results[s][r]['EV'], label='{}'.format(s))
            plt.legend()
            plt.title('Reduction in excursion set')
            plt.savefig(fname)

            fname = str(my_path + '/Evaluation_results/plots/RunTimes_{}.pdf'.format(r))
            plt.figure(figsize=(fig_width, fig_height))
            for s in strategies:
                plt.plot(results[s][r]['run_time'], label='{}'.format(s))
            plt.legend()
            plt.title('Run times for strategies')
            plt.savefig(fname)

            fname = str(my_path + '/Evaluation_results/plots/R2_{}.pdf'.format(r))
            plt.figure(figsize=(fig_width, fig_height))
            for s in strategies:
                plt.plot(results[s][r]['R2'], label='{}'.format(s))
            plt.legend()
            plt.title('R2 (explained variance)')
            plt.savefig(fname)

    else:
        print('[ERROR] - Plot mode not set')

    plt.close('all')
    print('# == Script Finished == #')

#%%

from scipy import ndimage, misc
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(10, 3))
ax1, ax2, ax3 = fig.subplots(1, 3)
img = misc.ascent()
img_45 = ndimage.rotate(img, 45, reshape = False)

#%% test out the fft gaussian random field
import numpy as np
import matplotlib.pyplot as plt
print("hello world")
# a = range(0, 10/2 + 1)
def fft_ind_gen(n):
    a = np.arange(0, n / 2 + 1)
    b = np.arange(0, n / 2 + 1)
    # print("here")
    b = - np.flipud(b)
    return a + b

test = fft_ind_gen(10)
print(test)
print("test end")

def gaussian_random_field(Pk = lambda k : k ** -3.0, size = 100):
    def Pk2(kx, ky):
        if kx == 0 and ky == 0:
            return 0.0
        return np.sqrt(Pk(np.sqrt(kx ** 2 + ky ** 2)))
    noise = np.fft.fft2(np.random.normal(size = (size, size)))
    amplitude = np.zeros((size, size))
    for i, kx in enumerate(fft_ind_gen(size)):
        for j, ky in enumerate(fft_ind_gen(size)):
            amplitude[i, j] = Pk2(kx, ky)
    return np.fft.ifft2(noise * amplitude)

for alpha in [-10.0, -3.0, -2.0]:
    out = gaussian_random_field(Pk = lambda k: k ** alpha, size = 256)
    plt.figure()
    plt.imshow(out.real, interpolation = "none")
    plt.colorbar()
    plt.show()


#%% testing of genertating random GP
import numpy as np
import matplotlib.pyplot as plt

n = 100
t = np.arange(0, n).reshape(-1, 1)
mu = np.random.rand(n).reshape(-1, 1)
sigma2 = 0.5
phiM = 0.19
H = np.zeros([n, n])
for i in range(n):
    for j in range(n):
        H[i, j] = np.abs(t[i] - t[j])

Sigma = sigma2 * (1 + phiM * H) * np.exp(-phiM * H)
L = np.linalg.cholesky(Sigma)
z = np.random.randn(n).reshape(-1, 1)
x = mu + np.dot(L, z)
plt.figure()
plt.plot(x)
plt.show()












