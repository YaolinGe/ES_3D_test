from __future__ import print_function
from __future__ import division
import sys
import pickle
from es_strategies import *
from es_cases import *

# == Plotting
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from matplotlib import pyplot as plt
plt.ioff()  # Running plt.ioff() - plots are kept in background, plt.ion() plots are shown as they are generated
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 26})

my_path = os.path.dirname(os.path.abspath(__file__))

config_1 = [std_low, std_high]
config_2 = [cor_low, cor_high]
config_3 = [ts_cor_low, ts_cor_high]
config_4 = [t_only]
config_5 = [basecase]

sim_configs = config_1

# === Simname
simname = 'std_oc'

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
sitesSOURCE = [[31], [31]]

# ==== Excursion set threshold
Tthreshold = 7.16
Sthreshold = 31.185
Th = [Tthreshold, Sthreshold]

# ==== Script mode (sim or view old results)
sim = True

# ==== Sampling strategies
strategies = ['naive', 'myopic', 'look-ahead', 'static_north', 'static_east', 'static_zigzag']
static_north_nodes_0 = [54, 55, 56, 57, 58, 59, 60, 61, 62, 61]
static_east_nodes_0 = [16, 27, 38, 49, 60, 71, 82, 93, 104, 115]
#static_zigzag_nodes_0 = [67, 79, 91, 103, 115, 105, 95, 85, 75, 65]
static_zigzag_nodes_0 = [15, 26, 36, 47, 57, 68, 78, 89, 99, 110]
strategies_colors = ['r', 'b', 'k', 'g', 'y', 'm', 'c']

# ==== Simulation steps
timesteps = 10
replicates = 20  # Replicate simulations

# ==== Start point
curr_point0, curr_point = 53, 53

# ==== The AUV route
route = []

# ==== Toggle to generate plots
t_min = 5.8
t_max = 8.435
s_min = 29.0
s_max = 33.278
p_min = 0
p_max = 0.2525

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
width = wgs84_dist(grid.llcoor[nx - 1][0][1], grid.llcoor[nx - 1][0][0], grid.llcoor[nx - 1][ny - 1][1],
                   grid.llcoor[nx - 1][ny - 1][0]) * 0.002
height = wgs84_dist(grid.llcoor[0][0][1], grid.llcoor[0][0][0], grid.llcoor[nx - 1][0][1],
                    grid.llcoor[nx - 1][0][0]) * 0.002

# == Collecting the lat/lon coordinates from the new grid class into a new variable "coord"
coord = zip(glat, glon)  # Set the lat lon pairs into the coord variable (will be used in for-loop for retrieving data

a, b = grid.getGridCoordinates(grid.llcoor[nx - 1][ny - 1][1], grid.llcoor[nx - 1][ny - 1][0])
assert a == nx - 1, "ASSERT ERROR: getGridCoordinates reports wrong value"
assert b == ny - 1, "ASSERT ERROR: getGridCoordinates reports wrong value"

sgrid = Field(nx, ny, 0)
sgrid.setGrid(lat_lc, lon_lc, lat_uc, lon_uc, proj='wgs84')

# ==== Plotting the guide graph with the correct width and height
plot_wpgraph = False
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
    fig.savefig(fname)

print('Grids generated.')

n1 = nx  # size in east
n2 = ny  # size in north
n = n1 * n2  # total size
xgv = np.arange(1 / (2 * n1), 1, (1 / n1))

sites1 = np.tile(np.arange(1, n1 + 1, 1), n1)
sites2 = np.repeat(np.arange(1, n2 + 1, 1), n2)
sites = [sites1, sites2]

# ==== Analysis structure
analysis = {}
for conf in sim_configs:
    analysis[conf['id']] = []

for conf in sim_configs:

    print('#=#=#=#=# Running sens. analysis. Config: {} #=#=#=#=#'.format(conf['name']))

    # == Timer
    t_init = time.time()

    # ==== Trend parameters
    beta1 = conf['beta_t']  # Start_temp, temperature_trend_factor
    beta2 = conf['beta_s']  # Start_sal, salinity_trend_factor

    # ==== Correlation T-S
    sig_st = conf['temp_sal_corr']

    # ==== Noise
    noise = np.array([[conf['noise'][0] ** 2, 0], [0, conf['noise'][1] ** 2]])
    nugg = 0.05

    # ===== Prior knowledge
    sig = conf['std_dev']

    # ===== Spatial correlation
    corr_range = conf['spatial_corr']

    if sim:

        # Co-variates
        covars = np.array([np.repeat(1, n), np.sqrt((sites[0] - np.repeat(1, n) * sitesSOURCE) ** 2)[0, :]]).T

        # Add trend (Assumes middle centered)
        mu1 = np.dot(covars, [[5.8], [0.085]])
        mu2 = np.dot(covars, [[29.0], [0.136]])
        mu = np.concatenate((mu1, mu2))

        print('Generating a simulated sal-temp field (prior-environment)')
        x_prior = np.array(mu)

        # True trend (off-centre if different trends are used - Not here)
        mu1 = np.dot(covars, beta1)
        mu2 = np.dot(covars, beta2)
        print('True Environment Trend used T:{}, S:{}'.format(beta1, beta2))
        mu = np.concatenate((mu1, mu2))

        print('Generating spatial covariance. \n sigma: {} \n range: {} \n noise: {} \n cell.dim {}'.format(sig, corr_range, nugg**2, grid_cell_dimension))
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

        x_prior_0 = np.copy(x_prior)  # Copy this array, the x_prior will be mutated inside the next loop.
        Ct_0 = np.copy(Ct)  # Copy this array, the Ct will be mutated inside the next loop.
        Pred_error = None
        # Initial expected variance using the prior
        prior_ev = ExpectedVariance2(Th, x_prior, Ct_0, np.zeros((2 * 1, 2 * nx * ny)), (nugg**2) * np.eye(2 * 1), np.arange(0, nx * ny))

        # == Create structure for holding simulation data
        sim_data = {'true_env': [], 'x_prior_0': x_prior_0, 'Ct_0': Ct_0, 'prior_ev': prior_ev}
        for s in strategies:
            sim_data[s] = {}

        for r in range(0, replicates):

            # Draw a "true" field
            print('Generating a simulated sal-temp field (true-environment)')
            x_true = np.array(mu + np.dot(np.linalg.cholesky(Ct_true_field), np.random.randn(Ct_true_field.shape[0], 1)))
            sim_data['true_env'].append(x_true)

            strat_count = 0

            for s in strategies:

                print('Load the prior into the grid structures')
                x_prior = np.copy(x_prior_0)  # Start with the same unaltered prior
                grid.data = x_prior[0:n].reshape(n1, n2)
                sgrid.data = x_prior[n:].reshape(n1, n2)

                # Start with initial parameters
                Ct = np.copy(Ct_0)
                static_north_nodes = list(static_north_nodes_0)
                static_east_nodes = list(static_east_nodes_0)
                static_zigzag_nodes = list(static_zigzag_nodes_0)
                route = []

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

                current_strategy = s
                print('\n')
                print('**** Running {} strategy ****'.format(s))
                print('\n')

                run_init = time.time()

                for t in range(0, timesteps):

                    print('\n')
                    print('**** Strategy: {} - Replicate: {}/{} - Simulation step: {}/{} ****'.format(s, r + 1, replicates,
                                                                                                      t + 1, timesteps))
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
                    similarity = np.array(np.linalg.lstsq(C2, innovation, rcond=None)[0])
                    uncertainty_reduction = np.array(np.linalg.lstsq(C2, np.dot(G, Ct), rcond=None)[0])

                    # Print the initial distributions
                    if t == 0:

                        # Prediction error 0
                        Initial_prediction_error = np.copy(np.diag(Ct))
                        Initial_prediction_error = Initial_prediction_error[0:nx * ny].reshape(nx, ny)

                    # Load the new measurement into the predicted environment
                    print('Update the prior with measurements')
                    x_prior = x_prior + np.dot(C1, similarity)[:, None]  # Update the predicted environment
                    sim_data[s][r]['x_prior'].append(x_prior)
                    grid.data = x_prior[0:n].reshape(nx, ny)
                    sgrid.data = x_prior[n:].reshape(nx, ny)

                    # Update the new uncertainty
                    print('Update covariance matrix')
                    Ct -= np.dot(C1, uncertainty_reduction)

                    Pred_error = np.diag(Ct)[0:n]
                    sim_data[s][r]['Ct'].append(Pred_error)

                    # Simulate arrival
                    curr_point = next_point
                    rmse = np.sqrt((((x_true - x_prior) ** 2).sum()) / n)
                    r2 = 100 * (1 - (np.trace(Ct)) / init_trace)
                    evar = ExpectedVariance2(Th, x_prior, Ct, np.zeros((2 * 1, 2 * nx * ny)), noise * np.eye(2 * 1), np.arange(0, nx * ny))

                    # Save data
                    sim_data[s][r]['RMSE'].append(rmse)
                    sim_data[s][r]['R2'].append(r2)
                    sim_data[s][r]['EV'].append(evar)
                    sim_data[s][r]['run_time'].append(time.time() - run_init)

                    del rmse, r2, evar

                    print('Memory check: sim_data:{}, analysis:{}'.format(sys.getsizeof(sim_data), sys.getsizeof(analysis)))
                    print('RMSE {}'.format(sim_data[s][r]['RMSE'][-1]))
                    print('TRACE: {}'.format(np.trace(Ct)))
                    print('R-statistic: {}'.format(sim_data[s][r]['R2'][-1]))
                    print('EV: {}'.format(sim_data[s][r]['EV'][-1]))
                    print('\n')

                sim_data[s][r]['route'].append(route)
                sim_data[s][r]['run_time'].append(time.time() - run_init)

                strat_count += 1

    analysis[conf['id']].append(sim_data)
    del x_prior_0, Ct_0, prior_ev, sim_data

with open('analysis_{}.pickle'.format(simname), 'wb') as handle:  # Saving current simulation data (overwriting)
    pickle.dump(analysis, handle)

print('# == Script Finished == #')
