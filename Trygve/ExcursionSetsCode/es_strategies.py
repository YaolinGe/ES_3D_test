from supporting_functions import *
import time


def lookahead(starting_point, world_grid, world_graph, Th, Sig_cond, noise, world_node_loc, world_meas_nodes, pred_field, res_x, res_y, prev_evar):
    print('Searching for the optimal next waypoint - (Look-ahead) 2 Steps w/ GP update and Markow simulation')
    fnp_init = time.time()
    Resulting_Variance_Change = {}
    Criteria = {}
    nx = res_x
    ny = res_y
    n = nx * ny
    
    # Calculate area of computation - at which indexes should the expected variance be calculated

    cell_horizon = 15  # Specifies how long the calc. horizon should be - This to limit the number of evaluations

    cx, cy, cell_origin = world_grid.getGridCoordinates(world_node_loc[starting_point][1], world_node_loc[starting_point][0], res_gx=nx, res_gy=ny)
    cell_indexes = np.array(np.arange(0, res_x * res_y)).reshape(res_x, res_y)
    dummy = np.zeros((res_x, res_y))
    n_idx = np.arange(cx - cell_horizon, cx + cell_horizon)
    n_idy = np.arange(cy - cell_horizon, cy + cell_horizon)
    n_idx = n_idx[n_idx >= 0]
    n_idx = n_idx[n_idx < res_x]
    n_idy = n_idy[n_idy >= 0]
    n_idy = n_idy[n_idy < res_y]
    eval_indexes = np.array(cell_indexes[np.ix_(n_idx, n_idy)]).flatten()

    # For viewing restricted integral area
    # dummy[np.ix_(n_idx, n_idy)] = 1
    # plt.figure()
    # plt.imshow(dummy)
    # plt.show()

    # Find the alternative routes - recursively
    def find_all_routes(world_graph, starting_point, look_ahead=2):

        def find_all_paths(world_graph, starting_point, path=[], horizon=look_ahead):
            if len(path) >= horizon:
                return []
            path = path + [starting_point]
            if len(path) == horizon:
                return [path]
            paths = []
            for node in world_graph[starting_point].keys():
                paths += find_all_paths(world_graph, node, path, horizon=look_ahead)
            return paths

        all_paths = find_all_paths(world_graph, starting_point)

        sane_routes = []
        for p in all_paths:
            if len(np.unique(p)) == look_ahead:
                sane_routes.append(p)
            else:
                pass
        return sane_routes

    routes = find_all_routes(world_graph, starting_point)
    # route_dist = [wgs84_dist(world_node_loc[r[0]][1], world_node_loc[r[0]][0], world_node_loc[r[1]][1], world_node_loc[r[1]][0]) for r in routes]

    alternative = 0

    for route in routes:

        print('Evaluating route: {}'.format(route))

        start_node = route[0]
        end_node = route[1]
        pruned_nodes = [start_node]
        mc_Ev = []

        for mc in range(0, 10):

            estimated_field = pred_field  # Reset environment
            estimated_Sig_cond = Sig_cond  # Reset uncertainty

            print('MC - real {}'.format(mc))

            pred_yt = []
            pred_ys = []

            num_of_measurements = world_meas_nodes[(start_node, end_node)].shape[0]
            num_of_cells_assimilated = 0
            old_si = None
            GG = np.zeros((2 * num_of_measurements, 2 * res_x * res_y))
            obs_counter = 0

            # == Adding the intermediate samples
            for loc in world_meas_nodes[(start_node, end_node)]:
                # Observations
                sample_lt = loc[1]
                sample_ln = loc[0]
                sx, sy, si = world_grid.getGridCoordinates(sample_lt, sample_ln, res_gx=nx, res_gy=ny)
                if not old_si:
                    old_si = si
                    num_of_cells_assimilated += 1
                elif si != old_si:
                    old_si = si
                    num_of_cells_assimilated += 1
                GG[obs_counter, si] = 1
                GG[num_of_measurements + obs_counter, (nx * ny) + si] = 1
                obs_counter += 1

                # Simulated Measurement
                # sim_mu_t = estimated_field[0:n][si]
                # sim_mu_s = estimated_field[n:][si]
                # sim_sig_t = np.sqrt(np.diag(estimated_Sig_cond)[0:n][si])
                # sim_sig_s = np.sqrt(np.diag(estimated_Sig_cond)[n:][si])
                # sim_t = np.random.normal(sim_mu_t, sim_sig_t, 1)
                # sim_s = np.random.normal(sim_mu_s, sim_sig_s, 1)
                #
                # y_measurement_t.append(sim_t)
                # y_measurement_s.append(sim_s)

                # Predicted measurement
                pred_yt.append(estimated_field[0:n][si])
                pred_ys.append(estimated_field[n:][si])

            # Measurement noise
            RR = np.diag(np.repeat(noise[0, 0], num_of_measurements).tolist() + np.repeat(noise[1, 1], num_of_measurements).tolist())

            # Simulated Measurement - updated using Cholesky w/ cross-correlation
            mu_sim = np.dot(GG, estimated_field)
            sigma_sim = np.dot(np.dot(GG, estimated_Sig_cond), GG.T) + RR
            L_sim = np.linalg.cholesky(sigma_sim)
            sim_vals = mu_sim + np.dot(L_sim, np.random.randn(2 * num_of_measurements, 1))

            # Update the field - Simulate actual sampling
            y_measurement = sim_vals
            pred_y = np.vstack((pred_yt, pred_ys))

            # Conditional distribution
            C1 = np.dot(estimated_Sig_cond, GG.T)
            C2 = np.dot(GG, np.dot(estimated_Sig_cond, GG.T)) + RR
            innovation = y_measurement - pred_y
            similarity = np.array(np.linalg.lstsq(C2, innovation, rcond=None)[0])
            uncertainty_reduction = np.array(np.linalg.lstsq(C2, np.dot(GG, estimated_Sig_cond), rcond=None)[0])

            # Load the new measurement into the predicted environment
            estimated_field = estimated_field + np.dot(C1, similarity)  # Update the predicted environment

            # Update the new uncertainty
            estimated_Sig_cond = estimated_Sig_cond - np.dot(C1, uncertainty_reduction)

            next_routes = find_all_routes(world_graph, end_node, look_ahead=2)
            next_nodes = [node[1] for node in next_routes]
            for pruned in pruned_nodes:
                next_nodes.remove(pruned)
            Ev = []

            for nn in next_nodes:

                if nn in pruned_nodes:
                    continue

                print('Evaluating integral over node: {}'.format(nn))

                num_of_measurements2 = world_meas_nodes[(end_node, nn)].shape[0]
                GG2 = np.zeros((2 * num_of_measurements2, 2 * res_x * res_y))
                obs_counter2 = 0

                # == Adding the intermediate samples
                for loc in world_meas_nodes[(end_node, nn)]:
                    # Observations
                    sample_lt = loc[1]
                    sample_ln = loc[0]
                    sx, sy, si = world_grid.getGridCoordinates(sample_lt, sample_ln, res_gx=nx, res_gy=ny)
                    GG2[obs_counter2, si] = 1
                    GG2[num_of_measurements2 + obs_counter2, (nx * ny) + si] = 1
                    obs_counter2 += 1

                # Measurement noise
                RR2 = np.diag(np.repeat(noise[0, 0], num_of_measurements2).tolist() + np.repeat(noise[1, 1], num_of_measurements2).tolist())

                Ev.append(ExpectedVariance2(Th, estimated_field, estimated_Sig_cond, GG2, RR2, eval_indexes))

            mc_Ev.append(np.min(Ev))

            if len(pruned_nodes) <= 3 and len(next_nodes) > 3:
                pruned_nodes.append(next_nodes[np.argmax(Ev)])

        # Store calculation and save the resulting variance for the survey line
        Resulting_Variance_Change[alternative] = np.abs(prev_evar-np.mean(mc_Ev))
        alternative += 1

    # Step over alternatives and choose the best
    def findkey(d):
        v = list(d.values())
        k = list(d.keys())
        return k[v.index(max(v))]

    for alternative in Resulting_Variance_Change:
        Criteria[alternative] = Resulting_Variance_Change[alternative]

    best_node_c = findkey(Criteria)

    dt_fnp = time.time() - fnp_init

    print('\n')
    print('ANALYSIS RESULTS')
    print('\n')
    print('Analysis Time %.2fs.' % dt_fnp)
    print('\n')
    print('EXCURSION SET ALTERNATIVES:')
    print(Resulting_Variance_Change)
    print('\n')

    return routes[best_node_c][1]


def myopic(starting_point, world_grid, world_graph, Th, Sig_cond, noise, world_node_loc, world_meas_nodes, pred_field, res_x, res_y, prev_evar):
    print('Searching for the optimal next waypoint - Myopic')
    fnp_init = time.time()
    Resulting_Variance_Change = {}
    Criteria = {}
    nx = res_x
    ny = res_y
    n = nx * ny

    # Interesting cells to evaluate
    cx, cy, cell_origin = world_grid.getGridCoordinates(world_node_loc[starting_point][1], world_node_loc[starting_point][0], res_gx=nx, res_gy=ny)
    cell_indexes = np.array(np.arange(0, res_x * res_y)).reshape(res_x, res_y)
    dummy = np.zeros((res_x, res_y))
    cell_horizon = 15
    n_idx = np.arange(cx - cell_horizon, cx + cell_horizon)
    n_idy = np.arange(cy - cell_horizon, cy + cell_horizon)
    n_idx = n_idx[n_idx >= 0]
    n_idx = n_idx[n_idx < res_x]
    n_idy = n_idy[n_idy >= 0]
    n_idy = n_idy[n_idy < res_y]
    eval_indexes = np.array(cell_indexes[np.ix_(n_idx, n_idy)]).flatten()
    dummy[np.ix_(n_idx, n_idy)] = 1

    # Find the alternative routes - recursively
    def find_all_routes(world_graph, starting_point, look_ahead=2):

        def find_all_paths(world_graph, starting_point, path=[], horizon=look_ahead):
            if len(path) >= horizon:
                return []
            path = path + [starting_point]
            if len(path) == horizon:
                return [path]
            paths = []
            for node in world_graph[starting_point].keys():
                paths += find_all_paths(world_graph, node, path, horizon=look_ahead)
            return paths

        all_paths = find_all_paths(world_graph, starting_point)

        sane_routes = []
        for p in all_paths:
            if len(np.unique(p)) == look_ahead:
                sane_routes.append(p)
            else:
                pass
        return sane_routes

    routes = find_all_routes(world_graph, starting_point)
    # route_dist = [wgs84_dist(world_node_loc[r[0]][1], world_node_loc[r[0]][0], world_node_loc[r[1]][1], world_node_loc[r[1]][0]) for r in routes]

    for alternative in world_graph:

        if alternative in world_graph[starting_point]:

            num_of_measurements = world_meas_nodes[(starting_point, alternative)].shape[0]
            num_of_cells_assimilated = 0
            old_si = None
            GG = np.zeros((2 * num_of_measurements, 2 * res_x * res_y))
            obs_counter = 0

            # == Adding the intermediate samples
            for loc in world_meas_nodes[(starting_point, alternative)]:
                # Observations
                sample_lt = loc[1]
                sample_ln = loc[0]
                sx, sy, si = world_grid.getGridCoordinates(sample_lt, sample_ln, res_gx=nx, res_gy=ny)
                if not old_si:
                    old_si = si
                    num_of_cells_assimilated += 1
                elif si != old_si:
                    old_si = si
                    num_of_cells_assimilated += 1
                GG[obs_counter, si] = 1
                GG[num_of_measurements + obs_counter, (nx * ny) + si] = 1
                obs_counter += 1

            # Noise
            #print('Num of cells:{}'.format(num_of_cells_assimilated))
            RR = np.diag(np.repeat(noise[0, 0], num_of_measurements).tolist() + np.repeat(noise[1, 1], num_of_measurements).tolist())

            # Predicted observation covariance
            EVar = ExpectedVariance2(Th, pred_field, Sig_cond, GG, RR, eval_indexes)

            # Save the resulting reduction in variance for the survey line per observation
            Resulting_Variance_Change[alternative] = np.abs(prev_evar - EVar)

    # Step over alternatives and choose the best

    def normalize_dict(d):
        factor = 1.0 / sum(d.values())
        for k in d:
            d[k] *= factor
        return d

    def findkey(d):
        v = list(d.values())
        k = list(d.keys())
        return k[v.index(max(v))]

    dt_fnp = time.time() - fnp_init

    print('\n')
    print('ANALYSIS RESULTS')
    print('\n')
    print('Analysis Time %.2fs.' % dt_fnp)
    print('\n')
    print('EXCURSION SET ALTERNATIVES:')
    print(Resulting_Variance_Change)
    print('\n')

    for alternative in Resulting_Variance_Change:
        Criteria[alternative] = Resulting_Variance_Change[alternative]

    best_node_c = findkey(Criteria)

    return best_node_c


def naive(starting_point, world_grid, world_graph, Th, Sig_cond, world_node_loc, world_meas_nodes, pred_field, res_x, res_y):
    print('Searching for the optimal next waypoint - Naive')
    fnp_init = time.time()
    EP_score = {}
    Criteria = {}
    nx = res_x
    ny = res_y
    n = nx * ny

    print('Calculating excursion probabilities - for current estimate')

    pp = []
    for i in range(0, n):
        SS = Sig_cond[np.ix_([i, n + i], [i, n + i])]
        Mxi = [pred_field[i], pred_field[n + i]]
        pp.append(mvn.mvnun(np.array([[-np.inf], [-np.inf]]), np.array([[0], [0]]),
                            np.subtract([Th[0], Th[1]], np.array(Mxi).ravel()), SS)[0])

    for alternative in world_graph:

        if alternative in world_graph[starting_point]:

            num_of_measurements = world_meas_nodes[(starting_point, alternative)].shape[0]
            num_of_cells_assimilated = 0
            old_si = None
            GG = np.zeros((num_of_measurements, res_x * res_y))
            obs_counter = 0

            # == Adding the intermediate samples
            for loc in world_meas_nodes[(starting_point, alternative)]:
                # Observations
                sample_lt = loc[1]
                sample_ln = loc[0]
                sx, sy, si = world_grid.getGridCoordinates(sample_lt, sample_ln, res_gx=nx, res_gy=ny)
                if not old_si:
                    old_si = si
                    num_of_cells_assimilated += 1
                elif si != old_si:
                    old_si = si
                    num_of_cells_assimilated += 1
                GG[obs_counter, si] = 1
                obs_counter += 1

            pred_excursion_prob = np.dot(np.array(pp), GG.T)
            ep_score = []
            for ep in pred_excursion_prob:
                ep_score.append(np.abs(ep - 0.5))
            # Save the resulting variance for the survey line
            EP_score[alternative] = np.mean(ep_score)

    # Step over alternatives and choose the best (lowest EP score)
    def normalize_dict(d):
        factor = 1.0 / sum(d.values())
        for k in d:
            d[k] *= factor
        return d

    def findkey(d):
        v = list(d.values())
        k = list(d.keys())
        return k[v.index(min(v))]

    # Normalize results to sum to 1
    Resulting_EP_score = normalize_dict(EP_score)

    dt_fnp = time.time() - fnp_init

    print('\n')
    print('ANALYSIS RESULTS')
    print('\n')
    print('Analysis Time %.2fs.' % dt_fnp)
    print('\n')
    print('EP SCORE ALTERNATIVES:')
    print(EP_score)
    print('\n')

    for alternative in Resulting_EP_score:
        Criteria[alternative] = Resulting_EP_score[alternative]

    best_node_c = findkey(Criteria)

    return best_node_c
