import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import mvn                     # For calculating multivariate pdf and cdf distributions

plt.ioff()  # Running plt.ioff() - plots are kept in background, plt.ion() plots are shown as they are generated
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 20})

strategies = ['look-ahead', 'static_north', 'static_east', 'static_zigzag', 'naive', 'myopic']

clrs = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

print('Unloading previous simulation and plotting data to files')

# === Cases
basecase = {'spatial_corr': 0.3,  # Ca. 1500 m effective correlation
            'temp_sal_corr': 0.6,
            'std_dev': 0.25,
            'beta_t': [[5.8], [0.1]],  # Avoid a centered ex.set
            'beta_s': [[29.0], [0.138]],
            'name': 'basecase',
            'id': 'basecase'}

replicates = 2
#analysis = ['analysis_off-center_1_20.pickle', 'analysis_off-center_2_20.pickle', 'analysis_off-center_4_20.pickle', 'analysis_off-center_5_20.pickle']
analysis = []
for i in range(1, 6):
    analysis.append('analysis_{}_10.pickle'.format(i))

plot_priors = False
sim_data = {}

for s in strategies:
    sim_data[s] = {}
    for var in ['RMSE', 'R2', 'EV', 'run_time', 'route', 'Ct']:
        sim_data[s][var] = []
x_true = []
for a in analysis:

    results = pickle.load(open("{}".format(a), "rb"), encoding='latin1')

    x_true.append(np.mean(results['basecase'][0]['true_env'], axis=0))

    for s in strategies:

        if plot_priors:
            r = 0
            n = 900
            nx = 30
            ny = 30
            t_min = 5.8
            t_max = 8.2
            s_min = 29.0
            s_max = 33.0
            # ==== Excursion set threshold
            Tthreshold = 7.0
            Sthreshold = 31.0
            Th = [Tthreshold, Sthreshold]

            print('Plotting priors')
            fname = 'prior_temp.pdf'
            fig_prior_t = plt.figure(figsize=(8, 8))
            plt.imshow(np.array(results['basecase'][0]['x_prior_0'][0:n]).reshape(nx, ny), vmin=t_min, vmax=t_max)
            plt.colorbar()
            plt.ylabel('Northing')
            plt.xlabel('Easting')
            plt.title('Temperature', fontsize=12)
            plt.tight_layout()
            plt.savefig(fname)

            fname = 'prior_sal.pdf'
            fig_prior_s = plt.figure(figsize=(8, 8))
            plt.imshow(np.array(results['basecase'][0]['x_prior_0'][n:]).reshape(nx, ny), vmin=s_min, vmax=s_max)
            plt.colorbar()
            plt.ylabel('Northing')
            plt.xlabel('Easting')
            plt.title('Salinity', fontsize=12)
            plt.tight_layout()
            plt.savefig(fname)

            print('Calculating excursion probabilities - for prior estimate')
            pp = []
            for i in range(0, n):
                SS = np.array(results['basecase'][0]['Ct_0'])[np.ix_([i, n + i], [i, n + i])]
                Mxi = [np.array(results['basecase'][0]['x_prior_0'])[i], np.array(results['basecase'][0]['x_prior_0'])[n + i]]
                pp.append(mvn.mvnun(np.array([[-np.inf], [-np.inf]]), np.array([[0], [0]]),
                                    np.subtract([Th[0], Th[1]], np.array(Mxi).ravel()), SS)[0])

            print('Plotting prior excursion probabilities')
            fname = 'prior_excursion_prob.pdf'
            plt.figure(66, figsize=(8, 8))
            plt.imshow(np.array(pp).reshape(nx, ny))
            plt.colorbar()
            plt.ylabel('Northing')
            plt.xlabel('Easting')
            plt.title('Prior Exc.Set - T. {} & S. {}'.format(Tthreshold, Sthreshold), fontsize=12)
            plt.tight_layout()
            plt.savefig(fname)

        sim_data[s]['RMSE'] += [results['basecase'][0][s][r]['RMSE'] for r in range(0, replicates)]
        sim_data[s]['R2'] += [results['basecase'][0][s][r]['R2'] for r in range(0, replicates)]
        sim_data[s]['EV'] += [results['basecase'][0][s][r]['EV'] for r in range(0, replicates)]
        sim_data[s]['run_time'] += [results['basecase'][0][s][r]['run_time'] for r in range(0, replicates)]
        sim_data[s]['route'] += [results['basecase'][0][s][r]['route'] for r in range(0, replicates)]
        sim_data[s]['Ct'] += [results['basecase'][0][s][r]['Ct'] for r in range(0, replicates)]

# plt.figure(1, figsize=(8, 8))
# plt.imshow(np.mean(x_true, axis=0)[0:900].reshape(30, 30), vmin=5.8, vmax=7.79)
# plt.show()
plt.figure(2, figsize=(8, 8))
#graph_node_pos = pickle.load(open("/media/tof/DATA/Googledrive/Pycharm Projects/Excursion_Sets - New/graph_node_pos.pickle", "rb"), encoding='latin1')
graph_node_pos = pickle.load(open("D:\Googledrive\Pycharm Projects\Excursion_Sets - New\graph_node_pos.pickle", "rb"), encoding='latin1')
clr_cnt = 0
clrs = ['r', 'b', 'k', 'g', 'y', 'm', 'c']

for s in ['naive', 'look-ahead', 'static_north', 'static_east', 'static_zigzag', 'myopic']:
    visited_points = []
    for r in sim_data[s]['route']:
        if s == 'static_east':
            r[0] = [5] + r[0]
        else:
            r[0] = [55] + r[0]
        visited_points.append([graph_node_pos[p] for p in r[0]])
        for lat_lon_route in visited_points:
            lat_lon_route = np.array(lat_lon_route)
            plt.plot(lat_lon_route[:, 0], lat_lon_route[:, 1], clrs[clr_cnt], alpha=0.02)

        # for point in lat_lon_route:
        #     plt.scatter(point[0], point[1], c=clrs[clr_cnt])
    clr_cnt += 1

plt.show(block=True)

fig, ax = plt.subplots(figsize=(8, 8))
fig2, ax2 = plt.subplots(figsize=(8, 8))
fig3, ax3 = plt.subplots(figsize=(8, 8))
fig4, ax4 = plt.subplots(figsize=(8, 8))

clr_cnt = 0
for s in strategies:
    rmse = np.array(sim_data[s]['RMSE'])
    r2 = np.array(sim_data[s]['R2'])
    ev = np.array(sim_data[s]['EV'])
    rt = np.array(sim_data[s]['run_time'])

    rmse_std = np.std(rmse, axis=0)[0:]
    ev_std = np.std(ev, axis=0)[0:]
    r2_std = np.std(r2, axis=0)[0:]

    rmse_mean = np.mean(rmse, axis=0)[0:]
    ev_mean = np.mean(ev, axis=0)[0:]
    r2_mean = np.mean(r2, axis=0)[0:]
    rt_mean = [0] + np.mean(rt, axis=0)[0:10].tolist()

    plot_mode = 'avg'  # choose bet. 'avg' or 'individual'

    if plot_mode is 'avg':
        fname = 'oc_avg_RMSE.pdf'
        ax.errorbar(np.array(range(1, len(rmse_mean)+1)), rmse_mean, yerr=rmse_std, alpha=0.7, fmt='-o', c=clrs[clr_cnt], capsize=5, elinewidth=2, markeredgewidth=1, label='{}'.format(s))
        ax.legend(loc=3, fontsize=20)
        ax.set_ylabel('RMSE [C]')
        ax.set_xlabel('Step [-]')
        fig.tight_layout()
        #plt.title('RMSE - Avg. of {} Replicates'.format(totrep), fontsize=20)
        fig.savefig(fname)

        fname = 'oc_avg_EV.pdf'
        ax2.errorbar(np.array(range(1, len(ev_mean)+1)), ev_mean, yerr=ev_std, alpha=0.7, fmt='-o', c=clrs[clr_cnt], capsize=5, elinewidth=2, markeredgewidth=1, label='{}'.format(s))
        ax2.legend(loc=3, fontsize=20)
        ax2.set_ylabel('Excursion Set Variance [-]')
        ax2.set_xlabel('Step [-]')
        fig2.tight_layout()
        #plt.title('Reduction in excursion set - Avg. of {} Replicates'.format(totrep), fontsize=20)
        fig2.savefig(fname)

        fname = 'oc_avg_R2.pdf'
        ax3.errorbar(np.array(range(1, len(r2_mean)+1)), r2_mean, yerr=r2_std, alpha=0.7, fmt='-o', c=clrs[clr_cnt], capsize=5, elinewidth=2, markeredgewidth=1, label='{}'.format(s))
        ax3.legend(loc=2, fontsize=20)
        ax3.set_ylabel('Explained Variance [%]')
        ax3.set_xlabel('Step [-]')
        fig3.tight_layout()
        #plt.title('R2 (explained variance) - Avg. of {} Replicates'.format(totrep), fontsize=20)
        fig3.savefig(fname)

        fname = 'oc_avg_Time.pdf'

        ax4.plot(rt_mean, c=clrs[clr_cnt], alpha=0.7, label='{}'.format(s))
        ax4.set_ylabel('Time [s]')
        ax4.set_xlabel('Step [-]')
        plt.xticks(np.arange(2, 11, 2))
        #ax4.errorbar(np.array(range(0, len(rt))), rt, yerr=np.std(rt, axis=0), fmt=pltform[clr_cnt], c=clrs[clr_cnt], capsize=5, elinewidth=2, markeredgewidth=1)
        ax4.legend(loc=2, fontsize=20)
        #plt.title('Run times for strategies - Avg. of {} Replicates'.format(totrep), fontsize=20)
        fig4.tight_layout()
        fig4.savefig(fname)

        clr_cnt += 1

print('# == Script Finished == #')
