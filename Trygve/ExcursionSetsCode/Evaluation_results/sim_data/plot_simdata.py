import pickle
import numpy as np
import matplotlib.pyplot as plt


plt.ioff()  # Running plt.ioff() - plots are kept in background, plt.ion() plots are shown as they are generated
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 26})

strategies = ['naive', 'look-ahead', 'myopic', 'static_north', 'static_east', 'static_zigzag']

clrs = ['r', 'b', 'k', 'g', 'y', 'm', 'c']

print('Unloading previous simulation and plotting data to files')


sims = ['simdata_t10_r10_c05_int_-inf_0_bra_resultat.pickle', 'simdata_t10_r25_cr05_int_0_inf.pickle', 'simdata_t10_r50_cr05.pickle']
rmse = []
r2 = []
ev = []
rt = []
counter = 0
rep = [10, 25, 50]
pltform = ['-or', '-ob', '-ok', '-og', '-oy', '-om', '-oc']
totrep = np.sum(rep)
clr_cnt = 0
fig, ax = plt.subplots(figsize=(8, 8))
fig2, ax2 = plt.subplots(figsize=(8, 8))
fig3, ax3 = plt.subplots(figsize=(8, 8))
fig4, ax4 = plt.subplots(figsize=(8, 8))

for s in strategies:
    rmse = []
    r2 = []
    ev = []
    rt = []
    try:
        for sim in sims:
            results = pickle.load(open("{}".format(sim), "rb"), encoding='latin1')
            replicates = rep[counter]
            rmse = np.array([results[s][r]['RMSE'] for r in range(0, replicates)])
            r2 = np.array([results[s][r]['R2'] for r in range(0, replicates)])
            ev = np.array([results[s][r]['EV'] for r in range(0, replicates)])
            rt = np.array([results[s][r]['run_time'] for r in range(0, replicates)])

    except IndexError:
        pass

    # plt.figure()
    # plt.plot(t)
    # plt.plot(np.mean(np.array(t), axis=1), 'k')
    # plt.plot(np.std(t, axis=1), 'r')
    # plt.show()

    rmse_std = np.std(rmse, axis=0)[1:]
    ev_std = np.std(ev, axis=0)[1:]
    r2_std = np.std(r2, axis=0)[1:]

    rmse = np.mean(rmse, axis=0)[1:]
    ev = np.mean(ev, axis=0)[1:]
    r2 = np.mean(r2, axis=0)[1:]
    rt = [0] + np.mean(rt, axis=0)[0:10].tolist()


    plot_mode = 'avg'  # choose bet. 'avg' or 'individual'

    if plot_mode is 'avg':
        fname = 'avg_RMSE.pdf'
        ax.errorbar(np.array(range(1, len(rmse)+1)), rmse, yerr=rmse_std, fmt=pltform[clr_cnt], c=clrs[clr_cnt], capsize=5, elinewidth=2, markeredgewidth=1, label='{}'.format(s))
        ax.legend(loc=3, fontsize=20)
        ax.set_ylabel('RMSE [C]')
        ax.set_xlabel('Step [-]')
        fig.tight_layout()
        #plt.title('RMSE - Avg. of {} Replicates'.format(totrep), fontsize=20)
        fig.savefig(fname)

        fname = 'avg_EV.pdf'
        ax2.errorbar(np.array(range(1, len(ev)+1)), ev, yerr=ev_std, fmt=pltform[clr_cnt], c=clrs[clr_cnt], capsize=5, elinewidth=2, markeredgewidth=1, label='{}'.format(s))
        ax2.legend(loc=3, fontsize=20)
        ax2.set_ylabel('Excursion Set Variance [-]')
        ax2.set_xlabel('Step [-]')
        fig2.tight_layout()
        #plt.title('Reduction in excursion set - Avg. of {} Replicates'.format(totrep), fontsize=20)
        fig2.savefig(fname)

        fname = 'avg_R2.pdf'
        ax3.errorbar(np.array(range(1, len(r2)+1)), r2, yerr=r2_std, fmt=pltform[clr_cnt], c=clrs[clr_cnt], capsize=5, elinewidth=2, markeredgewidth=1, label='{}'.format(s))
        ax3.legend(loc=2, fontsize=20)
        ax3.set_ylabel('Explained Variance [%]')
        ax3.set_xlabel('Step [-]')
        fig3.tight_layout()
        #plt.title('R2 (explained variance) - Avg. of {} Replicates'.format(totrep), fontsize=20)
        fig3.savefig(fname)

        fname = 'avg_Time.pdf'

        ax4.plot(rt, c=clrs[clr_cnt], label='{}'.format(s))
        ax4.set_ylabel('Time [s]')
        ax4.set_xlabel('Step [-]')
        plt.xticks(np.arange(2, 11, 2))
        #ax4.errorbar(np.array(range(0, len(rt))), rt, yerr=np.std(rt, axis=0), fmt=pltform[clr_cnt], c=clrs[clr_cnt], capsize=5, elinewidth=2, markeredgewidth=1)
        ax4.legend(loc=2, fontsize=20)
        #plt.title('Run times for strategies - Avg. of {} Replicates'.format(totrep), fontsize=20)
        fig4.tight_layout()
        fig4.savefig(fname)

        clr_cnt += 1
#
#     fname = str(my_path + '/Evaluation_results/plots/EV_avg.pdf')
#     fig2, ax = plt.subplots(figsize=(fig_width, fig_height))
#     clr_cnt = 0
#     for s in strategies:
#         s_mean = np.mean([results[s][r]['EV'] for r in range(0, replicates)], axis=0)
#         s_std = np.std([results[s][r]['EV'] for r in range(0, replicates)], axis=0)
#         ax.plot(s_mean, c=clrs[clr_cnt], label='{}'.format(s))
#         #ax.fill_between(np.array(range(0, len(s_mean))), s_mean - s_std, s_mean + s_std, alpha=0.3, edgecolor='k', facecolor=clrs[clr_cnt])
#         ax.errorbar(np.array(range(0, len(s_mean))), s_mean, yerr=s_std, fmt='o', c=clrs[clr_cnt])
#         clr_cnt += 1
#     ax.legend(loc=3, fontsize=20)
#     plt.title('Reduction in excursion set - Avg. of {} Replicates'.format(replicates), fontsize=20)
#     plt.savefig(fname)
#
#     fname = str(my_path + '/Evaluation_results/plots/RunTimes_avg.pdf')
#     plt.figure(figsize=(fig_width, fig_height))
#     fig3, ax = plt.subplots(figsize=(fig_width, fig_height))
#     clr_cnt = 0
#     for s in strategies:
#         s_mean = np.mean([results[s][r]['run_time'] for r in range(0, replicates)], axis=0)
#         s_std = np.std([results[s][r]['run_time'] for r in range(0, replicates)], axis=0)
#         ax.plot(s_mean, c=clrs[clr_cnt], label='{}'.format(s))
#         #ax.fill_between(np.array(range(0, len(s_mean))), s_mean - s_std, s_mean + s_std, alpha=0.3, edgecolor='k', facecolor=clrs[clr_cnt])
#         ax.errorbar(np.array(range(0, len(s_mean))), s_mean, yerr=s_std, fmt='o', c=clrs[clr_cnt])
#         clr_cnt += 1
#     ax.legend(loc=2, fontsize=20)
#     plt.title('Run times for strategies - Avg. of {} Replicates'.format(replicates), fontsize=20)
#     plt.savefig(fname)
#
#     fname = str(my_path + '/Evaluation_results/plots/R2_avg.pdf')
#     fig4, ax = plt.subplots(figsize=(fig_width, fig_height))
#     clr_cnt = 0
#     for s in strategies:
#         s_mean = np.mean([results[s][r]['R2'] for r in range(0, replicates)], axis=0)
#         s_std = np.std([results[s][r]['R2'] for r in range(0, replicates)], axis=0)
#         ax.plot(s_mean, c=clrs[clr_cnt], label='{}'.format(s))
#         #ax.fill_between(np.array(range(0, len(s_mean))), s_mean - s_std, s_mean + s_std, alpha=0.3, edgecolor='k', facecolor=clrs[clr_cnt])
#         ax.errorbar(np.array(range(0, len(s_mean))), s_mean, yerr=s_std, fmt='o', c=clrs[clr_cnt])
#         clr_cnt += 1
#     ax.legend(loc=2, fontsize=20)
#     plt.title('R2 (explained variance) - Avg. of {} Replicates'.format(replicates), fontsize=20)
#     plt.savefig(fname)
#
# elif plot_mode is 'individual':
#
#     for r in range(0, 7):
#         fname = str(my_path + '/Evaluation_results/plots/RMSE_{}.pdf'.format(r))
#         plt.figure(figsize=(fig_width, fig_height))
#         for s in strategies:
#             plt.plot(results[s][r]['RMSE'], label='{}'.format(s))
#         plt.legend()
#         plt.title('RMSE')
#         plt.savefig(fname)
#
#         fname = str(my_path + '/Evaluation_results/plots/EV_{}.pdf'.format(r))
#         plt.figure(figsize=(fig_width, fig_height))
#         for s in strategies:
#             plt.plot(results[s][r]['EV'], label='{}'.format(s))
#         plt.legend()
#         plt.title('Reduction in excursion set')
#         plt.savefig(fname)
#
#         fname = str(my_path + '/Evaluation_results/plots/RunTimes_{}.pdf'.format(r))
#         plt.figure(figsize=(fig_width, fig_height))
#         for s in strategies:
#             plt.plot(results[s][r]['run_time'], label='{}'.format(s))
#         plt.legend()
#         plt.title('Run times for strategies')
#         plt.savefig(fname)
#
#         fname = str(my_path + '/Evaluation_results/plots/R2_{}.pdf'.format(r))
#         plt.figure(figsize=(fig_width, fig_height))
#         for s in strategies:
#             plt.plot(results[s][r]['R2'], label='{}'.format(s))
#         plt.legend()
#         plt.title('R2 (explained variance)')
#         plt.savefig(fname)
#
# else:
#     print('[ERROR] - Plot mode not set')
#

plt.close('all')
print('# == Script Finished == #')
