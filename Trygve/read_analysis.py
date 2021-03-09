import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.ioff()  # Running plt.ioff() - plots are kept in background, plt.ion() plots are shown as they are generated
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 26})

my_path = os.path.dirname(os.path.abspath(__file__))

strategies = ['look-ahead', 'myopic']

clrs = sns.color_palette("jet_r", len(strategies))

print('Unloading previous simulation and plotting data to files')

results = pickle.load(open("analysis.pickle", "rb"), encoding='latin1')

# plot_mode = 'avg'  # choose bet. 'avg' or 'individual'
#
# if plot_mode is 'avg':
#     fname = str(my_path + '/Evaluation_results/plots/RMSE_avg.pdf')
#     fig, ax = plt.subplots(figsize=(fig_width, fig_height))
#     clr_cnt = 0
#     for s in strategies:
#         s_mean = np.mean([results[s][r]['RMSE'] for r in range(0, replicates)], axis=0)
#         s_std = np.std([results[s][r]['RMSE'] for r in range(0, replicates)], axis=0)
#         ax.plot(s_mean, c=clrs[clr_cnt], label='{}'.format(s))
#         #ax.fill_between(np.array(range(0, len(s_mean))), s_mean - s_std, s_mean + s_std, alpha=0.3, edgecolor='k', facecolor=clrs[clr_cnt])
#         ax.errorbar(np.array(range(0, len(s_mean))), s_mean, yerr=s_std, fmt='o', c=clrs[clr_cnt])
#         clr_cnt += 1
#     ax.legend(loc=3, fontsize=20)
#     plt.title('RMSE - Avg. of {} Replicates'.format(replicates), fontsize=20)
#     plt.savefig(fname)
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
