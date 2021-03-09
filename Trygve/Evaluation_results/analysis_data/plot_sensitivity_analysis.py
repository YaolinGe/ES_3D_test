import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from es_cases import *

plt.ioff()  # Running plt.ioff() - plots are kept in background, plt.ion() plots are shown as they are generated
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 27})
plt.rcParams.update({'font.style': 'oblique'})

strategies = ['naive', 'myopic', 'look-ahead', 'static_north', 'static_east', 'static_zigzag']

clrs = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

print('Unloading previous simulation and plotting data to files')


sim_configs = [std_low, std_high,
               cor_low, cor_high,
               ts_cor_low, ts_cor_high, t_only]

def redef(res):
    if res == 'cor_low':
        return 'cor_low 0.8'
    if res == 'cor_high':
        return 'cor_high 0.2'
    if res == 'ts_cor_high':
        return 'ts_cor_high 0.8'
    if res == 'ts_cor_low':
        return 'ts_cor_low 0.2'
    if res == 'std_high':
        return 'std_high 0.5'
    if res == 'std_low':
        return 'std_low 0.1'
    if res == 't_only':
        return 'temp. only'
    if res == 'basecase':
        return 'basecase'

analysis = ['analysis_ts_cor_oc.pickle', 'analysis_std_oc.pickle', 'analysis_cor_oc.pickle', 'analysis_t_only_oc.pickle', 'analysis_basecase_oc.pickle']

rmse = []
r2 = []
ev = []
rt = []
counter = 0
acounter = 0
conv_conter = [0, 2, 4, 6, 7]
replicates = 10
pltform = ['-o', '-o', '-o', '-o', '-o', '-o', '-o', '-o', '-o']
clr_cnt = 0

final_results_rmse = np.zeros((6, 8))
final_results_rmse_std = np.zeros((6, 8))
final_results_ev = np.zeros((6, 8))
final_results_ev_std = np.zeros((6, 8))
final_results_r2 = np.zeros((6, 8))
final_results_r2_std = np.zeros((6, 8))

# == Plot the ensemble in the ensemble figure
plt.figure(100, figsize=(40, 40))

plot_count = 0

for a in analysis:
    strat_count = 0
    for s in strategies:
        clr_cnt = conv_conter[acounter]
        results = pickle.load(open("{}".format(a), "rb"), encoding='latin1')

        intermediate_results_r2_std = []
        intermediate_results_r2 = []
        intermediate_results_ev_std = []
        intermediate_results_ev = []
        intermediate_results_rmse_std = []
        intermediate_results_rmse = []

        for res in results:

            res_label = redef(res)
            print('Loading config: {} - {}'.format(res, res_label))

            rmse = np.array([results[res][0][s][r]['RMSE'] for r in range(0, replicates)])
            r2 = np.array([results[res][0][s][r]['R2'] for r in range(0, replicates)])
            ev = np.array([results[res][0][s][r]['EV'] for r in range(0, replicates)])
            rt = np.array([results[res][0][s][r]['run_time'] for r in range(0, replicates)])

            # plt.figure()
            # plt.plot(t)
            # plt.plot(np.mean(np.array(t), axis=1), 'k')
            # plt.plot(np.std(t, axis=1), 'r')
            # plt.show()

            rmse_std = np.std(rmse, axis=0)[0:]
            ev_std = np.std(ev, axis=0)[0:]
            r2_std = np.std(r2, axis=0)[0:]

            rmse = np.mean(rmse, axis=0)[0:]
            ev = np.mean(ev, axis=0)[0:]
            r2 = np.mean(r2, axis=0)[0:]
            rt = [0.0] + np.mean(rt, axis=0)[0:10].tolist()

            intermediate_results_rmse.append(rmse[-1])
            intermediate_results_ev.append(ev[-1])
            intermediate_results_r2.append(r2[-1])

            intermediate_results_rmse_std.append(rmse_std[-1])
            intermediate_results_ev_std.append(ev_std[-1])
            intermediate_results_r2_std.append(r2_std[-1])

            plot_mode = 'avg'  # choose bet. 'avg' or 'individual'

            if plot_mode is 'avg':

                if s == 'naive':

                    plot_count = 0
                    plot_count += 1
                    plt.subplot(6, 4, plot_count)
                    plt.errorbar(np.array(range(1, len(rmse) + 1)), rmse, yerr=rmse_std, fmt=pltform[clr_cnt],
                                c=clrs[clr_cnt], capsize=5, elinewidth=2, markeredgewidth=1, label='{}'.format(res_label))
                    plt.title('Naive RMSE')
                    plt.ylabel('RMSE [C]')
                    plt.xlabel('Step [-]')
                    plt.legend(loc=3, fontsize=12)

                    plot_count += 1
                    plt.subplot(6, 4, plot_count)
                    plt.errorbar(np.array(range(1, len(ev) + 1)), ev, yerr=ev_std, fmt=pltform[clr_cnt],
                                c=clrs[clr_cnt], capsize=5, elinewidth=2, markeredgewidth=1, label='{}'.format(res_label))
                    plt.title('Naive EV')
                    plt.ylabel('Excursion Set Variance [-]')
                    plt.xlabel('Step [-]')
                    plt.legend(loc=3, fontsize=12)

                    plot_count += 1
                    plt.subplot(6, 4, plot_count)
                    plt.errorbar(np.array(range(1, len(r2) + 1)), r2, yerr=r2_std, fmt=pltform[clr_cnt],
                                c=clrs[clr_cnt], capsize=5, elinewidth=2, markeredgewidth=1, label='{}'.format(res_label))
                    plt.title('Naive R2')
                    plt.ylabel('Explained Variance [%]')
                    plt.xlabel('Step [-]')
                    plt.legend(loc=3, fontsize=12)

                    plot_count += 1
                    plt.subplot(6, 4, plot_count)
                    plt.plot(rt, c=clrs[clr_cnt], label='{}'.format(res_label))
                    plt.title('Naive run time')
                    plt.ylabel('Time [s]')
                    plt.xlabel('Step [-]')
                    plt.xticks(np.arange(2, 11, 2))
                    plt.legend(loc=2, fontsize=12)

                    clr_cnt += 1
                    plot_count = 0

                if s == 'look-ahead':

                    plot_count = 4
                    plot_count += 1
                    plt.subplot(6, 4, plot_count)
                    plt.errorbar(np.array(range(1, len(rmse) + 1)), rmse, yerr=rmse_std, fmt=pltform[clr_cnt],
                                 c=clrs[clr_cnt], capsize=5, elinewidth=2, markeredgewidth=1, label='{}'.format(res_label))
                    plt.title('Look-ahead RMSE')
                    plt.ylabel('RMSE [C]')
                    plt.xlabel('Step [-]')
                    plt.legend(loc=3, fontsize=12)

                    plot_count += 1
                    plt.subplot(6, 4, plot_count)
                    plt.errorbar(np.array(range(1, len(ev) + 1)), ev, yerr=ev_std, fmt=pltform[clr_cnt],
                                 c=clrs[clr_cnt], capsize=5, elinewidth=2, markeredgewidth=1, label='{}'.format(res_label))
                    plt.title('Look-ahead EV')
                    plt.ylabel('Excursion Set Variance [-]')
                    plt.xlabel('Step [-]')
                    plt.legend(loc=3, fontsize=12)

                    plot_count += 1
                    plt.subplot(6, 4, plot_count)
                    plt.errorbar(np.array(range(1, len(r2) + 1)), r2, yerr=r2_std, fmt=pltform[clr_cnt],
                                 c=clrs[clr_cnt], capsize=5, elinewidth=2, markeredgewidth=1, label='{}'.format(res_label))
                    plt.title('Look-ahead R2')
                    plt.ylabel('Explained Variance [%]')
                    plt.xlabel('Step [-]')
                    plt.legend(loc=3, fontsize=12)

                    plot_count += 1
                    plt.subplot(6, 4, plot_count)
                    plt.plot(rt, c=clrs[clr_cnt], label='{}'.format(res_label))
                    plt.title('Look-ahead run time')
                    plt.ylabel('Time [s]')
                    plt.xlabel('Step [-]')
                    plt.xticks(np.arange(2, 11, 2))
                    plt.legend(loc=2, fontsize=12)

                    clr_cnt += 1
                    plot_count = 4

                if s == 'myopic':
                    plot_count = 8
                    plot_count += 1
                    plt.subplot(6, 4, plot_count)
                    plt.errorbar(np.array(range(1, len(rmse) + 1)), rmse, yerr=rmse_std, fmt=pltform[clr_cnt],
                                 c=clrs[clr_cnt], capsize=5, elinewidth=2, markeredgewidth=1, label='{}'.format(res_label))
                    plt.title('Myopic RMSE')
                    plt.ylabel('RMSE [C]')
                    plt.xlabel('Step [-]')
                    plt.legend(loc=3, fontsize=12)

                    plot_count += 1
                    plt.subplot(6, 4, plot_count)
                    plt.errorbar(np.array(range(1, len(ev) + 1)), ev, yerr=ev_std, fmt=pltform[clr_cnt],
                                 c=clrs[clr_cnt], capsize=5, elinewidth=2, markeredgewidth=1, label='{}'.format(res_label))
                    plt.title('Myopic EV')
                    plt.ylabel('Excursion Set Variance [-]')
                    plt.xlabel('Step [-]')
                    plt.legend(loc=3, fontsize=12)

                    plot_count += 1
                    plt.subplot(6, 4, plot_count)
                    plt.errorbar(np.array(range(1, len(r2) + 1)), r2, yerr=r2_std, fmt=pltform[clr_cnt],
                                 c=clrs[clr_cnt], capsize=5, elinewidth=2, markeredgewidth=1, label='{}'.format(res_label))
                    plt.title('Myopic R2')
                    plt.ylabel('Explained Variance [%]')
                    plt.xlabel('Step [-]')
                    plt.legend(loc=3, fontsize=12)

                    plot_count += 1
                    plt.subplot(6, 4, plot_count)
                    plt.plot(rt, c=clrs[clr_cnt], label='{}'.format(res_label))
                    plt.title('Myopic run time')
                    plt.ylabel('Time [s]')
                    plt.xlabel('Step [-]')
                    plt.xticks(np.arange(2, 11, 2))
                    plt.legend(loc=2, fontsize=12)

                    clr_cnt += 1
                    plot_count = 8

                if s == 'static_north':
                    plot_count = 12
                    plot_count += 1
                    plt.subplot(6, 4, plot_count)
                    plt.errorbar(np.array(range(1, len(rmse) + 1)), rmse, yerr=rmse_std, fmt=pltform[clr_cnt],
                                 c=clrs[clr_cnt], capsize=5, elinewidth=2, markeredgewidth=1, label='{}'.format(res_label))
                    plt.title('Static North RMSE')
                    plt.ylabel('RMSE [C]')
                    plt.xlabel('Step [-]')
                    plt.legend(loc=3, fontsize=12)

                    plot_count += 1
                    plt.subplot(6, 4, plot_count)
                    plt.errorbar(np.array(range(1, len(ev) + 1)), ev, yerr=ev_std, fmt=pltform[clr_cnt],
                                 c=clrs[clr_cnt], capsize=5, elinewidth=2, markeredgewidth=1, label='{}'.format(res_label))
                    plt.title('Static North EV')
                    plt.ylabel('Excursion Set Variance [-]')
                    plt.xlabel('Step [-]')
                    plt.legend(loc=3, fontsize=12)

                    plot_count += 1
                    plt.subplot(6, 4, plot_count)
                    plt.errorbar(np.array(range(1, len(r2) + 1)), r2, yerr=r2_std, fmt=pltform[clr_cnt],
                                 c=clrs[clr_cnt], capsize=5, elinewidth=2, markeredgewidth=1, label='{}'.format(res_label))
                    plt.title('Static North R2')
                    plt.ylabel('Explained Variance [%]')
                    plt.xlabel('Step [-]')
                    plt.legend(loc=3, fontsize=12)

                    plot_count += 1
                    plt.subplot(6, 4, plot_count)
                    plt.plot(rt, c=clrs[clr_cnt], label='{}'.format(res_label))
                    plt.title('Static North run time')
                    plt.ylabel('Time [s]')
                    plt.xlabel('Step [-]')
                    plt.xticks(np.arange(2, 11, 2))
                    plt.legend(loc=2, fontsize=12)

                    clr_cnt += 1
                    plot_count = 12

                if s == 'static_east':

                    plot_count = 16
                    plot_count += 1
                    plt.subplot(6, 4, plot_count)
                    plt.errorbar(np.array(range(1, len(rmse) + 1)), rmse, yerr=rmse_std, fmt=pltform[clr_cnt],
                                 c=clrs[clr_cnt], capsize=5, elinewidth=2, markeredgewidth=1, label='{}'.format(res_label))
                    plt.title('Static East RMSE')
                    plt.ylabel('RMSE [C]')
                    plt.xlabel('Step [-]')
                    plt.legend(loc=3, fontsize=12)

                    plot_count += 1
                    plt.subplot(6, 4, plot_count)
                    plt.errorbar(np.array(range(1, len(ev) + 1)), ev, yerr=ev_std, fmt=pltform[clr_cnt],
                                 c=clrs[clr_cnt], capsize=5, elinewidth=2, markeredgewidth=1, label='{}'.format(res_label))
                    plt.title('Static East EV')
                    plt.ylabel('Excursion Set Variance [-]')
                    plt.xlabel('Step [-]')
                    plt.legend(loc=3, fontsize=12)

                    plot_count += 1
                    plt.subplot(6, 4, plot_count)
                    plt.errorbar(np.array(range(1, len(r2) + 1)), r2, yerr=r2_std, fmt=pltform[clr_cnt],
                                 c=clrs[clr_cnt], capsize=5, elinewidth=2, markeredgewidth=1, label='{}'.format(res_label))
                    plt.title('Static East R2')
                    plt.ylabel('Explained Variance [%]')
                    plt.xlabel('Step [-]')
                    plt.legend(loc=3, fontsize=12)

                    plot_count += 1
                    plt.subplot(6, 4, plot_count)
                    plt.plot(rt, c=clrs[clr_cnt], label='{}'.format(res_label))
                    plt.title('Static East run time')
                    plt.ylabel('Time [s]')
                    plt.xlabel('Step [-]')
                    plt.xticks(np.arange(2, 11, 2))
                    plt.legend(loc=2, fontsize=12)

                    clr_cnt += 1
                    plot_count = 16

                if s == 'static_zigzag':
                    plot_count = 20
                    plot_count += 1
                    plt.subplot(6, 4, plot_count)
                    plt.errorbar(np.array(range(1, len(rmse) + 1)), rmse, yerr=rmse_std, fmt=pltform[clr_cnt],
                                 c=clrs[clr_cnt], capsize=5, elinewidth=2, markeredgewidth=1, label='{}'.format(res_label))
                    plt.title('Static ZigZag RMSE')
                    plt.ylabel('RMSE [C]')
                    plt.xlabel('Step [-]')
                    plt.legend(loc=3, fontsize=12)

                    plot_count += 1
                    plt.subplot(6, 4, plot_count)
                    plt.errorbar(np.array(range(1, len(ev) + 1)), ev, yerr=ev_std, fmt=pltform[clr_cnt],
                                 c=clrs[clr_cnt], capsize=5, elinewidth=2, markeredgewidth=1, label='{}'.format(res_label))
                    plt.title('Static ZigZag EV')
                    plt.ylabel('Excursion Set Variance [-]')
                    plt.xlabel('Step [-]')
                    plt.legend(loc=3, fontsize=12)

                    plot_count += 1
                    plt.subplot(6, 4, plot_count)
                    plt.errorbar(np.array(range(1, len(r2) + 1)), r2, yerr=r2_std, fmt=pltform[clr_cnt],
                                 c=clrs[clr_cnt], capsize=5, elinewidth=2, markeredgewidth=1, label='{}'.format(res_label))
                    plt.title('Static ZigZag R2')
                    plt.ylabel('Explained Variance [%]')
                    plt.xlabel('Step [-]')
                    plt.legend(loc=3, fontsize=12)

                    plot_count += 1
                    plt.subplot(6, 4, plot_count)
                    plt.plot(rt, c=clrs[clr_cnt], label='{}'.format(res_label))
                    plt.title('Static ZigZag run time')
                    plt.ylabel('Time [s]')
                    plt.xlabel('Step [-]')
                    plt.xticks(np.arange(2, 11, 2))
                    plt.legend(loc=2, fontsize=12)

                    clr_cnt += 1
                    plot_count = 20

        if s not in ['t_only', 'basecase']:
            final_results_rmse[strat_count, conv_conter[acounter]:conv_conter[acounter]+2] = intermediate_results_rmse
            final_results_rmse_std[strat_count, conv_conter[acounter]:conv_conter[acounter]+2] = intermediate_results_rmse_std
            final_results_ev[strat_count, conv_conter[acounter]:conv_conter[acounter]+2] = intermediate_results_ev
            final_results_ev_std[strat_count, conv_conter[acounter]:conv_conter[acounter]+2] = intermediate_results_ev_std
            final_results_r2[strat_count, conv_conter[acounter]:conv_conter[acounter]+2] = intermediate_results_r2
            final_results_r2_std[strat_count, conv_conter[acounter]:conv_conter[acounter]+2] = intermediate_results_r2_std
        else:
            final_results_rmse[strat_count, conv_conter[acounter]:conv_conter[acounter] + 1] = intermediate_results_rmse
            final_results_rmse_std[strat_count, conv_conter[acounter]:conv_conter[acounter] + 1] = intermediate_results_rmse_std
            final_results_ev[strat_count, conv_conter[acounter]:conv_conter[acounter] + 1] = intermediate_results_ev
            final_results_ev_std[strat_count, conv_conter[acounter]:conv_conter[acounter] + 1] = intermediate_results_ev_std
            final_results_r2[strat_count, conv_conter[acounter]:conv_conter[acounter] + 1] = intermediate_results_r2
            final_results_r2_std[strat_count, conv_conter[acounter]:conv_conter[acounter] + 1] = intermediate_results_r2_std
        strat_count += 1
    acounter += 1

plt.tight_layout()
plt.savefig('Analysis.pdf')
plt.close('all')

rownames = ['ts. cor. low: 0.2',
            'ts. cor. high: 0.8',
            'std. low: 0.1',
            'std. high: 0.5',
            'cor. low: 0.8',
            'cor. high: 0.2',
            'temp. only',
            'basecase']

df_rmse = pd.DataFrame(final_results_rmse.T, columns=strategies, index=rownames)
df_rmse_std = pd.DataFrame(final_results_rmse_std.T, columns=strategies, index=rownames)
df_ev = pd.DataFrame(final_results_ev.T, columns=strategies, index=rownames)
df_ev_std = pd.DataFrame(final_results_ev_std.T, columns=strategies, index=rownames)
df_r2 = pd.DataFrame(final_results_r2.T, columns=strategies, index=rownames)
df_r2_std = pd.DataFrame(final_results_r2_std.T, columns=strategies, index=rownames)

print(df_rmse.to_latex(float_format='%.2f'))
print(df_ev.to_latex(float_format='%.2f'))
print(df_r2.to_latex(float_format='%.2f'))

print('# == Script Finished == #')
