# === Cases

trend_temp = [[5.8], [0.098]]
trend_sal = [[29.0], [0.133]]

std_low = {'spatial_corr': 0.3,
           'temp_sal_corr': 0.6,
           'std_dev': 0.1,
           'beta_t': trend_temp,
           'beta_s': trend_sal,
           'noise': [0.05, 0.05],
           'name': 'low std 0.1',
           'id': 'std_low'}

std_high = {'spatial_corr': 0.3,
            'temp_sal_corr': 0.6,
            'std_dev': 0.5,
            'beta_t': trend_temp,
            'beta_s': trend_sal,
            'noise': [0.05, 0.05],
            'name': 'high std 0.5',
            'id': 'std_high'}

cor_high = {'spatial_corr': 0.2,
            'temp_sal_corr': 0.6,
            'std_dev': 0.25,
            'beta_t': trend_temp,
            'beta_s': trend_sal,
            'noise': [0.05, 0.05],
            'name': 'high spatial corr. 0.2',
            'id': 'cor_high'}

cor_low = {'spatial_corr': 0.8,
           'temp_sal_corr': 0.6,
           'std_dev': 0.25,
           'beta_t': trend_temp,
           'beta_s': trend_sal,
           'noise': [0.05, 0.05],
           'name': 'low spatial corr. 0.8',
           'id': 'cor_low'}

ts_cor_low = {'spatial_corr': 0.3,
              'temp_sal_corr': 0.2,
              'std_dev': 0.25,
              'beta_t': trend_temp,
              'beta_s': trend_sal,
              'noise': [0.05, 0.05],
              'name': 'low ts corr. 0.2',
              'id': 'ts_cor_low'}

ts_cor_high = {'spatial_corr': 0.3,
               'temp_sal_corr': 0.8,
               'std_dev': 0.25,
               'beta_t': trend_temp,
               'beta_s': trend_sal,
               'noise': [0.05, 0.05],
               'name': 'high ts corr. 0.8',
               'id': 'ts_cor_high'}

t_only = {'spatial_corr': 0.3,  # Ca. 1500 m effective correlation
          'temp_sal_corr': 0.6,
          'std_dev': 0.25,
          'beta_t': trend_temp,
          'beta_s': trend_sal,
          'noise': [0.05, 0.05*10000],
          'name': 't_only',
          'id': 't_only'}

basecase = {'spatial_corr': 0.3,  # Ca. 1500 m effective correlation
            'temp_sal_corr': 0.6,
            'std_dev': 0.25,
            'beta_t': trend_temp,
            'beta_s': trend_sal,
            'noise': [0.05, 0.05],
            'name': 'basecase',
            'id': 'basecase'}