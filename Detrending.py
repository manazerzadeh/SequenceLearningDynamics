import pandas as pd
import numpy as np
from typing import List

from tqdm import tqdm
import setglobals as gl
import seaborn as sns
from matplotlib import ticker
from statsmodels.stats.anova import AnovaRM
from scipy.optimize import curve_fit


n_trials_per_block = 20
n_blocks_per_day = 20
n_trials_per_day = n_trials_per_block * n_blocks_per_day

"""
single exponential model: y = a * exp(b * T) + c
"""
def exp_model(T, a, b, c):
    return a * np.exp(b * T) + c


"""
slow + fast model: 

slow component: y_t = A * y_t-1 + B -> y_t = A^t * y0 + B * (1 - A^t) / (1 - A)
fast component: y_t = C * y_t-1 + D -> y_t = D * (1 - C^t) / (1 - C) (initial value set to 0)
total: y = slow + fast
"""
def fast_slow_model(T, A, init_slow, B, C, D):
    within_day_tn = (T-1) % (n_trials_per_day)  # trial number within the day
    slow_component = A ** T * init_slow + B / (1 - A) * (1 - A ** T)
    fast_component = D / (1 - C) * (1 - C ** within_day_tn)
    return slow_component + fast_component



def fit_model(data, subj, feature, model_name):
    subj_data = data[data['SubNum'] == subj]
    T = subj_data['T'].values
    feature_values = subj_data[feature].values

    # Initial guess for parameters
    if model_name == 'exp_model':
        initial_guess = [0,0,0]
    elif model_name == 'fast_slow_model':
        if feature == 'speed':
            initial_guess = [0.999, 1, 0.1, 0.999, 0.1]
            lower_bounds = [0.5, 0, 0, 0.5, 0]
            upper_bounds = [1, 10, 0.5, 1, 0.5]

        if feature == 'ET':
            initial_guess = [0.999, 1000, 50, 0.999, 50]
            lower_bounds = [0.5, 0, 0, 0.5, 0]
            upper_bounds = [1, 4000, 500, 1, 500]
        else:
            initial_guess = [0.999, 0, 0, 0.999, 0]
    
    model = globals()[model_name]
    # if lower bound and upper bound are defined
    if 'lower_bounds' in locals() and 'upper_bounds' in locals():
        params = curve_fit(model, T, feature_values, p0=initial_guess, bounds=(lower_bounds, upper_bounds), full_output=True, maxfev=50000)
    else:
        params = curve_fit(model, T, feature_values, p0=initial_guess, full_output=True, maxfev=50000)

    residuals = params[2]['fvec']
    noise_scale = np.sqrt(np.sum(residuals**2) / (len(feature_values) - 1))
    return params[0], noise_scale



def detrend(data):
    correct_data = data[data['isError'] == False]
    residual_data = data.copy()

    features = [col for col in data.columns if col.startswith('PC_')] + ['speed']

    for subj in data['SubNum'].unique():
        subj_data = data[data['SubNum'] == subj]
        T = subj_data['T'].values

        for feature in features:
            feature_values = subj_data[feature].values
            try:
                params, sigma = fit_model(correct_data, subj, feature, 'fast_slow_model')
                feature_fit = fast_slow_model(T, *params)
            except RuntimeError:
                # print(f"Could not fit double exp model for subject {subj}, feature {feature}")
                try: 
                    params, sigma = fit_model(correct_data, subj, feature, 'exp_model')
                    feature_fit = exp_model(T, *params)

                except RuntimeError:
                    # print(f"Could not fit exp model for subject {subj}, feature {feature}")
                    continue

            residuals = feature_values - feature_fit
            residual_data.loc[residual_data['SubNum'] == subj, feature] = residuals


    ### Now residual_data contains the residuals after removing the exponential trend
    residual_data['forceVector'] = residual_data[[col for col in residual_data.columns if col.startswith('PC_')]].apply(lambda x: np.array(x), axis=1)

    return residual_data

        

