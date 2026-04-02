# -*- coding: utf-8 -*-
"""
@author: erdinck
"""

"""
ENHANCED MULTIPLICATIVE WINTERS' EXPONENTIAL SMOOTHING (WES) FOR SEASONAL FORECASTING

DATASET DESCRIPTION:
The source data includes specific time series indices from the R 'tsdl' (Time Series Data Library) package.
Targeted Indices: 9, 16, 74, 117, 153, 232, 262, 401, 455, and 460.
- All included datasets represent seasonal time series with a constant period (m = 12).
- Data is processed as Multiplicative Time Series, where seasonal variations change in proportion to the level.
Link: https://github.com/FinYang/tsdl 

OBJECTIVE:
This study aims to enhance the forecasting performance of the Multiplicative WES method by introducing:
1. BACK-CASTING: Used for accurate initialization of Level, Trend, and Seasonal components by reversing the 
   series to find optimal starting points.
2. GREY WOLF OPTIMIZATION (GWO): An automated heuristic algorithm used for optimal tuning of smoothing 
   parameters (alpha, beta, gamma).
3. SMOOTHED ERROR TRACKING SIGNAL (SETS): An adaptive bias correction mechanism that monitors 
   forecast errors in real-time and triggers model re-tuning if bias exceeds the defined threshold.

METRICS:
Performance is evaluated using RMSE, MAE, MAPE, and sMAPE across multiple training lengths.
"""

import numpy as np
import pandas as pd
import random
import itertools
import os
import glob
import time
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# ==========================================
# GLOBAL CONFIGURATION
# ==========================================
PERIOD = 12  # Seasonality cycle (e.g., 12 months)
SEARCH_AGENTS_NO = 10  # Number of Grey Wolves
MAX_ITER = 50  # Maximum iterations for GWO
THRESHOLD_VALUE = 0.6  # SETS bias threshold for re-tuning
ALPHA_VALUES = [0.05, 0.10, 0.15] # Smoothing parameters for error tracking
FOLDER_PATH = "Data_folder" # Folder containing tsdl indices: 9, 16, 74, etc.

# ==========================================
# PERFORMANCE METRICS
# ==========================================

def calculate_mape(actual, predicted):
    """Mean Absolute Percentage Error"""
    actual, predicted = np.array(actual), np.array(predicted)
    return np.mean(np.abs((actual - predicted) / actual)) * 100

def calculate_smape(actual, forecast):
    """Symmetric Mean Absolute Percentage Error"""
    return 1/len(actual) * np.sum(2 * np.abs(forecast - actual) / (np.abs(actual) + np.abs(forecast)) * 100)

def rmse(y_true, y_pred):
    """Root Mean Squared Error"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# ==========================================
# OPTIMIZATION COMPONENTS (GWO & OBL)
# ==========================================

def chaotic_mapping(num_agents, dim):
    """Chaotic initialization using Logistic Map for better search space coverage"""
    chaotic_map = np.zeros((num_agents, dim))
    x0 = np.random.rand(dim)
    r = 4.0
    for i in range(num_agents):
        x0 = r * x0 * (1 - x0)
        chaotic_map[i, :] = x0
    return chaotic_map

def opposition_based_learning(num_agents, dim, lower_bound, upper_bound):
    """Enhances search efficiency by considering opposite positions"""
    positions = chaotic_mapping(num_agents, dim)
    positions = lower_bound + (upper_bound - lower_bound) * positions
    opposite_positions = lower_bound + upper_bound - positions
    combined_positions = np.vstack((positions, opposite_positions))
    
    fitness = np.array([objective_function(pos) for pos in combined_positions])
    indices = np.argsort(fitness)[:num_agents]
    return combined_positions[indices]

# ==========================================
# WINTERS' CORE LOGIC
# ==========================================

def holt_winters_backcasting(y, alpha, beta, gamma, m):
    """
    Back-casting implementation: Runs the model backwards to stabilize
    initial Level, Trend, and Seasonal components.
    """
    n = len(y)
    l, b = np.zeros(n + 1), np.zeros(n + 1)
    s = np.zeros(n + m)
    y_hat = np.zeros(n + 1)

    # Linear regression to estimate initial slopes for seasonality
    intercepts, coefs = np.array([]), np.array([])
    for i in range(m):
        indices = np.arange(i, len(y), m)
        X_sub = y[indices].reshape((indices.size, 1))
        z_sub = (indices + 1).reshape((indices.size, 1))
        model = LinearRegression().fit(z_sub, X_sub)
        intercepts = np.append(intercepts, model.intercept_)
        coefs = np.append(coefs, model.coef_)
        
    cycle = itertools.cycle(np.arange(1, m + 1))
    vector = list(itertools.islice(cycle, len(y)))
    last_m_rev = np.array(vector[-m:][::-1])

    mean_coef = np.mean(coefs)
    sum_mean = mean_coef + np.mean(intercepts)

    # Multiplicative seasonal factors
    ss = np.array([(intercepts[i] + coefs[i]) / sum_mean for i in range(m)])
    initial_s = ss[last_m_rev - 1]
    y_rev = y[::-1] # Reverse series for backcasting

    l[0], b[0] = y_rev[0], -1 * mean_coef
    s[:len(initial_s)] = initial_s

    # Backcasting iterations
    for t in range(1, n + 1):
        l[t] = alpha * (y_rev[t-1] / s[t-1]) + (1 - alpha) * (l[t-1] + b[t-1])
        b[t] = beta * (l[t] - l[t-1]) + (1 - beta) * b[t-1]
        s[t+m-1] = gamma * (y_rev[t-1] / l[t]) + (1 - gamma) * s[t-1]
        y_hat[t] = (l[t-1] + b[t-1]) * s[t-1]
   
    last_m_s = s[len(y):len(y)+m]
    mean_last_m_s = np.mean(last_m_s)
    
    return {
        'Level': (l[-1] + b[-1]) * mean_last_m_s,
        'Trend': b[-1] * mean_last_m_s * -1,
        'Seasonal': (last_m_s[::-1]) / mean_last_m_s,
        'Predictions': y_hat[1:]
    }

def holt_winters_forecasting(y, alpha, beta, gamma, m, n_preds, initial_level, initial_trend, initial_s):
    """Standard Multiplicative Holt-Winters forecasting logic"""
    n = len(y)
    l, b = np.zeros(n + n_preds + 1), np.zeros(n + n_preds + 1)
    s = np.zeros(n + m + n_preds)
    y_hat = np.zeros(n + n_preds + 1)

    l[0], b[0], s[:len(initial_s)] = initial_level, initial_trend, initial_s

    for t in range(1, n + 1):
        l[t] = alpha * (y[t-1] / s[t-1]) + (1 - alpha) * (l[t-1] + b[t-1])
        b[t] = beta * (l[t] - l[t-1]) + (1 - beta) * b[t-1]
        s[t+m-1] = gamma * (y[t-1] / l[t]) + (1 - gamma) * s[t-1]
        y_hat[t] = (l[t-1] + b[t-1]) * s[t-1]
    
    # Projection into the future
    last_l, last_b, last_s = l[n], b[n], s[n:n+m]
    for idx, t in enumerate(range(n + 1, n + n_preds + 1)):
        y_hat[t] = (last_l + (t - n) * last_b) * last_s[idx % m]
            
    return {
        'Predictions': y_hat[1:n+1].tolist(),
        'Forecasts': y_hat[n+1:].tolist()
    }

# ==========================================
# OPTIMIZATION ENGINE
# ==========================================

def objective_function(params):
    """GWO Optimization Target: Minimize Mean Absolute Error (MAE) during initialization"""
    alpha, beta, gamma = params
    result = holt_winters_backcasting(train.values, alpha, beta, gamma, m=PERIOD)
    return mean_absolute_error(train.values[::-1], result['Predictions'])

def gwo(objf, lb, ub, dim, agents, iterations):
    """
    Grey Wolf Optimization Algorithm: Mimics leadership hierarchy and 
    hunting mechanism of grey wolves for parameter tuning.
    """
    lb, ub = np.array(lb), np.array(ub)
    alpha_pos, alpha_score = np.zeros(dim), float("inf") # Leader
    beta_pos, beta_score = np.zeros(dim), float("inf")   # Advisor
    delta_pos, delta_score = np.zeros(dim), float("inf") # Scout
    
    positions = opposition_based_learning(agents, dim, lb, ub)

    for l in range(iterations):
        for i in range(agents):
            positions[i, :] = np.clip(positions[i, :], lb, ub)
            fitness = objf(positions[i, :])
            
            # Update Alpha, Beta, and Delta positions
            if fitness < alpha_score:
                alpha_score, alpha_pos = fitness, positions[i, :].copy()
            elif fitness < beta_score:
                beta_score, beta_pos = fitness, positions[i, :].copy()
            elif fitness < delta_score:
                delta_score, delta_pos = fitness, positions[i, :].copy()

        a = 2 - l * (2 / iterations)
        for i in range(agents):
            for j in range(dim):
                # Wolf position update logic based on hierarchy
                r1, r2 = np.random.random(), np.random.random()
                A1, C1 = 2*a*r1-a, 2*r2; D_alpha = abs(C1*alpha_pos[j]-positions[i, j]); X1 = alpha_pos[j]-A1*D_alpha
                r1, r2 = np.random.random(), np.random.random()
                A2, C2 = 2*a*r1-a, 2*r2; D_beta = abs(C2*beta_pos[j]-positions[i, j]); X2 = beta_pos[j]-A2*D_beta
                r1, r2 = np.random.random(), np.random.random()
                A3, C3 = 2*a*r1-a, 2*r2; D_delta = abs(C3*delta_pos[j]-positions[i, j]); X3 = delta_pos[j]-A3*D_delta
                positions[i, j] = (X1 + X2 + X3) / 3
    return alpha_pos, alpha_score

# ==========================================
# ADAPTIVE ERROR TRACKING (SETS)
# ==========================================

def smoothed_error(errors, phi):
    s_err = np.zeros(len(errors))
    for i in range(1, len(errors)):
        s_err[i] = phi * errors[i-1] + (1 - phi) * s_err[i-1]
    return s_err

def mean_absolute_deviation(errors, phi):
    mad = np.zeros(len(errors))
    mad[0] = np.mean(np.abs(errors - np.mean(errors)))
    for i in range(1, len(errors)):
        mad[i] = phi * np.abs(errors[i-1]) + (1 - phi) * mad[i-1]
    return mad

def find_consecutive_indices(arr, threshold):
    """SETS Logic: Triggers if bias persists for 3 consecutive points"""
    return next((i for i in range(len(arr) - 2) if all(arr[i + j] > threshold for j in range(3))), -1)

# ==========================================
# MAIN EXECUTION FLOW
# ==========================================
start_time = time.perf_counter()
csv_files = sorted(glob.glob(os.path.join(FOLDER_PATH, "*.csv")))
df_result_metrics = pd.DataFrame()

# Loop through each dataset index from tsdl
for file in csv_files:
    print(f"Processing File: {file}")
    data_full = pd.read_csv(file, sep=",", decimal=".")
    
    # Testing different training split lengths
    for train_length in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        all_metrics = {k: pd.DataFrame() for k in ['RMSE', 'MAE', 'MAPE', 'SMAPE']}
        
        # Multiple iterations for statistical robustness
        for iterr in range(1, 11):
            rmse_list, mae_list, mape_list, smape_list = [], [], [], []
            
            for data_idx in range(1, data_full.shape[1]):
                series = data_full.iloc[:, data_idx]
                n = len(series)
                train_end, val_end = int(n * train_length), int(n * 0.9)
                
                train, val, test = series[:train_end], series[train_end:val_end], series[val_end:]
                train_val = series[:val_end]
                
                hyper_results = []
                # Tuning SETS tracking parameters
                for s_param in ALPHA_VALUES:
                    for mad_param in ALPHA_VALUES:
                        if mad_param <= s_param:
                            # Optimize smoothing factors (alpha, beta, gamma)
                            best_p, _ = gwo(objective_function, [0,0,0], [1,1,1], 3, SEARCH_AGENTS_NO, MAX_ITER)
                            a, b, g = best_p
                            
                            init = holt_winters_backcasting(train.values, a, b, g, m=PERIOD)
                            res = holt_winters_forecasting(train.values, a, b, g, PERIOD, len(val), 
                                                         init['Level'], init['Trend'], init['Seasonal'])
                            
                            forecasts, new_idx, total_forecasts = res['Forecasts'], 0, np.array([])
                            
                            # Sequential Error Tracking Signal (SETS) Loop
                            while True:
                                curr_val = val.values[new_idx:]
                                if len(curr_val) != len(forecasts):
                                    min_len = min(len(curr_val), len(forecasts))
                                    curr_val, forecasts = curr_val[:min_len], forecasts[:min_len]

                                errs = curr_val - forecasts
                                # Tracking Signal Calculation
                                ts = np.abs(smoothed_error(errs, s_param) / (mean_absolute_deviation(errs, mad_param) + 1e-10))
                                idx = find_consecutive_indices(ts, THRESHOLD_VALUE)
                                
                                if idx < 0:
                                    total_forecasts = np.append(total_forecasts, forecasts)
                                    break
                                
                                # If bias is detected, append current forecasts and re-optimize for remaining period
                                total_forecasts = np.append(total_forecasts, forecasts[:idx])
                                new_idx += idx
                                train_upd = train_val[new_idx:train_end + new_idx]
                                
                                best_p, _ = gwo(objective_function, [0,0,0], [1,1,1], 3, SEARCH_AGENTS_NO, MAX_ITER)
                                a, b, g = best_p
                                init_upd = holt_winters_backcasting(train_upd.values, a, b, g, m=PERIOD)
                                res = holt_winters_forecasting(train_upd.values, a, b, g, PERIOD, 
                                                             len(train_val) - (train_end + new_idx),
                                                             init_upd['Level'], init_upd['Trend'], init_upd['Seasonal'])
                                forecasts = res['Forecasts']

                            # Generate Final Test Forecasts
                            final_train = train_val[new_idx:]
                            best_p, _ = gwo(objective_function, [0,0,0], [1,1,1], 3, SEARCH_AGENTS_NO, MAX_ITER)
                            a, b, g = best_p
                            init_final = holt_winters_backcasting(final_train.values, a, b, g, m=PERIOD)
                            final_res = holt_winters_forecasting(final_train.values, a, b, g, PERIOD, len(test),
                                                               init_final['Level'], init_final['Trend'], init_final['Seasonal'])
                            
                            hyper_results.append({
                                'val_rmse': rmse(val.values, total_forecasts),
                                'test_rmse': rmse(test.values, final_res['Forecasts']),
                                'test_mae': mean_absolute_error(test.values, final_res['Forecasts']),
                                'test_mape': calculate_mape(test.values, final_res['Forecasts']),
                                'test_smape': calculate_smape(test.values, final_res['Forecasts'])
                            })
                
                # Selection of the best model based on validation performance
                best_h = min(hyper_results, key=lambda x: x['val_rmse'])
                rmse_list.append(best_h['test_rmse'])
                mae_list.append(best_h['test_mae'])
                mape_list.append(best_h['test_mape'])
                smape_list.append(best_h['test_smape'])

            all_metrics['RMSE'][all_metrics['RMSE'].shape[1]] = rmse_list
            all_metrics['MAE'][all_metrics['MAE'].shape[1]] = mae_list
            all_metrics['MAPE'][all_metrics['MAPE'].shape[1]] = mape_list
            all_metrics['SMAPE'][all_metrics['SMAPE'].shape[1]] = smape_list

        # Aggregate Results per File and Train Length
        file_name = os.path.splitext(os.path.basename(file))[0]
        summary = [all_metrics[k].mean(axis=1).iloc[-1] for k in ['RMSE', 'MAE', 'MAPE', 'SMAPE']]
        df_result_metrics[f"{file_name}_{train_length}"] = summary

print(f"Total Duration: {time.perf_counter() - start_time:.2f} seconds")