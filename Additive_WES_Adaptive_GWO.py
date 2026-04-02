# -*- coding: utf-8 -*-
"""
@author: erdinck
"""

"""
ENHANCED ADDITIVE WINTERS' EXPONENTIAL SMOOTHING (WES) FOR TEMPERATURE FORECASTING

DATASET DESCRIPTION:
The source data 'regional_averages_2024_full.csv' consists of 19 columns:
- Column 1: Year
- Column 2: Month
- Columns 3-18: Average monthly air temperatures for the 16 German Federal States (Bundesländer).
- Column 19: National average air temperature for Germany (Regional Average).

Data Source: German Weather Service (DWD) - CDC (Climate Data Center)
Link: https://opendata.dwd.de/climate_environment/CDC/regional_averages_DE/monthly/air_temperature_mean/

OBJECTIVE:
This study enhances the forecasting performance of the Additive WES method by introducing:
1. Back-casting for accurate initialization of Level, Trend, and Seasonal components.
2. Grey Wolf Optimization (GWO) algorithm for automated parameter tuning (alpha, beta, gamma).
3. Smoothed Error Tracking Signal (SETS) for adaptive bias correction and model re-tuning.
"""

import numpy as np
import pandas as pd
import itertools
import time
import random
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error

# ==========================================
# GLOBAL CONFIGURATION
# ==========================================
PERIOD = 12  # Seasonal cycle (Monthly)
SEARCH_AGENTS_NO = 10  # Population size for Grey Wolves
MAX_ITER = 100  # Max iterations for the GWO optimizer
THRESHOLD_VALUE = 0.6  # SETS threshold for triggering model re-tuning
ALPHA_VALS = [0.05, 0.10, 0.15]  # Parameters for the Error Tracking Signal

# ==========================================
# PERFORMANCE METRICS
# ==========================================

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
    """Chaotic initialization using Logistic Map to avoid local optima"""
    chaotic_map = np.zeros((num_agents, dim))
    x0 = np.random.rand(dim)
    r = 4.0
    for i in range(num_agents):
        x0 = r * x0 * (1 - x0)
        chaotic_map[i, :] = x0
    return chaotic_map

def opposition_based_learning(num_agents, dim, lower_bound, upper_bound):
    """Generates opposite agents to broaden the search space coverage"""
    positions = chaotic_mapping(num_agents, dim)
    positions = lower_bound + (upper_bound - lower_bound) * positions
    opposite_positions = lower_bound + upper_bound - positions
    combined_positions = np.vstack((positions, opposite_positions))
    
    fitness = np.array([objective_function(pos) for pos in combined_positions])
    indices = np.argsort(fitness)[:num_agents]
    return combined_positions[indices]

# ==========================================
# WINTERS' CORE LOGIC (ADDITIVE)
# ==========================================

def holt_winters_backcasting(y, alpha, beta, gamma, m):
    """
    Additive Back-casting: Runs WES backwards on the reversed series
    to determine stable starting points for the actual forward pass.
    """
    n = len(y)
    l, b = np.zeros(n + 1), np.zeros(n + 1)
    s = np.zeros(n + m)
    y_hat = np.zeros(n + 1)

    # Initial Seasonal estimation using linear dummy variables
    dummies = []
    for i in range(m - 1):
        row = [0] * (m - 1); row[i] = 1; dummies.append(row)
    dummies.append([0] * (m - 1))
    
    repeated_dummies = dummies * (n // m) + dummies[:n % m]
    df = pd.DataFrame(repeated_dummies, columns=[f'Dummy_{i + 1}' for i in range(m - 1)])
    df['t'] = range(1, len(df) + 1)
    
    model = LinearRegression().fit(df, y)
    initial_s = np.append(model.coef_[0:m-1], 0)
    initial_s = initial_s - (sum(model.coef_[0:m-1]) / m)
    
    cycle = itertools.cycle(np.arange(1, m + 1))
    vector = list(itertools.islice(cycle, len(y)))
    last_m_rev = np.array(vector[-m:][::-1])
    
    initial_s_final = initial_s[last_m_rev - 1]
    y_rev = y[::-1] # Reverse the series for backcasting
    
    l[0], b[0] = y_rev[0], -1 * model.coef_[-1]
    s[:len(initial_s_final)] = initial_s_final

    # Backcasting pass
    for t in range(1, n + 1):
        l[t] = alpha * (y_rev[t-1] - s[t-1]) + (1 - alpha) * (l[t-1] + b[t-1])
        b[t] = beta * (l[t] - l[t-1]) + (1 - beta) * b[t-1]
        s[t+m-1] = gamma * (y_rev[t-1] - l[t]) + (1 - gamma) * s[t-1]
        y_hat[t] = l[t-1] + b[t-1] + s[t-1]
   
    last_m_s = s[len(y):len(y)+m]
    mean_s = np.mean(last_m_s)
    
    return {
        'Level': (l[-1] + b[-1]) + mean_s,
        'Trend': b[-1] * -1,
        'Seasonal': (last_m_s[::-1]) - mean_s,
        'Predictions': y_hat[1:]
    }

def holt_winters_forecasting(y, alpha, beta, gamma, m, n_preds, initial_level, initial_trend, initial_s):
    """Forward Additive Holt-Winters forecasting"""
    n = len(y)
    l, b = np.zeros(n + n_preds + 1), np.zeros(n + n_preds + 1)
    s = np.zeros(n + m + n_preds)
    y_hat = np.zeros(n + n_preds + 1)
    
    l[0], b[0], s[:len(initial_s)] = initial_level, initial_trend, initial_s

    for t in range(1, n + 1):
        l[t] = alpha * (y[t-1] - s[t-1]) + (1 - alpha) * (l[t-1] + b[t-1])
        b[t] = beta * (l[t] - l[t-1]) + (1 - beta) * b[t-1]
        s[t+m-1] = gamma * (y[t-1] - l[t]) + (1 - gamma) * s[t-1]
        y_hat[t] = l[t-1] + b[t-1] + s[t-1]
    
    # Generate Forecasts
    last_l, last_b, last_s = l[n], b[n], s[n:n+m]
    for idx, t in enumerate(range(n + 1, n + n_preds + 1)):
        y_hat[t] = last_l + (t - n) * last_b + last_s[idx % m]
            
    return {'Predictions': y_hat[1:n+1].tolist(), 'Forecasts': y_hat[n+1:].tolist()}

# ==========================================
# OPTIMIZATION ENGINE
# ==========================================

def objective_function(params):
    """Minimize MAE on the Training set using Back-casted parameters"""
    alpha, beta, gamma = params
    result = holt_winters_backcasting(train_data.values, alpha, beta, gamma, m=PERIOD)
    return mean_absolute_error(train_data.values[::-1], result['Predictions'])

def gwo(objf, lb, ub, dim, agents, iterations):
    """Grey Wolf Optimizer for automated smoothing factor tuning"""
    lb, ub = np.array(lb), np.array(ub)
    alpha_pos, alpha_score = np.zeros(dim), float("inf")
    beta_pos, beta_score = np.zeros(dim), float("inf")
    delta_pos, delta_score = np.zeros(dim), float("inf")
    positions = opposition_based_learning(agents, dim, lb, ub)
    
    for l in range(iterations):
        for i in range(agents):
            positions[i, :] = np.clip(positions[i, :], lb, ub)
            fitness = objf(positions[i, :])
            if fitness < alpha_score: alpha_score, alpha_pos = fitness, positions[i, :].copy()
            elif fitness < beta_score: beta_score, beta_pos = fitness, positions[i, :].copy()
            elif fitness < delta_score: delta_score, delta_pos = fitness, positions[i, :].copy()
        
        a = 2 - l * (2 / iterations)
        for i in range(agents):
            for j in range(dim):
                # Update positions toward Alpha, Beta, and Delta leaders
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
    n, s_err = len(errors), np.zeros(len(errors))
    for i in range(1, n): s_err[i] = phi * errors[i-1] + (1 - phi) * s_err[i-1]
    return s_err

def mean_absolute_deviation(errors, phi):
    n, mad = len(errors), np.zeros(len(errors))
    mad[0] = np.mean(np.abs(errors - np.mean(errors)))
    for i in range(1, n): mad[i] = phi * np.abs(errors[i-1]) + (1 - phi) * mad[i-1]
    return mad

def find_consecutive_indices(arr, threshold):
    """Triggers if Tracking Signal remains above threshold for 3 consecutive points"""
    return next((i for i in range(len(arr) - 2) if all(arr[i + j] > threshold for j in range(3))), -1)

# ==========================================
# MAIN EXECUTION
# ==========================================
start_time = time.perf_counter()
file_path = 'regional_averages_2024_full.csv'
dataset_full = pd.read_csv(file_path, header=0, index_col=0, sep=';')

# Analyze specific data indices (e.g., German Weather Stations)
for data_idx in range(17, 18):
    raw_series = dataset_full.iloc[:, data_idx]
    train_end, val_end = 600, 1668
    train_data, val_data, test_data = raw_series[:train_end], raw_series[train_end:val_end], raw_series[val_end:]
    train_val_data = raw_series[:val_end]

    hyper_results = []
    for s_param in ALPHA_VALS:
        for mad_param in ALPHA_VALS:
            if mad_param <= s_param:
                # Initial optimization run
                best_params, _ = gwo(objective_function, [0,0,0], [1,1,1], 3, SEARCH_AGENTS_NO, MAX_ITER)
                alpha, beta, gamma = best_params
                hw_init = holt_winters_backcasting(train_data.values, alpha, beta, gamma, PERIOD)
                result = holt_winters_forecasting(train_data.values, alpha, beta, gamma, PERIOD, len(val_data), 
                                                hw_init['Level'], hw_init['Trend'], hw_init['Seasonal'])
                
                forecasts, new_index, total_forecasts = result['Forecasts'], 0, np.array([])
                
                # SETS Adaptive Loop
                while True:
                    curr_val = val_data.values[new_index:]
                    if len(curr_val) != len(forecasts):
                        min_l = min(len(curr_val), len(forecasts))
                        curr_val, forecasts = curr_val[:min_l], forecasts[:min_l]

                    errors = curr_val - forecasts
                    ts = np.abs(smoothed_error(errors, s_param) / (mean_absolute_deviation(errors, mad_param) + 1e-10))
                    trigger_idx = find_consecutive_indices(ts, THRESHOLD_VALUE)
                    
                    if trigger_idx < 0:
                        total_forecasts = np.append(total_forecasts, forecasts)
                        break
                    
                    total_forecasts = np.append(total_forecasts, forecasts[:trigger_idx])
                    new_index += trigger_idx
                    # Bias detected -> Re-tune model parameters
                    updated_train = train_val_data[new_index:train_end + new_index]
                    best_params, _ = gwo(objective_function, [0,0,0], [1,1,1], 3, SEARCH_AGENTS_NO, MAX_ITER)
                    alpha, beta, gamma = best_params
                    hw_upd = holt_winters_backcasting(updated_train.values, alpha, beta, gamma, PERIOD)
                    result = holt_winters_forecasting(updated_train.values, alpha, beta, gamma, PERIOD, 
                                                    len(train_val_data) - (train_end + new_index), 
                                                    hw_upd['Level'], hw_upd['Trend'], hw_upd['Seasonal'])
                    forecasts = result['Forecasts']

                # Final evaluation on the Test set
                final_train = train_val_data[new_index:]
                hw_f = holt_winters_backcasting(final_train.values, alpha, beta, gamma, PERIOD)
                test_res = holt_winters_forecasting(final_train.values, alpha, beta, gamma, PERIOD, len(test_data),
                                                  hw_f['Level'], hw_f['Trend'], hw_f['Seasonal'])
                
                hyper_results.append({
                    'val_rmse': rmse(val_data.values, total_forecasts), 'test_rmse': rmse(test_data.values, test_res['Forecasts']),
                    'test_mae': mean_absolute_error(test_data.values, test_res['Forecasts']),
                    'test_mape': mean_absolute_percentage_error(test_data.values, test_res['Forecasts']),
                    'test_smape': calculate_smape(test_data.values, test_res['Forecasts']),
                    'forecasts': test_res['Forecasts'], 'validations': total_forecasts
                })

    # Find the best hyperparameter configuration
    best = min(hyper_results, key=lambda x: x['val_rmse'])
    
    # Plotting Output
    plt.figure(figsize=(14, 7))
    plt.plot(np.arange(train_end), train_data.values, label='Training Data', color='blue')
    plt.plot(np.arange(train_end, val_end), val_data.values, label='True Validation', color='green')
    plt.plot(np.arange(train_end, val_end), best['validations'], label='Predicted Validation', color='orange', linestyle='--')
    plt.plot(np.arange(val_end, len(raw_series)), test_data.values, label='True Test', color='black')
    plt.plot(np.arange(val_end, len(raw_series)), best['forecasts'], label='Predicted Test', color='red')
    
    plt.axvline(x=train_end, color='gray', linestyle=':', label='Val Start')
    plt.axvline(x=val_end, color='gray', linestyle=':', label='Test Start')
    plt.title(f'Additive Holt-Winters GWO Adaptive Forecasting (DWD Temperature)')
    plt.xlabel('Time Points'), plt.ylabel('Temperature'), plt.legend(), plt.grid(True), plt.show()

    print(f"Final Metrics: RMSE: {best['test_rmse']:.4f}, MAE: {best['test_mae']:.4f}, MAPE: {best['test_mape']:.4f}")

print(f"Total Execution Time: {time.perf_counter() - start_time:.2f}s")