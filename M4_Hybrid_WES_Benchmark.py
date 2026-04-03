# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 21:38:42 2026

@author: erdinck
"""

# -*- coding: utf-8 -*-
"""
ENHANCED HYBRID WES FRAMEWORK: M4 COMPETITION BENCHMARKING

DATASET DESCRIPTION:
The experimental scope comprises 40 diverse series from the M4 Forecasting Competition 
representing four distinct temporal frequencies:
- Quarterly (m=4, h=8): IDs Q1024, Q1100, Q10210, Q10446, Q10499, Q10580, Q10583, Q10592, Q10608, Q10612
- Monthly (m=12, h=18): IDs M17, M19, M22, M23, M133, M135, M142, M151, M188, M191
- Weekly (m=52, h=13): IDs W2, W19, W57, W74, W83, W96, W110, W116, W235, W239
- Hourly (m=24, h=48): IDs H10, H31, H55, H73, H105, H133, H311, H318, H401, H410

OBJECTIVE:
To establish the external validity of the proposed method by enhancing Winters' Exponential Smoothing (WES):
1. BACK-CASTING: Accurate initialization of Level, Trend, and Seasonal components for both ADD and MUL models.
2. 5D GREY WOLF OPTIMIZATION (GWO): Automated simultaneous tuning of smoothing factors, SETS threshold, and damping factor (phi).
3. SMOOTHED ERROR TRACKING SIGNAL (SETS): Adaptive bias correction mechanism that monitors forecast errors.
4. MODEL SELECTION: Automated switching between Additive and Multiplicative logic based on validation sMAPE.

SOURCE: datasetsforecast Python library (M4 Competition Data).
Reference: Makridakis et al. (2019) https://doi.org/10.1016/j.ijforecast.2019.04.014

NOTE: This comprehensive framework is designed for large-scale benchmarking across the M4 Competition dataset. 
It integrates both additive and multiplicative versions of the Enhanced Winters’ Exponential Smoothing (WES) method.
Functional Enhancements & Differences
In this version, the architecture has been upgraded to handle diverse data frequencies and domains:
- Unified Adaptive Engine: Unlike the standalone scripts, this version features a logic that automatically evaluates 
both additive and multiplicative models for each series, selecting the best fit based on validation performance.
- 5-Dimensional Grey Wolf Optimization (GWO): The search space is expanded to dim=5. It optimizes not only the smoothing factors 
(alpha, beta, gamma) but also the SETS Threshold and the Bias Damping Factor (phi) simultaneously.
- External Validity Testing: By utilizing 40 specific series (10 for each frequency), the model's ability to correct bias is tested 
against varied trend slopes and noise levels.
- Bias Damping (phi): A new parameter, phi, ensures that the bias adjustment derived via SETS is gradually reduced over the 
forecast horizon, preventing over-correction in long-term steps.

"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import itertools
import time
from sklearn.linear_model import LinearRegression
from datasetsforecast.m4 import M4
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pmdarima as pm

warnings.filterwarnings("ignore")

# ======================================================
# 1. CORE PROPOSED METHODS (INITIALIZATION & ENGINE)
# ======================================================

def hw_add_backcasting(y, m):
    """Calculates initial states for Additive WES using reversed regression."""
    n = len(y)
    if n < 2 * m: return {'Level': y[-1] if n>0 else 0, 'Trend': 0, 'Seasonal': np.zeros(m)}
    dummies = []
    for i in range(m - 1):
        row = [0] * (m - 1); row[i] = 1; dummies.append(row)
    dummies.append([0] * (m - 1))
    df = pd.DataFrame(dummies * (n // m) + dummies[:n % m])
    df.columns = df.columns.astype(str)
    df['t'] = range(1, len(df)+1)
    model = LinearRegression().fit(df, y)
    initial_s = np.append(model.coef_[0:m-1], 0)
    initial_s -= np.mean(initial_s)
    y_rev = y[::-1]
    l0, b0 = y_rev[0], -1 * model.coef_[-1]
    cycle = itertools.cycle(np.arange(1, m+1))
    vector = list(itertools.islice(cycle, len(y)))
    last_m_rev = np.array(vector[-m:][::-1])
    sss = initial_s[np.array(last_m_rev) - 1]
    return {'Level': l0 + np.mean(sss), 'Trend': b0, 'Seasonal': sss - np.mean(sss)}

def hw_multi_backcasting(y, m):
    """Calculates initial states for Multiplicative WES."""
    y = np.clip(y, 1e-5, None)
    n = len(y)
    if n < 2 * m: return {'Level': y[-1] if n>0 else 1, 'Trend': 0, 'Seasonal': np.ones(m)}
    mean_f = np.mean(y[:m])
    b0 = (np.mean(y[m:2*m]) - mean_f) / m
    l0 = mean_f - (m / 2) * b0
    s0 = y[:m] / (l0 + np.arange(m) * b0 + 1e-10)
    return {'Level': l0, 'Trend': b0, 'Seasonal': s0}

def forecasting_engine(y_history, h, alpha, beta, gamma, m, model_type='add', threshold=0.5, phi=0.9, sets_logic=False):
    """Adaptive forecasting engine with SETS and Damped Bias Correction."""
    if len(y_history) == 0: return np.zeros(h)
    init = hw_add_backcasting(y_history, m) if model_type == 'add' else hw_multi_backcasting(y_history, m)
    l, b, s = init['Level'], init['Trend'], init['Seasonal'].copy()
    s_err, mad, bias_adj = 0.0, 1.0, 0.0
    for i in range(len(y_history)):
        s_idx = i % m
        y_hat = (l + b + s[s_idx] + bias_adj) if model_type == 'add' else ((l + b) * s[s_idx] + bias_adj)
        err = y_history[i] - y_hat
        if sets_logic:
            s_err = 0.15 * err + 0.85 * s_err
            mad = 0.10 * abs(err) + 0.90 * mad
            if abs(s_err / max(mad, 1e-10)) > threshold: bias_adj = s_err
            else: bias_adj *= 0.8
        l_old = l
        if model_type == 'add':
            l = alpha * (y_history[i] - s[s_idx]) + (1 - alpha) * (l + b)
            b = beta * (l - l_old) + (1 - beta) * b
            s[s_idx] = gamma * (y_history[i] - l) + (1 - gamma) * s[s_idx]
        else:
            l = alpha * (y_history[i] / max(s[s_idx], 1e-10)) + (1 - alpha) * (l + b)
            b = beta * (l - l_old) + (1 - beta) * b
            s[s_idx] = gamma * (y_history[i] / max(l, 1e-10)) + (1 - gamma) * s[s_idx]
    preds = []
    for step in range(1, h + 1):
        idx = (len(y_history) + step - 1) % m
        db = (phi ** step) * bias_adj
        preds.append((l + step * b + s[idx] + db) if model_type == 'add' else ((l + step * b) * s[idx] + db))
    return np.array(preds)

# ======================================================
# 2. OPTIMIZATION & PERFORMANCE HELPERS
# ======================================================

def smape(a, f):
    if len(a) == 0 or len(f) == 0: return 999999
    return np.mean(2 * np.abs(f - a) / (np.abs(a) + np.abs(f) + 1e-10)) * 100

def gwo(objf, agents=10, iters=20):
    dim = 5
    lb, ub = np.array([0, 0, 0, 0.2, 0.6]), np.array([1, 1, 1, 0.8, 1.0])
    alpha_pos, alpha_score = np.zeros(dim), float("inf")
    pos = lb + (ub - lb) * np.random.rand(agents, dim)
    for l in range(iters):
        for i in range(agents):
            try:
                score = objf(pos[i])
                if score < alpha_score: alpha_score, alpha_pos = score, pos[i].copy()
            except: continue
        a_p = 2 - l * (2 / iters)
        for i in range(agents):
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            A, C = 2 * a_p * r1 - a_p, 2 * r2
            pos[i] = np.clip(alpha_pos - A * abs(C * alpha_pos - pos[i]), lb, ub)
    return alpha_pos

def get_ranked_results_strings(results_list):
    """Formats detailed table with SMAPE and local rank for each series."""
    df = pd.DataFrame(results_list)
    cols = ["Proposed", "Static", "ARIMA", "ETS", "SNaive"]
    
    display_df = df.copy()
    for col in cols:
        display_df[col] = display_df[col].astype(object)
    
    for i, row in df.iterrows():
        vals = np.array([row[c] for c in cols]).astype(float)
        vals = np.nan_to_num(vals, nan=999999.0)
        
        ranks = pd.Series(vals).rank(method='min').values.astype(int)
        
        for idx, col in enumerate(cols):

            display_df.at[i, col] = f"{vals[idx]:.4f} ({ranks[idx]})"
            
    return display_df

# ======================================================
# 3. BENCHMARK EXECUTION & VISUALIZATION
# ======================================================

def run_experiment():
    data_map = {
        'Quarterly': {'m': 4,  'h': 8,  'ids': ['Q1024', 'Q1100', 'Q10210', 'Q10446', 'Q10499', 'Q10580', 'Q10583', 'Q10592', 'Q10608', 'Q10612']},
        'Monthly':   {'m': 12, 'h': 18, 'ids': ['M17', 'M19', 'M22', 'M23', 'M133', 'M135', 'M142', 'M151', 'M188', 'M191']},
        'Weekly':    {'m': 52, 'h': 13, 'ids': ['W2', 'W19', 'W57', 'W74', 'W83', 'W96', 'W110', 'W116', 'W235', 'W239']},
        'Hourly':    {'m': 24, 'h': 48, 'ids': ['H10', 'H31', 'H55', 'H73', 'H105', 'H133', 'H311', 'H318', 'H401','H410']},
    }
    visualize_ids = {'Quarterly': ['Q1024', 'Q10446', 'Q10580', 'Q10583', 'Q10592'], 'Monthly': ['M17', 'M23', 'M135', 'M142', 'M191'], 
                      'Weekly': ['W2', 'W19', 'W110', 'W235', 'W239'], 'Hourly': ['H10', 'H31', 'H55', 'H73', 'H311']}
    
    results, plot_data = [], {g: {} for g in data_map.keys()}

    for g_name, meta in data_map.items():
        m, h = meta['m'], meta['h']
        print(f"\nProcessing Group: {g_name} (m={m}, h={h})")
        df, *_ = M4.load("./data", g_name)
        
        for uid in meta['ids']:
            series_data = df[df['unique_id'] == uid]['y'].values
            if len(series_data) < h + 5: continue
            
            cal_idx = min(int(len(series_data) * 0.7), len(series_data) - h)
            split_point = max(1, cal_idx - h)
            y_tr, y_te = series_data[:split_point], series_data[split_point:cal_idx]
            
            # Optimization & Selection
            p_add = gwo(lambda p: smape(y_te, forecasting_engine(y_tr, len(y_te), p[0], p[1], p[2], m, 'add', p[3], p[4], True)), agents=5, iters=10)
            p_mul = gwo(lambda p: smape(y_te, forecasting_engine(y_tr, len(y_te), p[0], p[1], p[2], m, 'mul', p[3], p[4], True)), agents=5, iters=10)
            
            m_type = 'add' if smape(y_te, forecasting_engine(y_tr, len(y_te), p_add[0], p_add[1], p_add[2], m, 'add', p_add[3], p_add[4], True)) < \
                              smape(y_te, forecasting_engine(y_tr, len(y_te), p_mul[0], p_mul[1], p_mul[2], m, 'mul', p_mul[3], p_mul[4], True)) else 'mul'
            best_p = p_add if m_type == 'add' else p_mul
            
            # Rolling-origin evaluation
            fold_smapes = {k: [] for k in ["Proposed", "Static", "ARIMA", "ETS", "SNaive"]}
            n_folds = min(4, max(1, len(series_data) // (h + 2)))
            
            for f in range(n_folds):
                origin = len(series_data) - (f * h)
                if origin <= h + 1: break
                y_hist, y_test = series_data[:origin-h], series_data[origin-h:origin]
                
                fold_smapes["Proposed"].append(smape(y_test, forecasting_engine(y_hist, h, best_p[0], best_p[1], best_p[2], m, m_type, best_p[3], best_p[4], True)))
                fold_smapes["Static"].append(smape(y_test, forecasting_engine(y_hist, h, best_p[0], best_p[1], best_p[2], m, m_type, best_p[3], best_p[4], False)))
                
                # Benchmarks
                p_snaive = np.tile(y_hist[-m:], (h // m) + 1)[:h] if len(y_hist) >= m else np.full(h, y_hist[-1])
                fold_smapes["SNaive"].append(smape(y_test, p_snaive))
                
                try:
                    m_ets = ETSModel(y_hist, error="add", trend="add", seasonal="add", seasonal_periods=m).fit(disp=0)
                    fold_smapes["ETS"].append(smape(y_test, m_ets.forecast(h)))
                except: fold_smapes["ETS"].append(smape(y_test, np.full(h, y_hist[-1])))
                
                try:
                    m_ari = pm.auto_arima(y_hist, seasonal=(m<=24), m=m if m<=24 else 1, error_action="ignore")
                    fold_smapes["ARIMA"].append(smape(y_test, m_ari.predict(h)))
                except: fold_smapes["ARIMA"].append(smape(y_test, np.full(h, y_hist[-1])))

                if f == 0 and uid in visualize_ids.get(g_name, []):
                    plot_data[g_name][uid] = {'hist': y_hist, 'test': y_test, 'origin': origin-h,
                        'sets': forecasting_engine(y_hist, h, best_p[0], best_p[1], best_p[2], m, m_type, best_p[3], best_p[4], True),
                        'stat': forecasting_engine(y_hist, h, best_p[0], best_p[1], best_p[2], m, m_type, best_p[3], best_p[4], False)}

            results.append({"Group": g_name, "ID": uid, **{k: np.mean(v) for k, v in fold_smapes.items()}})
            print(f"  ✓ {uid} Processed")

    # GRID PLOT GENERATION
    fig, axes = plt.subplots(10, 4, figsize=(24, 45))
    for col, g_name in enumerate(data_map.keys()):
        for row_pair, uid in enumerate(visualize_ids.get(g_name, [])):
            if uid not in plot_data[g_name]: continue
            d = plot_data[g_name][uid]
            full_y = np.concatenate([d['hist'], d['test']])
            idx = np.arange(len(full_y))
            
            # Main View
            ax = axes[row_pair*2, col]
            ax.plot(idx[:len(d['hist'])], d['hist'], color='gray', alpha=0.4)
            ax.plot(idx[len(d['hist']):], d['test'], 'k', label='Actual')
            ax.plot(idx[len(d['hist']):], d['sets'], 'blue', linewidth=2, label='Proposed')
            ax.plot(idx[len(d['hist']):], d['stat'], 'r--', label='Static HW')
            ax.axvline(len(d['hist']), color='green', linestyle=':')
            ax.set_title(f"{g_name} - {uid}")
            
            # Zoom View
            axz = axes[row_pair*2+1, col]
            z_start = max(0, len(d['hist']) - len(d['test'])*2)
            axz.plot(idx[z_start:len(d['hist'])], d['hist'][z_start-len(d['hist']):], color='gray')
            axz.plot(idx[len(d['hist']):], d['test'], 'k', linewidth=1.5)
            axz.plot(idx[len(d['hist']):], d['sets'], 'blue', linewidth=2)
            axz.plot(idx[len(d['hist']):], d['stat'], 'r--')
            axz.axvline(len(d['hist']), color='green', linestyle=':')
            axz.set_title(f"Zoom: {uid}")

    plt.tight_layout()
    plt.show()

    # FINAL RANKED TABLE
    final_df = get_ranked_results_strings(results)
    print("\n" + "="*120 + "\nFINAL PERFORMANCE REPORT (SMAPE & RANK)\n" + "="*120)
    print(final_df.to_string(index=False))
    print("\nMEAN GLOBAL RANK:")
    print(pd.DataFrame(results)[["Proposed", "Static", "ARIMA", "ETS", "SNaive"]].rank(axis=1).mean())

if __name__ == "__main__":
    run_experiment()