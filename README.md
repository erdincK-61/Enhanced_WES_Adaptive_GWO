Enhanced Hybrid Winters’ Exponential Smoothing (WES) Framework

This repository contains a state-of-the-art forecasting framework that enhances the traditional Winters’ Exponential Smoothing (WES) method. 
By integrating heuristic optimization and adaptive bias correction, this framework achieves superior forecasting accuracy across diverse time series domains.

Key Features

Back-casting Initialization: Reverses the time series to accurately stabilize the initial states of Level, Trend, and Seasonal components, reducing the warm-up period error.

Grey Wolf Optimization (GWO): Implements an automated 5D parameter tuning mechanism (optimizing alpha, beta, gamma, SETS threshold, and damping factor phi) using the Grey Wolf Optimizer.

Smoothed Error Tracking Signal (SETS): An adaptive monitoring system that detects forecast bias in real-time and triggers automatic model re-tuning with a damped bias correction.

Project Structure

Additive_WES_Adaptive_GWO.py: Tailored for additive trends, specifically tested on DWD monthly air temperature datasets.

Multiplicative_WES_Adaptive_GWO.py: Optimized for multiplicative seasonal series. Validated using TSDL indices.

M4_Hybrid_WES_Benchmark.py: A comprehensive benchmark script that automatically switches between Additive and Multiplicative models. Tested on 40 diverse series from the M4 Forecasting Competition.

Dataset Specifications

Below are the details for the specific datasets used in this study, including access dates and identifiers:

1. DWD Climate Data (Regional Averages)
   
Description: Monthly mean air temperatures for 16 German Federal States and national averages.

Source: German Weather Service (DWD) - Climate Data Center (CDC).

Version/Date Range: Data includes records up to 2024.

Date Accessed: February 20, 2025.

License: GeoNutzV (Data is free for use under the terms of the Ordinance on the Use of Data from the German Weather Service).

URL: https://opendata.dwd.de/climate_environment/CDC/regional_averages_DE/monthly/air_temperature_mean/

2. TSDL (Time Series Data Library)
   
Description: Seasonal time series indices 9, 16, 74, 117, 153, 232, 262, 401, 455, 460 with m=12.

Source: Originally maintained by Rob J. Hyndman via the 'tsdl' R package.

Version: 0.1.0 (R package version).

Date Accessed: February 20, 2025.

License: CC BY-SA 4.0.

Identifier: https://pkg.yangzhuoranyang.com/tsdl/

3. M4 Competition Dataset
   
Description: 40 diverse series representing Hourly, Weekly, Monthly, and Quarterly frequencies.

Source: M4 Forecasting Competition (Makridakis et al. 2018).

DOI: 10.1016/j.ijforecast.2018.06.001

Version: Access via 'datasetsforecast' Python library (v0.0.8).

Date Accessed: March 6, 2026.

License: MIT License (Provided by Nixtla via datasetsforecast).

Requirements
The following Python libraries are required:

pip install numpy pandas matplotlib scikit-learn statsmodels pmdarima datasetsforecast

Performance Evaluation

The models are evaluated using:

RMSE (Root Mean Squared Error)

MAE (Mean Absolute Error)

MAPE (Mean Absolute Percentage Error)

sMAPE (Symmetric Mean Absolute Percentage Error)

License

This project is licensed under the MIT License. See the LICENSE file for details.

Author

Erdinc K.

Karadeniz Technical University
