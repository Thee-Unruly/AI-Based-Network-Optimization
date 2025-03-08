# Load the Abilene traffic data (already preprocessed).
# Use statistical methods (Z-score) and machine learning models (Isolation Forest).
# Visualize the anomalies in the traffic data.

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest

# Function to load preprocessed traffic data
def load_traffic_data(file_path="traffic_data.npy"):
    return np.load(file_path)

# Function to detect anomalies using Z-score
def detect_anomalies_zscore(data, threshold=2.5):
    time_series = np.array([np.sum(matrix) for matrix in data])
    z_scores = np.abs(zscore(time_series))
    anomalies = np.where(z_scores > threshold)[0]
    
    return time_series, anomalies

