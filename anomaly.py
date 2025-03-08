import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest

# Define the directory where traffic data is stored
DATA_DIR = "traffic_data"

def load_traffic_matrix(file_path):
    with open(file_path, 'r') as f:
        data = f.readlines()

    # Take only the first 12 lines
    data = data[:12]

    # Convert text to NumPy array, taking only the first 12 values per line
    traffic_matrix = np.array([list(map(float, line.split()[:12])) for line in data])

    print(f"Loaded {file_path} with shape {traffic_matrix.shape}")  # Debugging output
    return traffic_matrix

# Load all traffic matrices
traffic_matrices = []
for i in range(1, 25):
    file_path = os.path.join(DATA_DIR, f'X{i:02d}')  # Formats as traffic_data/X01, X02, ..., X24
    if os.path.exists(file_path):
        traffic_matrix = load_traffic_matrix(file_path)
        
        if traffic_matrix.shape == (12, 12):  # Ensure correct shape
            traffic_matrices.append(traffic_matrix)
        else:
            print(f"Skipping {file_path}, incorrect shape: {traffic_matrix.shape}")
    else:
        print(f"File not found: {file_path}")

# Stack into a 3D array
if traffic_matrices:
    traffic_data = np.stack(traffic_matrices)
    print("Final traffic data shape:", traffic_data.shape)  # Should be (24, 12, 12)
else:
    raise ValueError("No valid traffic matrices loaded!")

# Function to detect anomalies using Z-score
def detect_anomalies_zscore(data, threshold=2.5):
    time_series = np.array([np.sum(matrix) for matrix in data])
    z_scores = np.abs(zscore(time_series))
    anomalies = np.where(z_scores > threshold)[0]
    
    return time_series, anomalies

# Function to detect anomalies using Isolation Forest
def detect_anomalies_isolation_forest(data, contamination=0.05):
    time_series = np.array([np.sum(matrix) for matrix in data]).reshape(-1, 1)
    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(time_series)
    anomaly_labels = model.predict(time_series)
    anomalies = np.where(anomaly_labels == -1)[0]
    
    return time_series.flatten(), anomalies

# Function to plot anomalies
def plot_anomalies(time_series, anomalies, title="Anomaly Detection"):
    plt.figure(figsize=(10, 5))
    plt.plot(time_series, marker="o", linestyle="-", label="Total Traffic")
    plt.scatter(anomalies, time_series[anomalies], color="red", label="Anomalies", zorder=3)
    plt.xlabel("Time Step")
    plt.ylabel("Total Traffic Volume")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Detect anomalies using Z-score
    time_series_z, anomalies_z = detect_anomalies_zscore(traffic_data)
    print(f"Z-score Anomalies detected at time steps: {anomalies_z}")

    # Plot Z-score anomalies
    plot_anomalies(time_series_z, anomalies_z, "Anomaly Detection using Z-Score")

    # Detect anomalies using Isolation Forest
    time_series_if, anomalies_if = detect_anomalies_isolation_forest(traffic_data)
    print(f"Isolation Forest Anomalies detected at time steps: {anomalies_if}")

    # Plot Isolation Forest anomalies
    plot_anomalies(time_series_if, anomalies_if, "Anomaly Detection using Isolation Forest")
