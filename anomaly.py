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
    # Load traffic data
    traffic_data = load_traffic_data()

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
