import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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

# Data Exploration & Visualization
def plot_heatmap(traffic_matrix, time_index):
    plt.figure(figsize=(8, 6))
    sns.heatmap(traffic_matrix, cmap="viridis", annot=False)
    plt.title(f"Traffic Matrix at Time Index {time_index}")
    plt.xlabel("Destination Node")
    plt.ylabel("Source Node")
    plt.show()

# Example usage: Plot the traffic matrix at time index 5
plot_heatmap(traffic_data[5], 5)

# Traffic Flow Analysis
def calculate_total_traffic(traffic_matrix):
    return np.sum(traffic_matrix)

total_traffic = calculate_total_traffic(traffic_data[0])
print(f"Total Traffic at Time Index 0: {total_traffic}")

# Network Optimization
def calculate_average_traffic_per_source_node(traffic_matrix):
    return np.mean(traffic_matrix, axis=1)

average_traffic_per_source_node = calculate_average_traffic_per_source_node(traffic_data[0])
print(f"Average Traffic per Source Node at Time Index 0: {average_traffic_per_source_node}")

# Traffic Bottleneck Analysis
def find_traffic_bottleneck(traffic_matrix):
    return np.unravel_index(np.argmin(traffic_matrix), traffic_matrix.shape)

bottleneck_source_node, bottleneck_destination_node = find_traffic_bottleneck(traffic_data[0])
print(f"Traffic Bottleneck: Source Node {bottleneck_source_node}, Destination Node {bottleneck_destination_node}")

# Visualize traffic matrices at different time steps
for t in [0, 6, 12, 18]:  # Change these indices to explore different times
    plot_heatmap(traffic_data[t], t)

# Time Series Analysis
# Aggregate total traffic per time step (sum of all 12x12 values)
time_series_data = [np.sum(matrix) for matrix in traffic_data]

# Create DataFrame
df = pd.DataFrame({'Time Step': range(1, 25), 'Total Traffic': time_series_data})

# Plot the trend
plt.figure(figsize=(10, 5))
sns.lineplot(x="Time Step", y="Total Traffic", data=df, marker="o", linestyle="-", color="b")
plt.xlabel("Time Interval")
plt.ylabel("Total Traffic Volume")
plt.title("Traffic Volume Over Time")
plt.grid()
plt.show()
