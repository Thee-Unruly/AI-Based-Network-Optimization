# TRAFFIC ANALYTICS AND NETWORK OPTIMIZATIONS

import numpy as np

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
    file_path = f'X{i:02d}'  # Formats as X01, X02, ..., X24
    traffic_matrix = load_traffic_matrix(file_path)
    
    if traffic_matrix.shape == (12, 12):  # Ensure correct shape
        traffic_matrices.append(traffic_matrix)
    else:
        print(f"Skipping {file_path}, incorrect shape: {traffic_matrix.shape}")

# Stack into a 3D array
traffic_data = np.stack(traffic_matrices)
print("Final traffic data shape:", traffic_data.shape)  # Should be (24, 12, 12)
