import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib.lines import Line2D

# --- Parameters ---
num_sensors = 25    # number of sensor nodes in the field
num_robots  = 3     # number of mobile robots
np.random.seed(42)  # seed for reproducibility of random sensor layout

# Generate random sensor coordinates within a 100x100 area (uniformly distributed)
sensor_coords = np.random.rand(num_sensors, 2) * 100  # shape: (num_sensors, 2)
base_station = np.array([0.0, 0.0])  # define base station location (e.g., at origin)

# --- Clustering sensors into k clusters (for k robots) using KMeans ---
kmeans = KMeans(n_clusters=num_robots, n_init=10, random_state=0)
labels = kmeans.fit_predict(sensor_coords)       # labels[i] is cluster index (0..k-1) for sensor i
centroids = kmeans.cluster_centers_              # coordinates of cluster centroids

# Prepare data structures for robot routes and times
robot_paths = []   # list of routes (each route is a list of [x,y] points) for each robot
robot_times = []   # list of total times for each robot's tour

# Time parameters
speed = 1.0           # robot travel speed (distance units per second)
download_time = 5.0   # time to download data from one sensor (seconds)

# --- Plan route for each robot's cluster ---
for r in range(num_robots):
    # Get all sensors (indices and coordinates) belonging to cluster r
    cluster_indices = np.where(labels == r)[0]
    cluster_points = sensor_coords[cluster_indices]
    # Define start/end point as the cluster centroid
    start_point = centroids[r]
    # If there are no sensors in this cluster (unlikely here), skip
    if cluster_points.shape[0] == 0:
        robot_paths.append([start_point])  # no movement, just stays at centroid
        robot_times.append(0.0)
        continue

    # Use a nearest-neighbor heuristic to order the sensor visits
    unvisited = list(range(cluster_points.shape[0]))   # indices of sensors in this cluster
    current_pos = start_point
    visit_order = []  # order of visits (indices into cluster_points)
    while unvisited:
        # Find the nearest unvisited sensor to the current position
        distances = np.linalg.norm(cluster_points[unvisited] - current_pos, axis=1)
        nearest_idx = np.argmin(distances)       # index in unvisited list of closest sensor
        next_sensor = unvisited[nearest_idx]      # actual index of that sensor in cluster_points
        visit_order.append(next_sensor)
        # Move to that sensor
        current_pos = cluster_points[next_sensor]
        # Mark it visited by removing from unvisited list
        unvisited.remove(next_sensor)

    # Construct the route: start at centroid, visit sensors in order, then return to centroid
    route = [start_point] 
    for idx in visit_order:
        route.append(cluster_points[idx])
    route.append(start_point)  # return to start
    robot_paths.append(route)

    # Calculate total distance traveled on this route
    total_dist = 0.0
    for i in range(len(route) - 1):
        a = route[i]
        b = route[i+1]
        total_dist += np.linalg.norm(b - a)
    # Calculate total time = travel time + (download time * number of sensors)
    travel_time = total_dist / speed
    total_download = cluster_points.shape[0] * download_time
    total_time = travel_time + total_download
    robot_times.append(total_time)

# Print out the estimated completion time for each robot
for i, t in enumerate(robot_times, start=1):
    print(f"Robot {i} estimated completion time: {t:.1f} seconds")

# --- Visualization ---
fig, ax = plt.subplots(figsize=(8, 8))
colors = plt.get_cmap('tab10').colors  # color palette for clusters (up to 10 distinct colors)

# Plot each robot's path (as a colored line)
for r, route in enumerate(robot_paths):
    route = np.array(route)
    ax.plot(route[:,0], route[:,1], color=colors[r], linewidth=2, label=None)

# Plot sensors for each cluster (colored points) and label each sensor with an ID
for r in range(num_robots):
    cluster_indices = np.where(labels == r)[0]
    cluster_points = sensor_coords[cluster_indices]
    ax.scatter(cluster_points[:,0], cluster_points[:,1], marker='o', 
               facecolor=colors[r], edgecolor='black', s=60)
    # Label each sensor with its ID number
    for idx in cluster_indices:
        sx, sy = sensor_coords[idx]
        ax.text(sx+1, sy+1, str(idx+1), fontsize=8)  # +1 offset to avoid overlapping the marker

# Plot robot start/end positions (centroids) as star markers
ax.scatter(centroids[:,0], centroids[:,1], marker='*', facecolor=colors[:num_robots], 
           edgecolor='black', s=250, label=None)
# Plot the base station as a black square
ax.scatter(base_station[0], base_station[1], marker='s', facecolor='black', s=100, label=None)

# Annotate each robot's time near the start position
for r, (cx, cy) in enumerate(centroids):
    ax.annotate(f"T = {robot_times[r]:.1f} s", (cx, cy), textcoords="offset points", 
                xytext=(0, 10), ha='center', fontsize=9, color='black')

# Configure legend to explain markers and lines (one example for each type)
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=8, 
           label='Sensor node (cluster-coded)'),
    Line2D([0], [0], color='black', lw=2, label='Robot path (cluster-coded)'),
    Line2D([0], [0], marker='*', color='black', markersize=12, label='Robot start/end (centroid)'),
    Line2D([0], [0], marker='s', color='black', markersize=10, label='Base station')
]
ax.legend(handles=legend_elements, loc='best')
ax.set_title("2D Data Gathering Problem Simulation")
ax.set_xlabel("X coordinate")
ax.set_ylabel("Y coordinate")
ax.set_aspect('equal', 'box')
ax.grid(True, linestyle='--', alpha=0.7)

# Adjust plot limits to include all points comfortably
all_x = np.concatenate([sensor_coords[:,0], [base_station[0]]])
all_y = np.concatenate([sensor_coords[:,1], [base_station[1]]])
pad_x = 0.05 * (all_x.max() - all_x.min())
pad_y = 0.05 * (all_y.max() - all_y.min())
ax.set_xlim(all_x.min() - pad_x, all_x.max() + pad_x)
ax.set_ylim(all_y.min() - pad_y, all_y.max() + pad_y)

plt.show()