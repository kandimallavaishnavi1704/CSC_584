# main.py
import json
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from path_palnner import a_star

# Load sensor locations from JSON
with open("data/sensors.json") as f:
    data = json.load(f)
grid_size = data["grid_size"]
sensor_points = [tuple(pt) for pt in data["sensors"]]

# User-defined parameters
num_robots = 5    # for example, use 5 robots (must be between 3 and 30 as per requirements)
num_obstacles = 20

# Sanity check: adjust robots if more than sensors
if num_robots > len(sensor_points):
    num_robots = len(sensor_points)

# Generate random obstacle locations (avoid placing on sensors)
obstacles = set()
while len(obstacles) < num_obstacles:
    ox = random.randint(0, grid_size - 1)
    oy = random.randint(0, grid_size - 1)
    if (ox, oy) in obstacles or (ox, oy) in sensor_points:
        continue
    obstacles.add((ox, oy))

# Cluster sensors into groups (KMeans for task allocation)
coords = [list(p) for p in sensor_points]
kmeans = KMeans(n_clusters=num_robots, n_init=10, random_state=0)
labels = kmeans.fit_predict(coords)

# Organize sensors by cluster
clusters = {i: [] for i in range(num_robots)}
for point, label in zip(sensor_points, labels):
    clusters[label].append(point)

# Assign robot start positions as cluster centroids (rounded to nearest grid cell)
robot_starts = {}
for i in range(num_robots):
    if clusters[i]:
        cx, cy = kmeans.cluster_centers_[i]
        # Round centroid to nearest integer coordinate on grid
        sx = int(round(cx))
        sy = int(round(cy))
        # Ensure start is inside bounds
        sx = max(0, min(grid_size - 1, sx))
        sy = max(0, min(grid_size - 1, sy))
        # If an obstacle is exactly at the centroid, adjust or remove that obstacle
        if (sx, sy) in obstacles:
            obstacles.remove((sx, sy))
        robot_starts[i] = (sx, sy)
    else:
        # If a cluster is empty (happens if number of robots > sensors), place start at (0,0) arbitrarily
        robot_starts[i] = (0, 0)

# Dynamic reassignment: check each sensor is reachable by its assigned robot, otherwise reassign
reassign_log = []
for i in list(clusters.keys()):
    for sensor in clusters[i].copy():
        path = a_star(robot_starts[i], sensor, obstacles, grid_size)
        if path is None:
            # This sensor is not reachable by robot i due to obstacles
            # Find an alternative robot that can reach it
            new_owner = None
            best_dist = float('inf')
            for j in range(num_robots):
                if j == i:
                    continue
                alt_path = a_star(robot_starts[j], sensor, obstacles, grid_size)
                if alt_path:
                    dist = len(alt_path)
                    if dist < best_dist:
                        best_dist = dist
                        new_owner = j
            if new_owner is not None:
                clusters[i].remove(sensor)
                clusters[new_owner].append(sensor)
                reassign_log.append(f"Sensor {sensor} moved from Robot {i} to Robot {new_owner}")
            # If no robot can reach this sensor (isolated by obstacles), it remains unreachable (rare scenario)

# Plan route for each robot through its cluster of sensors
routes = {}  # will store the actual grid path for each robot
for i in range(num_robots):
    sensors = clusters[i]
    if not sensors:
        routes[i] = [robot_starts[i]]  # no tasks, route is just start
        continue
    # Greedy nearest-neighbor ordering for visiting sensors
    unvisited = sensors.copy()
    current = robot_starts[i]
    route_path = [current]
    while unvisited:
        # Find nearest sensor by straight-line distance
        nearest = min(unvisited, key=lambda s: (s[0] - current[0])**2 + (s[1] - current[1])**2 )
        unvisited.remove(nearest)
        # Find path to the nearest sensor avoiding obstacles
        path_segment = a_star(current, nearest, obstacles, grid_size)
        if path_segment is None:
            # This should not happen if reassignment handled unreachable cases
            # Skip if somehow unreachable
            print(f"Warning: Robot {i} cannot reach sensor {nearest}. Skipping.")
        else:
            # Append the segment (omit the first point to avoid duplicates)
            if path_segment[0] == current:
                route_path.extend(path_segment[1:])
            else:
                route_path.extend(path_segment)
        current = nearest
    # Optionally, return to start position (to complete a tour loop)
    return_path = a_star(current, robot_starts[i], obstacles, grid_size)
    if return_path and len(return_path) > 1:
        route_path.extend(return_path[1:])
    routes[i] = route_path

# Output any dynamic reassignments that occurred
if reassign_log:
    print("Dynamic Reassignments:")
    for msg in reassign_log:
        print("  -", msg)
else:
    print("No dynamic reassignments were necessary.")

# Visualization of the final paths and assignments
colors = []  # distinct colors for each robot path
for k in range(num_robots):
    # generate distinct color (using HSV space for variety)
    hue = k / num_robots
    colors.append(plt.cm.hsv(hue))  # returns RGBA tuple

plt.figure(figsize=(10, 10))

# Plot Obstacles
if obstacles:
    ox, oy = zip(*obstacles)
    plt.scatter(ox, oy, marker='X', s=80, color='black', label='Obstacles')

# Plot Sensors clearly as red circles
all_sensors_x, all_sensors_y = zip(*sensor_points)
plt.scatter(all_sensors_x, all_sensors_y, marker='o', s=80, edgecolors='black', 
            facecolors='red', linewidth=1, label='Sensors')

# Plot Robot paths and starting points
for robot_id, path in routes.items():
    path_x, path_y = zip(*path)
    plt.plot(path_x, path_y, '-', linewidth=2, label=f'Robot {robot_id} path')
    
    # Clearly mark robot start positions as green stars
    start_x, start_y = robot_starts[robot_id]
    plt.scatter(start_x, start_y, marker='*', s=300, c='green', edgecolors='black', label=f'Robot {robot_id} Start' if robot_id == 0 else "")

# Add clear legend, title, and labels
plt.title("Multi-Robot Data Gathering with Obstacles", fontsize=16)
plt.xlabel("X Coordinate", fontsize=14)
plt.ylabel("Y Coordinate", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)
plt.xlim(-1, grid_size+1)
plt.ylim(-1, grid_size+1)

# Avoid duplicate legend entries
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='upper right', bbox_to_anchor=(1.35, 1.0))

# Mark each sensor with a clear number
for idx, (sx, sy) in enumerate(sensor_points):
    plt.text(sx+0.3, sy+0.3, str(idx), fontsize=8, color='blue')

plt.tight_layout()
plt.show()
