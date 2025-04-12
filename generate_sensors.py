# generate_sensors.py
import json
import random

# Configuration
grid_size = 100          # Grid dimensions (100x100)
num_sensors = 50         # Number of sensor points to generate
output_file = "data/sensors.json"

# Generate unique random sensor coordinates within the grid
sensors = set()
while len(sensors) < num_sensors:
    x = random.randint(0, grid_size - 1)
    y = random.randint(0, grid_size - 1)
    sensors.add((x, y))
sensors_list = list(map(list, sensors))  # convert to list of [x, y]

# Prepare data and write to JSON
data = {
    "grid_size": grid_size,
    "sensors": sensors_list
}
with open(output_file, "w") as f:
    json.dump(data, f, indent=2)
print(f"Generated {num_sensors} sensors in a {grid_size}x{grid_size} grid. Data saved to {output_file}.")
