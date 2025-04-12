import math
import random
import matplotlib.pyplot as plt

def generate_sensor_positions(n, spacing='random', fixed_gap=10, max_range=100, seed=None):
    """
    Generate a sorted list of sensor positions along the x-axis.
    - If spacing='fixed': sensors are placed at regular intervals (fixed_gap apart).
    - If spacing='random': sensors are placed at random positions in [0, max_range].
    """
    if seed is not None:
        random.seed(seed)
    if spacing == 'fixed':
        # Place sensors at distances: fixed_gap, 2*fixed_gap, ..., n*fixed_gap
        positions = [fixed_gap * i for i in range(1, n+1)]
    elif spacing == 'random':
        # Place sensors at random positions in the range [0, max_range]
        positions = sorted(random.uniform(0, max_range) for _ in range(n))
    else:
        raise ValueError("spacing must be 'fixed' or 'random'")
    return positions

def assign_sensors_to_robots(positions, k, T_d, v):
    """
    Assign sensors to k robots optimally using dynamic programming.
    Returns (segments, times, mission_time):
      - segments: list of (start_index, end_index) for each robot's sensor segment (1-indexed sensor indices).
      - times: list of times (travel + download) for each robot.
      - mission_time: the total mission time (max of times).
    """
    # Ensure sensors are sorted by position
    positions = sorted(positions)
    n = len(positions)
    if n == 0 or k == 0:
        return [], [], 0.0

    # Limit k to n (more robots than sensors means some robots are idle)
    k_eff = min(k, n)
    # Initialize DP table and back-pointer table
    DP = [[math.inf] * (k_eff + 1) for _ in range(n + 1)]
    prev_index = [[None] * (k_eff + 1) for _ in range(n + 1)]
    # Base cases: 0 sensors -> 0 time
    for m in range(k_eff + 1):
        DP[0][m] = 0.0
    # Base case for 1 robot: cover first i sensors (travel to farthest + download all i sensors)
    for i in range(1, n + 1):
        DP[i][1] = positions[i-1] / v + i * T_d
        prev_index[i][1] = 0  # 0 indicates that this one robot covers all up to i (no prior split)

    # Fill DP table for m = 2 to k_eff robots
    for m in range(2, k_eff + 1):
        for i in range(1, n + 1):
            if i < m:
                # Not enough sensors to assign to m robots (at least one sensor per robot),
                # so effectively treat it as using i robots for i sensors.
                DP[i][m] = DP[i][i]
                prev_index[i][m] = i - 1
            else:
                # Try all possible split points l between the first (m-1) sensors and the last segment
                best_time = math.inf
                best_l = None
                for l in range(m-1, i):  # l is end index for first m-1 robots, so at least m-1 sensors
                    # Time for first m-1 robots to cover sensors 1..l
                    time_first = DP[l][m-1]
                    # Time for m-th robot to cover sensors (l+1)..i
                    # travel to sensor i (farthest in this segment) + download (i - l) sensors
                    travel_time = positions[i-1] / v
                    download_time = (i - l) * T_d
                    time_second = travel_time + download_time
                    # The mission time if split at l is the slower of the two parts
                    mission_time = max(time_first, time_second)
                    if mission_time < best_time:
                        best_time = mission_time
                        best_l = l
                DP[i][m] = best_time
                prev_index[i][m] = best_l

    # Reconstruct the optimal segments from prev_index table
    segments = []
    times = []
    i = n
    m = k_eff
    while m > 0 and i > 0:
        l = prev_index[i][m]
        if l is None:
            l = i - 1  # fallback (should not happen under normal conditions)
        # Robot m covers sensors (l+1) through i
        segments.append((l+1, i))
        # Calculate that robot's time (travel to sensor i + downloads)
        seg_travel = positions[i-1] / v
        seg_count = i - l
        seg_time = seg_travel + seg_count * T_d
        times.append(seg_time)
        # Move to previous group of sensors for the remaining (m-1) robots
        i = l
        m -= 1
    segments.reverse()
    times.reverse()

    # If there are more robots than sensors, mark remaining robots as idle (no sensors)
    if k_eff < k:
        for extra in range(k_eff, k):
            segments.append((None, None))
            times.append(0.0)

    mission_time = DP[len(positions)][k_eff]
    return segments, times, mission_time

# Example usage:
if __name__ == "__main__":
    # Define parameters
    n = 10           # number of sensors
    k = 3            # number of robots
    T_d = 5.0        # download time per sensor
    v = 1.0          # robot speed (distance units per time unit)
    # Generate sensor positions (randomly in [0,100] for this example)
    sensor_positions = generate_sensor_positions(n, spacing="random", max_range=100, seed=42)
    print(f"Sensor positions (sorted): {sensor_positions}")
    # Compute optimal assignment
    segments, times, mission_time = assign_sensors_to_robots(sensor_positions, k, T_d, v)
    # Display results
    for idx, (seg, t) in enumerate(zip(segments, times), start=1):
        if seg[0] is None:
            print(f"Robot {idx}: no sensors assigned (idle)")
        else:
            start, end = seg
            print(f"Robot {idx}: sensors {start} to {end} -> time = {t:.2f}")
    print(f"Overall mission time (completion time) = {mission_time:.2f}")
    # Plot the sensor assignments for visualization
    plt.figure()
    # Mark base station at 0
    plt.scatter(0, 0, c='k', marker='*', s=100, label='Base Station')
    # Plot sensors for each robot's segment in different colors
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']  # default color cycle
    for idx, seg in enumerate(segments, start=1):
        if seg[0] is None:
            continue  # skip idle robots
        start, end = seg
        segment_positions = sensor_positions[start-1 : end]
        plt.scatter(segment_positions, [0]*len(segment_positions), 
                    color=color_cycle[idx % len(color_cycle)], marker='x', s=60, label=f'Robot {idx}')
    plt.xlabel("Sensor Position (distance from base)")
    plt.yticks([])  # hide y-axis
    plt.axhline(y=0, color='gray', linewidth=0.8)
    plt.legend(loc='upper center', ncol=k+1, bbox_to_anchor=(0.5, 1.15))
    plt.tight_layout()
    plt.show()
