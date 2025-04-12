# path_planning.py
import math
from heapq import heappush, heappop

def a_star(start, goal, obstacles, grid_size):
    """
    Compute shortest path from start to goal on a grid using A*,
    avoiding cells in the `obstacles` set. 
    Returns a list of (x, y) coordinates for the path or None if no path exists.
    """
    sx, sy = start
    gx, gy = goal
    if start == goal:
        return [start]
    # If start or goal is blocked, no path
    if start in obstacles or goal in obstacles:
        return None

    # Heuristic function: Manhattan distance (L1 distance)
    def heuristic(x, y):
        return abs(x - gx) + abs(y - gy)
    
    # Open set: priority queue of (f_score, g_score, (x, y), parent)
    open_set = []
    heappush(open_set, (heuristic(sx, sy), 0, (sx, sy), None))
    # Dictionaries to store best g_score and parent for each visited cell
    g_score = { (sx, sy): 0 }
    parent = { (sx, sy): None }
    closed = set()

    while open_set:
        f, g, current, prev = heappop(open_set)
        if current in closed:
            continue
        parent[current] = prev
        closed.add(current)
        cx, cy = current
        # Goal found
        if current == (gx, gy):
            # Reconstruct path by following parents
            path = []
            node = current
            while node is not None:
                path.append(node)
                node = parent[node]
            path.reverse()
            return path
        # Explore neighbors (4-directional moves)
        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            nx, ny = cx + dx, cy + dy
            neighbor = (nx, ny)
            # Check bounds
            if nx < 0 or ny < 0 or nx >= grid_size or ny >= grid_size:
                continue
            # Skip if obstacle or already processed
            if neighbor in obstacles or neighbor in closed:
                continue
            # Tentative g cost to neighbor
            new_g = g + 1
            # If this route to neighbor is better than any previous one, record it
            if new_g < g_score.get(neighbor, math.inf):
                g_score[neighbor] = new_g
                f_score = new_g + heuristic(nx, ny)
                heappush(open_set, (f_score, new_g, neighbor, current))
    # No path found
    return None
