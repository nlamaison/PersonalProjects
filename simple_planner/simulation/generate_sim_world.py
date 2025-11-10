"""
Utilities for generating a mock simulation environment.
"""
import dataclasses
import logging
import numpy as np
import matplotlib.pyplot as plt

from shapely import get_coordinates
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from typing import Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger(__name__)

def generate_random_convex_2d_polygon(bounds_x: tuple[float, float], bounds_y: tuple[float, float]) -> Polygon:
    """
    Generate a random convex 2D polygon within the given coordinate bounds. 
    """
    # Don't sample any less than 3 (need a valid polygon) or any more than 10 points (keep things simple/fast) for the convex hull. 
    MIN_N_POINTS = 3
    MAX_N_POINTS = 10

    n_points = np.random.randint(MIN_N_POINTS, MAX_N_POINTS + 1) # max exclusive

    points: list[np.ndarray] = []
    for _ in range(n_points): 
        x = np.random.uniform(bounds_x[0], bounds_x[1])
        y = np.random.uniform(bounds_y[0], bounds_y[1])
        points.append(np.array([x, y]))

    raw_polygon = Polygon(points)
    assert raw_polygon.area > 0, "Polygon has area <= 0. Something has gone horribly wrong!"

    return Polygon(get_coordinates(raw_polygon.convex_hull))


@dataclasses.dataclass
class SimWorld:
    bounds_x: tuple[float, float]
    bounds_y: tuple[float, float]
    obstacles: list[Polygon]


def generate_mock_sim_world(bounds_x: tuple[float, float], bounds_y: tuple[float, float], n_obstacles: int, guarantee_connectivity: bool = True, max_iterations: int = 100) -> SimWorld:
    """
    Generate a mock sim world with obstacles that the robot can plan in.
    By default, the sim world will be generated such that there is connectivity between any non-obstacle point.
    """
    assert n_obstacles < max_iterations, f"Number of obstacles may not be more than the number of max iterations, which is {max_iterations}!"

    # The min/max fraction of space in any dimension an obstacle may occupy. 
    # These parameters help guarantee connectivity of the workspace. 
    MIN_DIM_FRACTION = 0.05 
    MAX_DIM_FRACTION = 0.25

    # Initial free space is just the world box. 
    free_space = box(bounds_x[0], bounds_y[0], bounds_x[1], bounds_y[1])
    # Track the union of all obstacles in the scene.
    union_obstacles = None

    n_iterations = 0
    obstacles: list[Polygon] = []
    while (len(obstacles) < n_obstacles): 
        # Try again from scratch if n_iterations exceeded.
        if n_iterations > max_iterations:
            warning_msg = f"Could not generate a connected free workspace with {n_obstacles} in {max_iterations} iterations. Trying again..."
            LOGGER.warning(warning_msg)
            obstacles.clear()

        # Generate obstacle bounds within the sim world.
        obs_dim_fraction_x = np.random.uniform(MIN_DIM_FRACTION, MAX_DIM_FRACTION)
        obs_profile_x = obs_dim_fraction_x * (bounds_x[1] - bounds_x[0])
        obs_dim_fraction_y = np.random.uniform(MIN_DIM_FRACTION, MAX_DIM_FRACTION)
        obs_profile_y = obs_dim_fraction_y * (bounds_y[1] - bounds_y[0])

        # Compute the x bounds of the obstacle.
        obs_bounds_x_first = np.random.uniform(bounds_x[0], bounds_x[1])
        obs_bounds_x_offset = np.random.choice([1, -1]) * obs_profile_x
        obs_bounds_x_second = obs_bounds_x_first + obs_bounds_x_offset
        if obs_bounds_x_second < bounds_x[0] or obs_bounds_x_second > bounds_x[1]:
            obs_bounds_x_second = obs_bounds_x_first + (-obs_bounds_x_offset)

        obs_bounds_x = (min(obs_bounds_x_first, obs_bounds_x_second), max(obs_bounds_x_first, obs_bounds_x_second))

        # Compute the y bounds of the obstacle.
        obs_bounds_y_first = np.random.uniform(bounds_y[0], bounds_y[1])
        obs_bounds_y_offset = np.random.choice([1, -1]) * obs_profile_y
        obs_bounds_y_second = obs_bounds_y_first + obs_bounds_y_offset
        if obs_bounds_y_second < bounds_y[0] or obs_bounds_y_second > bounds_y[1]:
            obs_bounds_y_second = obs_bounds_y_first + (-obs_bounds_y_offset)

        obs_bounds_y = (min(obs_bounds_y_first, obs_bounds_y_second), max(obs_bounds_y_first, obs_bounds_y_second))
        candidate_obstacle = generate_random_convex_2d_polygon(bounds_x=obs_bounds_x, bounds_y=obs_bounds_y)

        # Add at least 1 obstacle. 
        if len(obstacles) == 0 or union_obstacles is None:
            obstacles.append(candidate_obstacle)
            union_obstacles = unary_union(obstacles)
            continue

        # Does the new obstacle overlap with any other obstacles?
        if union_obstacles.intersection(candidate_obstacle).area > 0.0:
            continue

        # Would the new workspace be connected (or connectivity not required)?
        new_free_space = free_space.difference(union_obstacles)
        if new_free_space.geom_type != "Polygon" and guarantee_connectivity:
            warning_msg = f"Generated a polygon that prevented workspace connectivity. Trying another!"
            LOGGER.warning(warning_msg)
            continue

        # Add a new obstacle!
        obstacles.append(candidate_obstacle)
        union_obstacles = unary_union(obstacles + [candidate_obstacle])
        free_space = new_free_space
            
    return SimWorld(bounds_x=bounds_x, bounds_y=bounds_y, obstacles=obstacles)


def plot_sim_world(sim_world: SimWorld, filepath: str | None = None) -> Any:
    fig = plt.figure(figsize=(5, 5))
    axes = fig.add_subplot(autoscale_on=False, xlim=sim_world.bounds_x, ylim=sim_world.bounds_y)
    axes.set_aspect('equal')

    cmap = plt.get_cmap('plasma')

    for obs in sim_world.obstacles:
        x, y = obs.exterior.xy

        color = cmap(np.random.rand()) 
        axes.fill(x, y, alpha=0.5, color=color)

    if filepath and len(filepath) > 0:
        plt.savefig(filepath)

    plt.show(block=True)
