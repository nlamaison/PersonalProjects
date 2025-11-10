"""
Generate a demo simulation world with specified parameters.
"""
import argparse

import generate_sim_world

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(__name__)
    parser.add_argument(
        "--x_bounds",
        required=True,
        nargs=2,
        type=int,
        help="Comma-separated values values representing the (lower, upper) x-coordinate bounds of the workspace."
    )
    parser.add_argument(
        "--y_bounds",
        required=True,
        nargs=2,
        type=int,
        help="Comma-separated values values representing the (lower, upper) y-coordinate bounds of the workspace."
    )
    parser.add_argument(
        "--n_obstacles",
        required=True,
        type=int,
        help="Number of obstacles to randomly generate in the sim world."
    )
    parser.add_argument(
        "--plot_filepath",
        required=False,
        type=str,
        help="Save the plot of the simulation world to a file at the given path."
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    x_bounds = args.x_bounds
    y_bounds = args.y_bounds

    sim_world = generate_sim_world.generate_mock_sim_world(x_bounds, y_bounds, args.n_obstacles)
    generate_sim_world.plot_sim_world(sim_world, args.plot_filepath)

