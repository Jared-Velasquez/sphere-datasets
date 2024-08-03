import argparse
import os
import sys
from typing import List, Tuple
import numpy as np
from py_factor_graph.modifiers import (
    split_single_robot_into_multi,
    add_landmark_at_position,
    RangeMeasurementModel
)
from py_factor_graph.io.g2o_file import parse_3d_g2o_file
from py_factor_graph.io.pyfg_file import save_to_pyfg_file
from py_factor_graph.utils.logging_utils import logger
from itertools import product

from py_factor_graph.factor_graph import FactorGraphData

# How to use split_single_robot_into_multi_and_add_ranges?
# split_single_robot_into_multi: 
# add_inter_robot_range_measurements:

# split_single_robot_into_multi_and_add_ranges adds robot-robot range measurements,
# not robot-landmark range measurements.

# Could use add_landmark_at_position?

ROBOT_CASES = [2, 5, 10]
NOISE_STDDEV = [0.10, 0.05]
CUBE_RADIUS = 10.0
TILTED_CUBE_RADIUS = 5.0

# Landmark cases are hardcoded; the first 8 landmarks are arranged as a cube that contains 
# the sphere encoded by the g2o file; the next 8 landmarks are arranged as a cube that
# resides inside the first cube, but still contains the sphere.

SENSING_HORIZON = 25.0
MEAS_PROB = 1.0

def generate_datasets(args) -> None:
    data_fp = args.dataset
    output_dir = args.output_dir

    def generate_cube_coords(radius: float) -> List[Tuple[float, float, float]]:
        """Generate eight coordinates for a cube.

        Args:
            radius (float): radius of the cube
        
        Returns:
            List[Tuple[float, float, float]]: list of coordinates
        """
        return [
            (radius, radius, radius),
            (radius, radius, -radius),
            (radius, -radius, radius),
            (radius, -radius, -radius),
            (-radius, radius, radius),
            (-radius, radius, -radius),
            (-radius, -radius, radius),
            (-radius, -radius, -radius),
        ]

    def generate_tilted_cube_coords(radius: float) -> List[Tuple[float, float, float]]:
        """Generate eight coordinates for a cube that is tilted 45 degrees on the z-axis.

        Args:
            radius (float): radius of the cube
        
        Returns:
            List[Tuple[float, float, float]]: list of coordinates
        """
        return [
            (radius, 0, radius),
            (radius, 0, -radius),
            (0, radius, radius),
            (0, radius, -radius),
            (-radius, 0, radius),
            (-radius, 0, -radius),
            (0, -radius, radius),
            (0, -radius, -radius),
        ]
    
    def add_landmarks_to_fg(fg, coords: List[Tuple[float, float, float]], range_model: RangeMeasurementModel) -> FactorGraphData:
        new_fg = fg
        for coord in coords:
            new_fg = add_landmark_at_position(new_fg, np.array(coord), range_model)
        return new_fg

    assert data_fp.endswith(".g2o"), logger.critical(
        "Dataset must be in g2o format."
    )

    if (not os.path.isdir(output_dir)):
        os.makedirs(output_dir)

    fg = parse_3d_g2o_file(data_fp)

    prod = product(ROBOT_CASES, NOISE_STDDEV)
    cube_coords = generate_cube_coords(CUBE_RADIUS)
    tilted_cube_coords = generate_tilted_cube_coords(TILTED_CUBE_RADIUS)
    for num_robots, noise in prod:
        logger.info(f"Creating synthetic datasets for {num_robots} robots with noise stddev {noise}m")
        range_model = RangeMeasurementModel(SENSING_HORIZON, noise, MEAS_PROB)

        fg_mod = split_single_robot_into_multi(fg, num_robots)
        fg_mod = add_landmarks_to_fg(fg_mod, cube_coords, range_model)
        save_to_pyfg_file(fg_mod, f"{output_dir}/{num_robots}_robots_{noise}_stddev_{len(cube_coords)}_landmarks.pyfg")
        fg_mod = add_landmarks_to_fg(fg_mod, tilted_cube_coords, range_model)
        save_to_pyfg_file(fg_mod, f"{output_dir}/{num_robots}_robots_{noise}_stddev_{len(cube_coords) + len(tilted_cube_coords)}_landmarks.pyfg")

def main(args):
    parser = argparse.ArgumentParser(
        description="This script is used to generate multi-robot synthetic datasets with range measurements from an existing g2o file."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="g2o filepath to create synthetic datasets from"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="directory where evaluation results are saved",
    )

    args = parser.parse_args()
    generate_datasets(args)

if __name__ == "__main__":
    main(sys.argv[1:])