import argparse
import os
import sys
from typing import List, Tuple
import numpy as np
from evo.tools import file_interface, plot
from evo.tools.plot import PlotMode, prepare_axis, traj
from py_factor_graph.modifiers import (
    split_single_robot_into_multi,
    add_landmark_at_position,
    RangeMeasurementModel
)
from py_factor_graph.io.g2o_file import parse_3d_g2o_file
from py_factor_graph.io.pyfg_file import save_to_pyfg_file
from py_factor_graph.io.tum_file import save_robot_trajectories_to_tum_file
from py_factor_graph.utils.logging_utils import logger
from itertools import product

from py_factor_graph.factor_graph import FactorGraphData

ROBOT_CASES = [2, 5, 10]
NOISE_STDDEV = [0.10, 0.05]

# Landmark cases are hard-coded; the first 8 landmarks are arranged as a cube that contains 
# the sphere encoded by the g2o file; the next 8 landmarks are arranged as a cube that
# resides inside the first cube, but still contains the sphere.

# sphere2500.g2o; observation of xyz data shows that the sphere is centered at 
# (x = 0.0681175550956	y=-1.0389495768684 z=-50.0233964188792) and radius of ~50m.

CUBE_RADIUS = 70.0
TILTED_CUBE_RADIUS = 60.0
CUBE_CENTER = (0.0681175550956, -1.0389495768684, -50.0233964188792)
ORIGIN = (0, 0, 0)

SENSING_HORIZON = 50.0
MEAS_PROB = 1.0

def generate_datasets(args) -> None:
    data_fp = args.dataset
    output_dir = args.output_dir

    def generate_cube_coords(radius: float, center: Tuple[float, float, float]) -> List[Tuple[float, float, float]]:
        """Generate eight coordinates for a cube.

        Args:
            radius (float): radius of the cube
            center (Tuple[float, float, float]): center of the cube
        
        Returns:
            List[Tuple[float, float, float]]: list of coordinates
        """
        x, y, z = center
        return [
            (x + radius, y + radius, z + radius),
            (x + radius, y + radius, z - radius),
            (x + radius, y - radius, z + radius),
            (x + radius, y - radius, z - radius),
            (x - radius, y + radius, z + radius),
            (x - radius, y + radius, z - radius),
            (x - radius, y - radius, z + radius),
            (x - radius, y - radius, z - radius),
        ]

    def generate_tilted_cube_coords(radius: float, center: Tuple[float, float, float]) -> List[Tuple[float, float, float]]:
        """Generate eight coordinates for a cube that is tilted 45 degrees on the z-axis.

        Args:
            radius (float): radius of the cube
            center (Tuple[float, float, float]): center of the cube
        
        Returns:
            List[Tuple[float, float, float]]: list of coordinates
        """

        # Hard-coded to rotation of 45 degres on the z-axis
        rot_z = np.matrix([[0.7071, -0.7071, 0.0],
                           [0.7071, 0.7071, 0.0],
                           [0.0, 0.0, 1.0]])

        cube_coords = generate_cube_coords(radius, ORIGIN)
        tilted_cube_coords = []

        # Convert each coord tuple into a np.ndarray and perform matrix rotation with rot_z
        # First rotate, then translate to specified center
        for cube_coord in cube_coords:
            coord = np.asarray(cube_coord)
            rot_coord = tuple(np.squeeze(np.asarray(rot_z @ coord)))
            tilted_cube_coords.append(tuple(map(lambda x, y: x + y, center, rot_coord)))
        return tilted_cube_coords

    
    def add_landmarks_to_fg(fg: FactorGraphData, coords: List[Tuple[float, float, float]], range_model: RangeMeasurementModel) -> FactorGraphData:
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
    # save_robot_trajectories_to_tum_file(fg, f"{output_dir}", use_ground_truth=False)

    prod = product(ROBOT_CASES, NOISE_STDDEV)
    cube_coords = generate_cube_coords(CUBE_RADIUS, CUBE_CENTER)
    tilted_cube_coords = generate_tilted_cube_coords(TILTED_CUBE_RADIUS, CUBE_CENTER)
    for i, tup in enumerate(prod):
        num_robots, noise = tup
        logger.info(f"Creating synthetic datasets for {num_robots} robots with noise stddev {noise}m")
        range_model = RangeMeasurementModel(SENSING_HORIZON, noise, MEAS_PROB)

        case_dir = os.path.join(output_dir, f"{num_robots}_robots_{noise}_stddev")

        if (not os.path.isdir(case_dir)):
            os.makedirs(case_dir)

        fg_mod = split_single_robot_into_multi(fg, num_robots)
        fg_mod = add_landmarks_to_fg(fg_mod, cube_coords, range_model)
        save_to_pyfg_file(fg_mod, f"{case_dir}/{num_robots}_robots_{noise}_stddev_{len(cube_coords)}_landmarks.pyfg")
        fg_mod = add_landmarks_to_fg(fg_mod, tilted_cube_coords, range_model)
        save_to_pyfg_file(fg_mod, f"{case_dir}/{num_robots}_robots_{noise}_stddev_{len(cube_coords) + len(tilted_cube_coords)}_landmarks.pyfg")
        # save_robot_trajectories_to_tum_file(fg_mod, f"{case_dir}", use_ground_truth=False) # For visualization; saves case with 16 range measurements to TUM

def main(args):
    parser = argparse.ArgumentParser(
        description="This script is used to generate multi-robot synthetic datasets with range measurements from an existing g2o file."
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        required=True,
        help="g2o filepath to create synthetic datasets from"
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=True,
        help="directory where evaluation results are saved",
    )

    args = parser.parse_args()
    generate_datasets(args)

if __name__ == "__main__":
    main(sys.argv[1:])