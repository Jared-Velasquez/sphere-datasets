import argparse
import math
import os
import sys
from typing import List, Tuple
import numpy as np
from evo.tools import file_interface, plot
from evo.tools.plot import PlotMode, prepare_axis, traj
from py_factor_graph.modifiers import (
    split_single_robot_into_multi,
    add_landmark_at_position,
    add_inter_robot_range_measurements,
    RangeMeasurementModel
)
from py_factor_graph.io.g2o_file import parse_3d_g2o_file
from py_factor_graph.io.pyfg_file import save_to_pyfg_file
from py_factor_graph.io.tum_file import save_robot_trajectories_to_tum_file
from py_factor_graph.utils.logging_utils import logger
from itertools import product

from py_factor_graph.factor_graph import FactorGraphData

ROBOT_CASES = [2, 4, 8]
NOISE_STDDEV_CASES = [0.10]
MEAS_PROB_CASES = [0.5, 1.0]

DEFAULT_PYFG_FILENAME = "synthetic_dataset"

CUBE_RADIUS = 50.0
TILTED_CUBE_RADIUS = 50.0/math.sqrt(2)

SPHERE2500_SE_SYNC_COORDS_1 = [(-50, 50, -50), (50, -50, -50), (-50, -50, -50), (50, 50, -50), (-50, 50, 50), (50, -50, 50), (-50, -50, -50), (50, 50, 50)]
SPHERE2500_SE_SYNC_COORDS_2 = [(-50, 0, -50), (0, -50, -50), (0, 50, -50), (50, 0, -50), (-50, 0, 50), (0, -50, 50), (0, 50, 50), (50, 0, 50)]
SMALLGRID3D_SE_SYNC_COORDS_1 = [(-3, 3, -3), (3, -3, -3), (-3, -3, -3), (3, 3, -3), (-3, 3, 3), (3, -3, 3), (-3, -3, -3), (3, 3, 3)]
SMALLGRID3D_SE_SYNC_COORDS_2 = [(-3, 0, -3), (0, -3, -3), (0, 3, -3), (3, 0, -3), (-3, 0, 3), (0, -3, 3), (0, 3, 3), (3, 0, 3)]

ORIGIN = (0, 0, 0)

def generate_datasets(args) -> None:
    data_fp = args.dataset
    output_dir = args.output_dir
    sensing_horizon = args.sensing
    inter_robot_sensing_horizon = args.inter_robot_sensing
    add_robot_robot_range_measurements = args.inter_robot
    cube_radius = args.cube
    tilted_cube_radius = args.tilted_cube

    def generate_cube_coords(radius: float, center: Tuple[float, float, float]) -> List[Tuple[float, float, float]]:
        """Generate eight coordinates for a cube.

        Args:
            radius (float): radius of the cube
            center (Tuple[float, float, float]): center of the cube
        
        Returns:
            List[Tuple[float, float, float]]: list of coordinates (x, y, z)
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
            List[Tuple[float, float, float]]: list of coordinates (x, y, z)
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
        """Adds a list of landmarks, where each landmark position is indicated by a tuple (x, y, z)

        Args:
            fg (FactorGraphData): factor graph that will be modified
            coords (List[Tuple[float, float, float]]): list of coordinates (x, y, z)
            range_model (RangeMeasurementModel): Sensor model for range measurements
        
        Returns:
            FactorGraphData: modified factor graph
        """
        new_fg = fg
        for coord in coords:
            new_fg = add_landmark_at_position(new_fg, np.array(coord), range_model)
        return new_fg
    
    def name_pyfg_file(inter_robot_enabled: bool = False, num_robots: int = None, noise: float = None, meas_prob: float = None, num_landmarks: int = None, add_extension: bool = True) -> str:
        name = ""
        if inter_robot_enabled:
            name += "inter_robot_"
        if num_robots is not None:
            name += f"{num_robots}_robots_"
        if noise is not None:
            name += f"{noise}_stddev_"
        if meas_prob is not None:
            name += f"{meas_prob}_meas_prob_"
        if num_landmarks is not None:
            name += f"{num_landmarks}_landmarks_"
        if len(name) != 0 and name[-1] == "_":
            name = name[:-1]
        if len(name) == 0:
            name = DEFAULT_PYFG_FILENAME
        if add_extension:
            name += ".pyfg"

        return name

    assert data_fp.endswith(".g2o"), logger.critical(
        "Dataset must be in g2o format."
    )

    if (not os.path.isdir(output_dir)):
        os.makedirs(output_dir)

    fg = parse_3d_g2o_file(data_fp)
    # save_robot_trajectories_to_tum_file(fg, f"{output_dir}", use_ground_truth=False)

    prod = product(ROBOT_CASES, NOISE_STDDEV_CASES, MEAS_PROB_CASES)
    tilted_cube_coords = generate_tilted_cube_coords(TILTED_CUBE_RADIUS, ORIGIN)
    for i, tup in enumerate(prod):
        num_robots, noise, meas_prob = tup
        logger.info(f"Creating synthetic datasets for {num_robots} robots with noise stddev {noise}m")
        range_model = RangeMeasurementModel(sensing_horizon, noise, meas_prob)
        inter_robot_range_model = RangeMeasurementModel(inter_robot_sensing_horizon, noise, meas_prob)

        case_dir = os.path.join(output_dir, name_pyfg_file(add_robot_robot_range_measurements, num_robots, noise, meas_prob, add_extension = False))

        if (not os.path.isdir(case_dir)):
            os.makedirs(case_dir)

        fg_mod = split_single_robot_into_multi(fg, num_robots)
        # save_to_pyfg_file(fg_mod, f"{case_dir}/{name_pyfg_file(False, num_robots, noise)}")
        # if (add_robot_robot_range_measurements):
        #      fg_mod = add_inter_robot_range_measurements(fg_mod, inter_robot_range_model)
        #      save_to_pyfg_file(fg_mod, f"{case_dir}/{name_pyfg_file(add_robot_robot_range_measurements, num_robots, noise)}")
        if (cube_radius is not None):
            cube_coords = generate_cube_coords(cube_radius, ORIGIN)
            fg_mod = add_landmarks_to_fg(fg_mod, cube_coords, range_model)
            save_to_pyfg_file(fg_mod, f"{case_dir}/{name_pyfg_file(add_robot_robot_range_measurements, num_robots, noise, fg_mod.num_landmarks)}")
        if (tilted_cube_radius is not None):
            tilted_cube_coords = generate_tilted_cube_coords(tilted_cube_radius, ORIGIN)
            fg_mod = add_landmarks_to_fg(fg_mod, tilted_cube_coords, range_model)
            save_to_pyfg_file(fg_mod, f"{case_dir}/{name_pyfg_file(add_robot_robot_range_measurements, num_robots, noise, fg_mod.num_landmarks)}")

def main(args):
    parser = argparse.ArgumentParser(
        description="This script is used to generate synthetic multi-robot datasets with range measurements from an existing g2o file."
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
    parser.add_argument(
        "-s",
        "--sensing",
        type=float,
        default=1.0,
        help="sensing horizon for robot-landmark range measurements"
    )
    parser.add_argument(
        "-c",
        "--cube",
        type=float,
        help="radius of a cube where each vertex is a landmark"
    )
    parser.add_argument(
        "-t",
        "--tilted_cube",
        type=float,
        help="radius of a cube tilted 45 degrees ccw where each vertex is a landmark"
    )
    parser.add_argument(
        "-i",
        "--inter_robot",
        action="store_true",
        help="add inter-robot range measurements"
    )
    parser.add_argument(
        "-r",
        "--inter_robot_sensing",
        type=float,
        default=1.0,
        help="sensing horizon for inter-robot range measurements"
    )

    args = parser.parse_args()
    generate_datasets(args)

if __name__ == "__main__":
    main(sys.argv[1:])