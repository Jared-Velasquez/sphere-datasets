import sys
sys.path.append('../')

import os
import argparse
import math
import sys
from typing import Dict, List, Tuple
import numpy as np
from utils.modifiers import add_landmarks_to_fg
from utils.name_utils import name_pyfg_file
from utils.coord_utils import generate_cube_coords, generate_tilted_cube_coords

from evo.tools import file_interface, plot
from evo.tools.plot import PlotMode, prepare_axis, traj
from py_factor_graph.modifiers import (
    split_single_robot_into_multi,
    add_landmark_at_position,
    add_inter_robot_range_measurements,
    RangeMeasurementModel
)
from py_factor_graph.io.g2o_file import parse_3d_g2o_file
from py_factor_graph.io.pyfg_file import save_to_pyfg_file, read_from_pyfg_file
from py_factor_graph.io.tum_file import save_robot_trajectories_to_tum_file
from py_factor_graph.utils.name_utils import get_robot_char_from_number, get_robot_idx_from_char
from py_factor_graph.utils.logging_utils import logger
from itertools import product

from py_factor_graph.factor_graph import FactorGraphData

ROBOT_CASES = [2, 4, 8]
NOISE_STDDEV_CASES = [0.10]
MEAS_PROB_CASES = [0.5, 1.0]

CUBE_RADIUS = 50.0
TILTED_CUBE_RADIUS = 50.0/math.sqrt(2)

SPHERE2500_SE_SYNC_FILENAME = "sphere2500_se_synch_gt.g2o"
SPHERE2500_SE_SYNC_COORDS_1 = [(-50, 50, -50), (50, -50, -50), (-50, -50, -50), (50, 50, -50), (-50, 50, 50), (50, -50, 50), (-50, -50, -50), (50, 50, 50)]
SPHERE2500_SE_SYNC_COORDS_2 = [(-50, 0, -50), (0, -50, -50), (0, 50, -50), (50, 0, -50), (-50, 0, 50), (0, -50, 50), (0, 50, 50), (50, 0, 50)]
SPHERE2500_SE_SYNC_SENSING_HORIZON = math.sqrt(50.0**2 + 50.0**2 + 50.0**2)

SMALLGRID3D_SE_SYNC_FILENAME = "smallGrid3D_se_synch_gt.g2o"
SMALLGRID3D_SE_SYNC_COORDS_1 = [(-3, 3, -3), (3, -3, -3), (-3, -3, -3), (3, 3, -3), (-3, 3, 3), (3, -3, 3), (-3, -3, -3), (3, 3, 3)]
SMALLGRID3D_SE_SYNC_COORDS_2 = [(-3, 0, -3), (0, -3, -3), (0, 3, -3), (3, 0, -3), (-3, 0, 3), (0, -3, 3), (0, 3, 3), (3, 0, 3)]
SMALLGRID3D_SE_SYNC_SENSING_HORIZON = math.sqrt(3.0**2 + 3.0**2 + 3.0**2)

ORIGIN = (0, 0, 0)

def generate_sphere2500_se_sync_datasets(args) -> None:
    data_fp = args.dataset
    output_dir = args.output_dir

    filename = os.path.basename(data_fp)
    assert filename == SPHERE2500_SE_SYNC_FILENAME, logger.critical(
        f"Dataset must be {SPHERE2500_SE_SYNC_FILENAME}"
    )

    if (not os.path.isdir(output_dir)):
        os.makedirs(output_dir)

    fg = parse_3d_g2o_file(data_fp)

    prod = product(ROBOT_CASES, NOISE_STDDEV_CASES, MEAS_PROB_CASES)
    for i, tup in enumerate(prod):
        num_robots, noise, meas_prob = tup
        logger.info(f"Creating synthetic datasets for {num_robots} robots with noise stddev {noise}m and measurement probability {meas_prob}")
        range_model = RangeMeasurementModel(SPHERE2500_SE_SYNC_SENSING_HORIZON, noise, meas_prob)
        inter_robot_range_model = RangeMeasurementModel(SPHERE2500_SE_SYNC_SENSING_HORIZON, noise, meas_prob)

        case_dir = os.path.join(output_dir, name_pyfg_file(False, num_robots, noise, meas_prob, add_extension = False))

        if (not os.path.isdir(case_dir)):
            os.makedirs(case_dir)

        fg_mod = split_single_robot_into_multi(fg, num_robots)

        # sphere2500_se_synch datasets
        fg_mod = add_inter_robot_range_measurements(fg_mod, inter_robot_range_model)
        fg_mod = add_landmarks_to_fg(fg_mod, SPHERE2500_SE_SYNC_COORDS_1, range_model)
        save_to_pyfg_file(fg_mod, f"{case_dir}/{name_pyfg_file(True, num_robots, noise, meas_prob, fg_mod.num_landmarks)}")
        fg_mod = add_landmarks_to_fg(fg_mod, SPHERE2500_SE_SYNC_COORDS_1, range_model)
        save_to_pyfg_file(fg_mod, f"{case_dir}/{name_pyfg_file(True, num_robots, noise, meas_prob, fg_mod.num_landmarks)}")

def generate_smallgrid3d_se_sync_datasets(args) -> None:
    data_fp = args.dataset
    output_dir = args.output_dir

    filename = os.path.basename(data_fp)
    assert filename == SMALLGRID3D_SE_SYNC_FILENAME, logger.critical(
        f"Dataset must be {SMALLGRID3D_SE_SYNC_FILENAME}"
    )

    if (not os.path.isdir(output_dir)):
        os.makedirs(output_dir)

    fg = parse_3d_g2o_file(data_fp)

    prod = product(ROBOT_CASES, NOISE_STDDEV_CASES, MEAS_PROB_CASES)
    for i, tup in enumerate(prod):
        num_robots, noise, meas_prob = tup
        logger.info(f"Creating synthetic datasets for {num_robots} robots with noise stddev {noise}m and measurement probability {meas_prob}")
        range_model = RangeMeasurementModel(SMALLGRID3D_SE_SYNC_SENSING_HORIZON, noise, meas_prob)
        inter_robot_range_model = RangeMeasurementModel(SMALLGRID3D_SE_SYNC_SENSING_HORIZON, noise, meas_prob)

        case_dir = os.path.join(output_dir, name_pyfg_file(False, num_robots, noise, meas_prob, add_extension = False))

        if (not os.path.isdir(case_dir)):
            os.makedirs(case_dir)

        fg_mod = split_single_robot_into_multi(fg, num_robots)

        # smallGrid3D_se_synch datasets
        fg_mod = add_inter_robot_range_measurements(fg_mod, inter_robot_range_model)
        fg_mod = add_landmarks_to_fg(fg_mod, SMALLGRID3D_SE_SYNC_COORDS_1, range_model)
        save_to_pyfg_file(fg_mod, f"{case_dir}/{name_pyfg_file(True, num_robots, noise, meas_prob, fg_mod.num_landmarks)}")
        fg_mod = add_landmarks_to_fg(fg_mod, SMALLGRID3D_SE_SYNC_COORDS_2, range_model)
        save_to_pyfg_file(fg_mod, f"{case_dir}/{name_pyfg_file(True, num_robots, noise, meas_prob, fg_mod.num_landmarks)}")

def generate_datasets(args) -> List[str]:
    data_fp = args.dataset
    output_dir = args.output_dir
    sensing_horizon = args.sensing
    inter_robot_sensing_horizon = args.inter_robot_sensing
    add_robot_robot_range_measurements = args.inter_robot
    cube_radius = args.cube
    tilted_cube_radius = args.tilted_cube

    assert data_fp.endswith(".g2o"), logger.critical(
        "Dataset must be in g2o format."
    )

    if (not os.path.isdir(output_dir)):
        os.makedirs(output_dir)

    fg = parse_3d_g2o_file(data_fp)
    # save_robot_trajectories_to_tum_file(fg, f"{output_dir}", use_ground_truth=False)

    prod = product(ROBOT_CASES, NOISE_STDDEV_CASES, MEAS_PROB_CASES)
    tilted_cube_coords = generate_tilted_cube_coords(TILTED_CUBE_RADIUS, ORIGIN)

    synthetic_dataset_filepaths: List[str] = []
    for i, tup in enumerate(prod):
        num_robots, noise, meas_prob = tup
        logger.info(f"Creating synthetic datasets for {num_robots} robots with noise stddev {noise}m and measurement probability {meas_prob}")
        range_model = RangeMeasurementModel(sensing_horizon, noise, meas_prob)
        inter_robot_range_model = RangeMeasurementModel(inter_robot_sensing_horizon, noise, meas_prob)

        case_dir = os.path.join(output_dir, name_pyfg_file(add_robot_robot_range_measurements, num_robots, noise, meas_prob, add_extension = False))

        if (not os.path.isdir(case_dir)):
            os.makedirs(case_dir)

        fg_mod = split_single_robot_into_multi(fg, num_robots)
        multi_robot_filepath = f"{case_dir}/{name_pyfg_file(False, num_robots, noise)}"
        save_to_pyfg_file(fg_mod, multi_robot_filepath)
        synthetic_dataset_filepaths.append(multi_robot_filepath)

        # Add inter-robot range measurements
        if (add_robot_robot_range_measurements):
            fg_mod = add_inter_robot_range_measurements(fg_mod, inter_robot_range_model)
            inter_robot_filepath = f"{case_dir}/{name_pyfg_file(True, num_robots, noise, meas_prob, fg_mod.num_landmarks)}"
            save_to_pyfg_file(fg_mod, inter_robot_filepath)
            synthetic_dataset_filepaths.append(inter_robot_filepath)

        # Add landmarks
        if (cube_radius is not None):
            cube_coords = generate_cube_coords(cube_radius, ORIGIN)
            fg_mod = add_landmarks_to_fg(fg_mod, cube_coords, range_model)
            cube_filepath = f"{case_dir}/{name_pyfg_file(add_robot_robot_range_measurements, num_robots, noise, meas_prob, fg_mod.num_landmarks)}"
            save_to_pyfg_file(fg_mod, cube_filepath)
            synthetic_dataset_filepaths.append(cube_filepath)

        if (tilted_cube_radius is not None):
            tilted_cube_coords = generate_tilted_cube_coords(tilted_cube_radius, ORIGIN)
            fg_mod = add_landmarks_to_fg(fg_mod, tilted_cube_coords, range_model)
            tilted_cube_filepath = f"{case_dir}/{name_pyfg_file(add_robot_robot_range_measurements, num_robots, noise, meas_prob, fg_mod.num_landmarks)}"
            save_to_pyfg_file(fg_mod, tilted_cube_filepath)
            synthetic_dataset_filepaths.append(tilted_cube_filepath)
    
    return synthetic_dataset_filepaths

def landmark_connectivity(synthetic_dataset_filepaths) -> None:

    def append_frequency_to_csv(frequency: Dict[str, List[int]], num_robots: int, csv_fp: str) -> None:
        """Append the frequency of each robot seeing each landmark to a CSV file.

        Args:
            frequency (Dict[str, List[int]]): frequency of each robot seeing each landmark
            csv_fp (str): filepath to save the CSV file

        Returns:
            None
        """
        with open(csv_fp, "w") as f:
            f.write("landmark_symbol")
            for i in range(num_robots):
                f.write(f",{get_robot_char_from_number(i)}_count")
            f.write("\n")

            for landmark, freq in frequency.items():
                f.write(landmark)
                for count in freq:
                    f.write(f",{count}")
                f.write("\n")

    for synthetic_dataset_filepath in synthetic_dataset_filepaths:
        fg = read_from_pyfg_file(synthetic_dataset_filepath)

        if (fg.num_landmarks == 0):
            continue
        robot_landmark_frequency = {}

        logger.info(f"Counting frequency of connectivity between robots and landmarks in {synthetic_dataset_filepath}")

        # For each landmark, initialize array where each element corresponds to the frequency of a robot seeing a landmark
        for i in range(fg.num_landmarks):
            landmark = f"L{i}"
            robot_landmark_frequency[landmark] = [0] * fg.num_robots

        range_measurements = fg.range_measurements
        for measure in range_measurements:
            # Skip over inter-robot range measurements
            if measure.association[1][0] != "L":
                continue

            landmark = measure.association[1]
            robot = measure.association[0]

            robot_idx = get_robot_idx_from_char(robot[0])
            robot_landmark_frequency[landmark][robot_idx] += 1

        # Save the frequency of each robot seeing each landmark to a CSV file
        base = os.path.basename(synthetic_dataset_filepath)
        output_dir = os.path.dirname(synthetic_dataset_filepath)
        frequency_csv_file = os.path.join(output_dir, f"{os.path.splitext(base)[0]}_landmark_connectivity.csv")

        append_frequency_to_csv(robot_landmark_frequency, fg.num_robots, frequency_csv_file)

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
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="assign ownership of landmarks to the robot that has the highest frequency of seeing the landmark",
    )

    args = parser.parse_args()
    synthetic_dataset_filepaths = generate_datasets(args)
    landmark_connectivity(synthetic_dataset_filepaths)

if __name__ == "__main__":
    main(sys.argv[1:])