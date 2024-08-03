import argparse
import os
import sys
from py_factor_graph.modifiers import (
    split_single_robot_into_multi_and_add_ranges,
    RangeMeasurementModel
)
from py_factor_graph.io.g2o_file import parse_3d_g2o_file
from py_factor_graph.io.pyfg_file import save_to_pyfg_file
from py_factor_graph.utils.logging_utils import logger
from itertools import product

# How to use split_single_robot_into_multi_and_add_ranges?
# split_single_robot_into_multi: 
# add_inter_robot_range_measurements:

ROBOT_CASES = [2, 5, 10]
NOISE_STDDEV = [0.10, 0.05]

# Landmark cases are hardcoded; the first 8 landmarks are arranged as a cube that contains 
# the sphere encoded by the g2o file; the next 8 landmarks are arranged as a cube that
# resides inside the first cube, but still contains the sphere.

SENSING_HORIZON = 25.0
MEAS_PROB = 1.0

def generate_datasets(args) -> None:
    data_fp = args.dataset
    output_dir = args.output_dir

    assert data_fp.endswith(".g2o"), logger.critical(
        "Dataset must be in g2o format."
    )

    if (not os.path.isdir(output_dir)):
        os.makedirs(output_dir)

    fg = parse_3d_g2o_file(data_fp)

    prod = product(ROBOT_CASES, NOISE_STDDEV)
    for num_robots, noise in prod:
        logger.info(f"Creating synthetic datasets for {num_robots} robots with noise stddev {noise}m")
        range_model = RangeMeasurementModel(SENSING_HORIZON, noise, MEAS_PROB)

        fg_multi = split_single_robot_into_multi_and_add_ranges(fg, num_robots, range_model)
        save_to_pyfg_file(fg_multi, f"{output_dir}/{num_robots}_robots_{noise}.pyfg")

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