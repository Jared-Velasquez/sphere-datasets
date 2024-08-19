# synthetic_datasets
Synthetic Datasets: generation of synthetic multi-robot PyFactorGraph files with range measurements from an existing g2o file (with preferrably a spherical trajectory).

## Getting Started

Run the script with:

```bash
cd ~/synthetic_datasets/scripts
python3 generate_datasets.py --help
```

Example:

```bash
cd ~/synthetic_datasets/scripts
python3 generate_datasets.py --dataset ../data/sphere2500.g2o --output_dir ../output --sensing 50.0 --meas_prob 0.1
```