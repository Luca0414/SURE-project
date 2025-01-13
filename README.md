# Evolving Regression Models for Causal Testing

This repository contains the replication package for our paper entitled "[Evolving Regression Models for Causal Testing](#)".

## Pre-Requisites
We use [anaconda](https://www.anaconda.com) to manage dependencies and virtual environments.
While this is not essential, we strongly recommend you use some kind of virtual environment.
Our Python version and dependencies can be found in `environment.yaml`.

## Setup
1. Clone the repository and `cd` into it. All subsequent commands are to be run from within this directory.
2. Create a conda environment: `conda env create -f environment.yaml`. This will create a conda environment called `gplr` (Genetic Programming Linear Regression), although you can name the environment differently if you wish.
3. Activate the conda environment: `conda activate gplr`.

## Running controlled experiments for random expressions
1. Generate the configurations: `python generate_configurations.py`. This will create a file called `configurations.txt` which contains the 300 configurations we used for our paper.
2. Run `learn_equations.py` with each of the configurations in `configurations.txt`. This will create a directory called `results` containing JSON that record the output of each run.

The easiest way to do this to run `seq 1 300 | xargs -n 1 bash learn_equations.sh configurations.txt`.
This iteratively runs `learn_equations.py` with the configuration on each line of `configurations.txt`.
You can do this in parallel by adding `-P [number of processes]` to the `xargs` command.
You can also run this on an HPC cluster running slurm with `seq 1 300 | xargs -n 1 sbatch learn_equations.sh configurations.txt`.
This will submit 300 separate jobs, one for each configuration.

3. Generate the figures in the paper by running `python plotting/process_results.py`. This will create a directory called `figures` in which it will place the figures and statistical analyses.

## Running experiments for the Causal Testing Framework example equations
1. Run `python learn_ctf_examples.py -o ctf_example_results -s $seed` for 30 random seeds. This will create a directory called `ctf_example_results`. This will create a directory called `ctf_example_results` containing JSON files that record the output of each run.
For our paper, we used seeds 1-30. The easiest way to recreate this is by doing `seq 1 300 | xargs -n 1 python learn_ctf_examples.py -o ctf_example_results -s`.
This can also be run on HPC by modifying `learn_equations.bash` to call `learn_ctf_examples.py`.
3. Generate the figures in the paper by running `python plotting/process_ctf_results.py`. This will place the figures and statistical analyses within the `figures` directory (first creating the directory if it does not already exist).
