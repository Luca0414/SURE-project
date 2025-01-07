#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=4000
#SBATCH --time=2:00:00

echo sbatch learn_equations.sh $@

configurations=$1
line=$2

module load Anaconda3/2019.07
source activate gplr

python learn_equations.py $(sed -n "${line} p" $configurations)

echo "__COMPLETED__"
