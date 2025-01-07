#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=24000
#SBATCH --time=8:00:00

echo sbatch learn_equations.sh $@

configurations=$1
line=$2

module load Anaconda3/5.3.0
source activate gplr

python learn_equations.py $(sed -n "${line} p" $configurations)

echo "__COMPLETED__"
