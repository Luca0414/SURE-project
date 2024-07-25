#!/bin/bash
#SBATCH --nodes=1
#sBatch -c=4
#SBATCH --mem=4000
#SBATCH --time=04:00:00
#SBATCH --mail-user=ldevlin1@sheffield.ac.uk

module load Anaconda3/2019.07
source activate myconda

echo "sbatch run_bessemer.sh "$@
python deap_GP_first.py --seed $1 --equation $2 --operator_level $3 --data_points $4 --noise $5 --method $6 --num_vars $7