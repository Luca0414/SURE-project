#!/bin/bash

run_gp() {
    seed=$1
    eq=$2
    op=$3
    dp=$4
    noise=$5
    python deap_GP_first.py --seed $seed --equation "$eq" --operators "$op" --data_points $dp --noise $noise --method "GP"
}

run_lr() {
    seed=$1
    eq=$2
    op=$3
    dp=$4
    noise=$5
    python deap_GP_first.py --seed $seed --equation "$eq" --operators "$op" --data_points $dp --noise $noise --method "LR"
}

run_gplr() {
    seed=$1
    eq=$2
    op=$3
    dp=$4
    noise=$5
    python deap_GP_first.py --seed $seed --equation "$eq" --operators "$op" --data_points $dp --noise $noise --method "GPLR"
}

equations=(
    "x**2 + x + 1"
    "x**3 - x + 2"
    "x**2 - 4*x + 4"
    "2*x**2 + 3*x + 5"
    "x**4 + x**3 + x**2 + x + 1"
    "x**2 * math.sin(x)"
    "math.cos(x) + x**3"
    "math.exp(x) - x**2"
    "math.log(x) + x"
    "x**3 - 3*x**2 + 3*x - 1"
)

# Number of seeds
seeds=30

# RQ2 configurations
operators1="operator.add operator.sub operator.mul operator.truediv"
operators2="operator.add operator.sub operator.mul operator.truediv operator.neg math.cos math.sin"
operators3="operator.add operator.sub operator.mul operator.truediv operator.neg math.cos math.sin math.log math.sqrt square cube fourth_power"

# RQ3 configurations
data_points=(10 100 1000)

# RQ4 configurations
noise_levels=(0 0.1 0.2 0.3)


for eq in "${equations[@]}"; do
    for seed in $(seq 1 $seeds); do #use arrayjob
        for op in "$operators1" "$operators2" "$operators3"; do
            for dp in "${data_points[@]}"; do
                for noise in "${noise_levels[@]}"; do
                    run_gp $seed "$eq" "$op" $dp $noise
                    run_lr $seed "$eq" "$op" $dp $noise
                    run_gplr $seed "$eq" "$op" $dp $noise
                done
            done
        done
    done
done