#!/bin/bash

run_gp() {
    seed=$1
    eq=$2
    op=$3
    dp=$4
    noise=$5
    nv=$6
    srun run_bessemer.sh --seed $seed --equation "$eq" --operator_level $op --data_points $dp --noise $noise --method "GP" --num_vars $nv
}

run_lr() {
    seed=$1
    eq=$2
    op=$3
    dp=$4
    noise=$5
    nv=$6
    srun run_bessemer.sh --seed $seed --equation "$eq" --operator_level $op --data_points $dp --noise $noise --method "LR" --num_vars $nv
}

run_gplr() {
    seed=$1
    eq=$2
    op=$3
    dp=$4
    noise=$5
    nv=$6
    srun run_bessemer.sh --seed $seed --equation "$eq" --operator_level $op --data_points $dp --noise $noise --method "GPLR" --num_vars $nv
}

equations=()
num_vars=()
line_num=0

while IFS= read -r line; do
    ((line_num++))
    if (( line_num % 2 == 1)); then
        equations+=("$line")
    else
        num_vars+=("$line")
    fi
done < small_equations.txt

seeds=30

# RQ2 configurations
operator_level=(0 1 2 3)

# RQ3 configurations
data_points=(10 100)

# RQ4 configurations
noise_levels=(0 0.1 0.25)

counter=0


for eq in "${equations[@]}"; do
    for seed in $(seq 1 $seeds); do
        for op in "${operator_level[@]}"; do
            for dp in "${data_points[@]}"; do
                for noise in "${noise_levels[@]}"; do
                    run_gp $seed "$eq" $op $dp $noise ${num_vars[counter]}
                    run_lr $seed "$eq" $op $dp $noise ${num_vars[counter]}
                    run_gplr $seed "$eq" $op $dp $noise ${num_vars[counter]}
                done
            done
        done
    done
    ((counter++))
done
