for seed in 1..30 do
    for equation in equations do
      sbatch bessemer-run.sh seed equation ...
    done
done
