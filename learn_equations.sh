configurations=$1
line=$2
python learn_equations.py $(sed -n "${line} p" $configurations)
