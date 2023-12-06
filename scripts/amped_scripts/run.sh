#!/bin/bash

TP_intra=$(grep "intra_node_tensor_parallel_degree" $1 | awk -F ':' '{gsub(/[^0-9]/, "", $2); print $2}' | tr -d '[:space:]')
echo "TP intra degree: $TP_intra"

# Input data file
input_file="/imec/other/csainfra/kundu16/DeepFlow/scripts/amped_scripts/mat_dims_amped.txt"

# Relevant directories
RUNDIR="/imec/other/csainfra/kundu16/DeepFlow"
CONFIG_DIR="/imec/other/csainfra/kundu16/DeepFlow/configs/new-configs"
OUTDIR="/imec/other/csainfra/kundu16/DeepFlow/results/output"

# Remove the old files
rm -rf "$OUTDIR"/LLM/*

# Read lines from the input file
l=0
while IFS= read -r line; do
    # Extract values from the line
    value1=$(echo "$line" | awk '{print $1}')
    value2=$(echo "$line" | awk '{print $2}')
    value3=$(echo "$line" | awk '{print $3}')

    # Submit a job using python perf.py with extracted values as arguments
    python "$RUNDIR"/perf.py --args_input true --exp_config "$CONFIG_DIR"/v100_new.yaml --exp_dir "$OUTDIR"/LLM/ --debug false --gemm true --t RC --kp1 1 --kp2 "$TP_intra" --m "$value1" --k "$value2" --n "$value3" --lev "$l" --lp 2
    ((l++))
done < "$input_file"
