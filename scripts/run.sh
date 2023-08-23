#!/bin/bash

# Input data file
input_file="mat_dims.txt"

# Relevant directories
RUNDIR="/imec/other/csainfra/kundu16/DeepFlow"
CONFIG_DIR="/imec/other/csainfra/kundu16/DeepFlow/configs/new-configs"
OUTDIR="/imec/other/csainfra/kundu16/DeepFlow/results/output"


# Read lines from the input file
while IFS= read -r line; do
    # Extract values from the line
    value1=$(echo "$line" | awk '{print $1}')
    value2=$(echo "$line" | awk '{print $2}')
    value3=$(echo "$line" | awk '{print $3}')

    # Submit a job using python perf.py with extracted values as arguments
    python "$RUNDIR"/perf.py --exp_config "$CONFIG_DIR"/v100.yaml --exp_dir "$OUTDIR"/LLM/ --debug True --gemm True --t RC --kp1 4 --kp2 1 --m "$value1" --n "$value2" --k "$value3"

    # You can add sleep between job submissions if needed
    # sleep 1
done < "$input_file"
