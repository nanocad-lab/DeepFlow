#!/bin/bash

# Input data file
input_file="mat_dims.txt"

# Read lines from the input file
while IFS= read -r line; do
    # Extract values from the line
    value1=$(echo "$line" | awk '{print $1}')
    value2=$(echo "$line" | awk '{print $2}')
    value3=$(echo "$line" | awk '{print $3}')

    # Submit a job using python perf.py with extracted values as arguments
    python perf.py --exp_config configs/new-configs/v100.yaml --exp_dir results/output/LLM/ --debug True --gemm True --t RC --kp1 1 --kp2 1 --m "$value1" --n "$value2" --k "$value3"

    # You can add sleep between job submissions if needed
    # sleep 1
done < "$input_file"
