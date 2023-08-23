#!/bin/bash

# creating GEMMs from LLM
python mat_dims_amped.py 128 2560 2048 80 32 2560 10240
#mat_dims_amped.py [-h] B D S h nheads h_MLP1 h_MLP2

# run DeepFlow; change parallelization in run.sh if needed
bash run.sh

# calculate total time
python cal_time.py 2048 300000000000

