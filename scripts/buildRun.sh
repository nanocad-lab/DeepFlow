#!/bin/bash

# creating GEMMs from LLM
python mat_dims_amped.py 128 1536 2048 96 16 1536 6144
#mat_dims_amped.py [-h] B D S h nheads h_MLP1 h_MLP2

# run DeepFlow; change parallelization in run.sh if needed
bash run.sh

# calculate total time
python cal_time.py 24 128 2048 300000000000 130000 1
#usage: cal_time.py [-h] N_L B S ntokens comm_time N_PP
