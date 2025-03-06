#!/bin/bash

# Source: https://dugas.ch/artificial_curiosity/GPT_architecture.html
B=3200000
D=12288
S=2048
h=128
nheads=96
h_MLP1=49152
h_MLP2=12288
# Source: https://lambdalabs.com/blog/demystifying-gpt-3#:~:text=GPT%2D3%20175B%20is%20trained%20with%20300%20Billion%20tokens
n_tokens=300000000000

# creating GEMMs from LLM
python mat_dims_amped.py $B $D $S $h $nheads $h_MLP1 $h_MLP2
#python mat_dims_amped.py 4096 2560 2048 80 32 2560 10240
#mat_dims_amped.py [-h] B D S h nheads h_MLP1 h_MLP2

# run DeepFlow; change parallelization in run.sh if needed
bash run.sh

# calculate total time
python cal_time.py $nheads $B $S $n_tokens 180013 16
#python cal_time.py 32 4096 2048 300000000000 180013 16 
#usage: cal_time.py [-h] N_L B S ntokens comm_time N_TP
