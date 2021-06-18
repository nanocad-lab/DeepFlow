## Installation Guide ##
==========================================
Pre-requirement: Python3

**Step 1**. git clone https://github.com/baidu-research/DeepFlow.git 

**Step 2**. cd DeepFlow

**Step 3**. Setup the environment:
	python3 -m venv /path/to/new/virtual/environment
	source /path/to/new/virtual/environment/bin/activate
	pip install --upgrade pip
	pip install click ruamel.yaml numpy
	mkdir /path/to/output/result/directory 

**Step 4**. Test if the installation has been successful:
	python perf.py --exp_config configs/v100.yaml --exp_dir debug
	check the output result: vim /path/to/output/result/directory/summary.txt




## Execution Modes ##
===========================================

The tool can be used in 5 different modes:

(1) Peformance Prediction Mode -- (gemm kernel) 
When: Mostly purposed for GEMM validation
How: python perf.py --exp_config configs/v100.yaml --exp_dir debug --debug True --gemm True --t RC --kp1 4 --kp2 16 --m 16384 --n 8192 --k 4096 

(2) Standalone mode -- end-2-end LM
python perf.py --exp_config configs/v100.yaml --exp_dir debug --batch_size 4096 --seq_len 20 --hidden_dim 19968 --vocab_size 800000 --dp 8 --kp1 8 --kp2 1 --t RC

(3) Standalone-mode -- using main.py standalone argument; this is somewhat equivalent of option 2
python main.py stand_alone --exp_dir /mnt/scratch/newsha/MechaFlow --exp_config configs/exp_config_SiIF.yaml --no_launch True

(4) Architecture search for a fixed parallelism strategy
python GD_search.py --exp_config configs/exp_config.yaml --exp_dir debug --debug False --index 40 --batch_size 256 --hidden_dim 19968 --data_scale 1 --dp 64 --lp 1 --kp_type 1 --kp1 1 --kp2 1 --inter_derate 0 --intra_derate 2 --kp1_inter False --kp2_inter False --dp_inter False --lp_inter False --wafer_dim 8

(5) Architecture Search mode for all types of parallelism strategies
python main.py arch_search --exp_dir /mnt/scratch/newsha/MechaFlow --exp_config configs/exp_config_SiIF.yaml --no_launch True
