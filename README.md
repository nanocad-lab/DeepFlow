## Installation Guide ##

Pre-requirement: Python3

**Step 1**. git clone https://github.com/nanocad-lab/DeepFlow.git

**Step 2**. cd DeepFlow

**Step 3**. Setup the environment:

	* python3 -m venv [/path/to/new/virtual/environment]
	* source [/path/to/new/virtual/environment]/bin/activate
	* pip install --upgrade pip
	* pip install -r requirements.txt

**Step 4**. Test if the installation has been successful:

	* python run_perf.py --hardware_config configs/hardware-config/waferscale_20v100_80hbm.yaml --model_config configs/model-config/LSTM.yaml --output_dir output
	* check the output result: vim output/LSTM/summary_LSTM.txt


## Execution Modes ##

DeepFlow can be used in 6 different modes:

(1) Peformance Prediction Mode (GEMM) 
 **When to use**: Use for distributed GEMM prediction  
 **How**:   
* Specify the GEMM parameters in configs/model-config/GEMM.yaml
* Specify the Hardware parameters in configs/hardware-config/[config.yaml]
* python run_perf.py --hardware_config configs/hardware-config/[config.yaml] --model_config configs/model-config/GEMM.yaml --output_dir [/path/to/output/directory]

(2) Performance Prediction Mode (LSTM End-2-End Application)
 **When to use**: Use for End-2-End LSTM prediction  
 **How**:   
* Specify the LSTM parameters in configs/model-config/LSTM.yaml
* Specify the Hardware parameters in configs/hardware-config/[config.yaml]
* python run_perf.py --hardware_config configs/hardware-config/[config.yaml] --model_config configs/model-config/LSTM.yaml --output_dir [/path/to/output/directory]
        

(3) Performance Prediction Mode (LLM mode)
 **When to use**: Use for End-2-End LLM prediction  
 **How**:   
* Specify the LLM parameters in configs/model-config/LLM.yaml
* Specify the Hardware parameters in configs/hardware-config/[config.yaml]
* python run_perf.py --hardware_config configs/hardware-config/[config.yaml] --model_config configs/model-config/LLM.yaml --output_dir [/path/to/output/directory]

(4) Performance Prediction Mode (using main.py standalone argument; this is somewhat equivalent of option 2, for running on slurm)
* python main.py stand_alone --exp_dir [/path/to/output/result/directory] --exp_config configs/[config.yaml]

(5) Architecture search for a fixed parallelism strategy
* python GD_search.py --exp_config configs/[config.yaml] --exp_dir [/path/to/output/directory] --debug False --index [index] --batch_size [batch] --hidden_dim [lstm_dim] --data_scale [dataset_scaling_factor] --dp [data parallel dim.] --lp [layer parallel dim.] --kp_type [0|1] --kp1 [kp1 dim.] --kp2 [kp2 dim.] --inter_derate [derate_factor_for_inter_package_bandwidth] --intra_derate [derate_factor_for_intra_package_bandwidth] --kp1_inter [False|True] --kp2_inter [False|True] --dp_inter [False|True] --lp_inter [False|True] --wafer_dim [package dim.]
* **Example**: python GD_search.py --exp_config configs/exp_config.yaml --exp_dir output --debug False --index 40 --batch_size 256 --hidden_dim 19968 --data_scale 1 --dp 64 --lp 1 --kp_type 1 --kp1 1 --kp2 1 --inter_derate 0 --intra_derate 2 --kp1_inter False --kp2_inter False --dp_inter False --lp_inter False --wafer_dim 8

(6) Architecture Search mode for all types of parallelism strategies
* python main.py arch_search --exp_dir [/path/to/output/directory] --exp_config configs/[config.yaml]


## Tips ##

* Use --no_launch True to see the command that would be used to launch the application w/o running
* Check config directory for  different architecture templates and technology node configurations
* Use --debug True to activate debugging mode
 
