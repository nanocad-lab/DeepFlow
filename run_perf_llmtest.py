#!/tools/lm-venv/py3.6-tf-1.3.0-svail/bin/python
import argparse
import os
import sys
import config
import pandas as pd
import yaml
import shutil
from tile import TiledGEMM, formatBytes
from time_calculation import TimeCalculation
from time_calculation_LLM import TimeCalculationLLM
from LLM_util import  process_gemm_shapes, caltime

algByte = False  # algorithmic ops false
proj = False  # consider projection layer, turn off for end-2-end validation, as baeline model does not have projection layer
validating_v100 = True
def parse_arguments():
    parser = argparse.ArgumentParser(description="Run performance analysis for LSTM, GEMM, or LLM models.")
    parser.add_argument("--hardware_config", required=True, help="Path to the hardware configuration file.")
    parser.add_argument("--model_config", required=True, help="Path to the model configuration file.")
    parser.add_argument("--output_dir", required=False, help="Directory to save the output files.")
    return parser.parse_args()

def get_mode_from_config(model_config_path):
    """Read the mode from the model configuration file."""
    with open(model_config_path, "r") as f:
        config_data = yaml.safe_load(f)  # Parse the YAML file
    
    # Access 'mode' under 'model_param'
    model_param = config_data.get("model_param")
    if not model_param or "mode" not in model_param:
        raise ValueError("Error: 'mode' is not specified in the model configuration file under 'model_param'.")
    
    return model_param["mode"]

def run_LSTM(
    exp_hw_config_path,
    exp_model_config_path,
    exp_dir,
    mode
):
    
    # exp_path = os.path.expandvars(os.path.expanduser(exp_config))
    exp_hw_path = os.path.expandvars(os.path.expanduser(exp_hw_config_path))
    exp_model_path = os.path.expandvars(os.path.expanduser(exp_model_config_path))
    exp_hw_config = config.parse_config(exp_hw_path, config_type="hardware")
    exp_model_config = config.parse_config(exp_model_path, config_type=mode)
    output_file = exp_dir + "/summary_%s.txt" % (
        mode,
    ) 


    TC = TimeCalculation(exp_hw_config, exp_model_config, mode)
    

    tot_time, tot_param = TC.calcTime()
    TC.printSysConfig(exp_hw_config, exp_model_config, output_file)

    with open(output_file, "a+") as f:
        f.write("\n\n==============================================\n")
        f.write("Performance Results\n")
        f.write("==============================================\n")
        f.write("Time: {0:.8f}\n".format(tot_time))
        f.write("Params (Billion): {0:.8f}\n".format(tot_param / 1e9))
    print("Performance Results written to {}".format(output_file))

def run_GEMM(
    exp_hw_config_path,
    exp_model_config_path,
    exp_dir,
    mode
):
    exp_hw_path = os.path.expandvars(os.path.expanduser(exp_hw_config_path))
    exp_model_path = os.path.expandvars(os.path.expanduser(exp_model_config_path))
    exp_hw_config = config.parse_config(exp_hw_path, config_type="hardware")
    exp_model_config = config.parse_config(exp_model_path, config_type=mode)


    TC = TimeCalculation(exp_hw_config, exp_model_config, mode)

    TC.validating_GEMM = True

    if TC.kp1 == 1 and TC.kp2 == 1:  # no parallelism
        gemm_time = TC.getCf(TC.M, TC.K, TC.N)
    elif TC.t == "CR":
        gemm_time = TC.getDistGEMM_f_kp1(TC.M, TC.K, TC.N, TC.kp1, "Cf_CR")
    elif TC.t == "RC":
        gemm_time = TC.getDistGEMM_f_kp2(TC.M, TC.K, TC.N, TC.kp1, TC.kp2, "Cf_RC")
    else:
        print("Incorrect parallelism type, CR: Column-Row, RC: Row-Column")
        sys.exit()

    output_file = exp_dir + "/summary_mode%s_M%s_K%s_N%s.txt" % (mode, TC.M, TC.K, TC.N)
    with open(output_file, "w") as f:
        f.write("Best Order: {}\n".format(gemm_time[1]))
        f.write("Best Tile: {}\n".format(gemm_time[2]))
        f.write("Time: {}\n".format(gemm_time[0]))
        for i in range(len(gemm_time[3])):
            f.write(f"L{i}: {formatBytes(gemm_time[3][i])}\n")
    print("Performance Results written to {}".format(output_file))
    return


def run_LLM(
    exp_hw_config_path,
    exp_model_config_path,
    exp_dir,
    mode):
    
    exp_hw_path = os.path.expandvars(os.path.expanduser(exp_hw_config_path))
    exp_model_path = os.path.expandvars(os.path.expanduser(exp_model_config_path))
    exp_hw_config = config.parse_config(exp_hw_path, config_type="hardware")
    exp_model_config = config.parse_config(exp_model_path, config_type=mode)
    output_file = exp_dir + "/summary_LLM.txt"
    
    
    

    TC = TimeCalculationLLM(exp_hw_config, exp_model_config, mode)
    

    time_fw, time_bw = TC.calcTime_LLM()
    # TC.printSysConfig(exp_hw_config, exp_model_config, output_file)
    
    
    
    with open(output_file, "a+") as f:
        f.write("\n\n==============================================\n")
        f.write("Performance Results\n")
        f.write("==============================================\n")
        f.write("Forward Time: {0:.8f}\n".format(time_fw))
        f.write("Backward Time: {0:.8f}\n".format(time_bw))
        
        f.write("Total Time: {0:.8f}\n".format(TC.getTime()))
        # f.write("Params (Billion): {0:.8f}\n".format(tot_param / 1e9))
    # print("Performance Results written to {}".format(output_file))
            
    # output_dir_LLM = os.path.join(exp_dir, "output_LLM")
    # caltime(TC.num_layers ,TC.batch_size, TC.seq_len,TC.n_tokens, TC.communication_time, TC.N_PP, exp_dir, exp_dir)
    
    return


if __name__ == "__main__":
    args = parse_arguments()
    # Load configurations
    config_hardware_path = args.hardware_config
    config_model_path = args.model_config
    output_dir = args.output_dir if args.output_dir else "output"


    # Read mode from the model configuration file
    mode = get_mode_from_config(config_model_path)
    exp_dir = os.path.join(output_dir, mode)
    # Check if the directory exists and delete it if it does
    if os.path.exists(exp_dir):
        shutil.rmtree(exp_dir)
    os.makedirs(exp_dir, exist_ok=True)
    
    
    if mode == "LLM":
        print("Using LLM parameters for computation...")
        run_LLM(
            exp_hw_config_path=config_hardware_path,
            exp_model_config_path=config_model_path,
            exp_dir=exp_dir,
            mode=mode,
        )
    
    elif mode == "LSTM":
        print("Using LSTM parameters for computation...")
        run_LSTM(
            exp_hw_config_path=config_hardware_path,
            exp_model_config_path=config_model_path,
            exp_dir=exp_dir,
            mode=mode,
        
        )
    elif mode == "GEMM":
        print("Using GEMM parameters for computation...")
        run_GEMM(
            exp_hw_config_path=config_hardware_path,
            exp_model_config_path=config_model_path,
            exp_dir=exp_dir,
            mode=mode,
        )
    else:
        print("Invalid mode selected. Please choose 'LLM', 'LSTM', or 'GEMM'.")
