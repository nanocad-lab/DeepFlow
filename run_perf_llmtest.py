#!/tools/lm-venv/py3.6-tf-1.3.0-svail/bin/python
import argparse
import math
import os
import sys
import config
import os
from astrasim_integration import ensure_cache_unlocked_if_standalone
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
    # Emit total time for astra_test parsing
    print("Total time: {}".format(tot_time))

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

    # Forward timing
    forward_time = None
    forward_red = 0.0
    if TC.kp1 == 1 and TC.kp2 == 1:  # no parallelism
        forward_time = TC.getCf(TC.M, TC.K, TC.N)
    elif TC.t == "CR":
        gemm_time, forward_red = TC.getDistGEMM_f_kp1(TC.M, TC.K, TC.N, TC.kp1, "Cf_CR")
        forward_time = gemm_time + forward_red
    elif TC.t == "RC":
        gemm_time, forward_red = TC.getDistGEMM_f_kp2(TC.M, TC.K, TC.N, TC.kp1, TC.kp2, "Cf_RC")
        forward_time = gemm_time + forward_red
    else:
        print("Incorrect parallelism type, CR: Column-Row, RC: Row-Column")
        sys.exit(1)

    # Optional backward timing + dp reduction
    backward_time = 0.0
    dp_reduction_time = 0.0
    backward_red = 0.0
    if getattr(TC.model, "backward", False):
        if TC.kp1 == 1 and TC.kp2 == 1:
            grad_act_time, _, _, _ = TC.getGEMMTime(TC.M, TC.N, TC.K, "Cb_act")
            grad_wt_time, _, _, _ = TC.getGEMMTime(TC.K, TC.M, TC.N, "Cb_wt")
            backward_time = grad_act_time + grad_wt_time
        elif TC.t == "CR":
            gemm_time, bg_red = TC.getDistGEMM_b_kp1(TC.M, TC.K, TC.N, TC.kp1, "Cb_CR")
            backward_time = gemm_time + bg_red
            backward_red = bg_red
        elif TC.t == "RC":
            gemm_time, bg_red = TC.getDistGEMM_b_kp2(TC.M, TC.K, TC.N, TC.kp1, TC.kp2, "Cb_RC")
            backward_time = gemm_time + bg_red
            backward_red = bg_red
        # Data-parallel reduction after backward, if applicable
        if TC.dp and TC.dp > 1:
            dp_reduction_time = TC.getDataParallelReduction(
                k=TC.K,
                n=TC.N,
                dim1=TC.kp1,
                dim2=TC.kp2,
                name="GEMM Reduction",
            )
            backward_red += dp_reduction_time

    total_time = forward_time + backward_time + dp_reduction_time

    output_file = exp_dir + "/summary_mode%s_M%s_K%s_N%s.txt" % (mode, TC.M, TC.K, TC.N)
    with open(output_file, "w") as f:
        # Forward/Backward breakdown (no tiling)
        f.write("Forward Compute Time: {}\n".format(forward_time - forward_red))
        f.write("Forward Reduction Time: {}\n".format(forward_red))
        if getattr(TC.model, "backward", False):
            f.write("Backward Compute Time: {}\n".format(backward_time - backward_red))
            f.write("Backward Reduction Time: {}\n".format(backward_red))
            if dp_reduction_time > 0:
                f.write("DP Reduction Time: {}\n".format(dp_reduction_time))
            f.write("Total Time: {}\n".format(total_time))
    print("Performance Results written to {}".format(output_file))
    # Emit lines for astra_test parsing
    print("Total time: {}".format(total_time))
    print("Reduction time: {}".format(forward_red + backward_red))
    print("Reduction FWD time: {}".format(forward_red))
    print("Reduction BWD time: {}".format(backward_red))
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
    

    time = TC.calcTime_LLM()
    # TC.printSysConfig(exp_hw_config, exp_model_config, output_file)
    
    
    
    with open(output_file, "a+") as f:
        f.write("\n\n==============================================\n")
        f.write("Performance Results\n")
        f.write("==============================================\n")
        # f.write("Forward Time: {0:.8f}\n".format(time_fw))
        f.write("Total Time: {0:.8f}\n".format(time))
        # f.write("Params (Billion): {0:.8f}\n".format(tot_param / 1e9))
    # print("Performance Results written to {}".format(output_file))
            
    # output_dir_LLM = os.path.join(exp_dir, "output_LLM")
    # caltime(TC.num_layers ,TC.batch_size, TC.seq_len,TC.n_tokens, TC.communication_time, TC.N_PP, exp_dir, exp_dir)
    
    # Emit lines for astra_test parsing
    print("Total time: {}".format(TC.getTime()))
    print("Reduction time: {}".format(TC.getReductionTotal()))
    return


if __name__ == "__main__":
    # Best-effort to clear any stale cache lock when running standalone
    # ensure_cache_unlocked_if_standalone()
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
