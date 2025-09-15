#!/tools/lm-venv/py3.6-tf-1.3.0-svail/bin/python
import argparse
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
    TC.debug = False

    tot_time, tot_param = TC.calcTime()
    TC.printSysConfig(exp_hw_config, exp_model_config, output_file)
    # print(f'IBD: {TC.IBD}, LLD: {TC.LLD}')
    # print(f'num_workers: {TC.num_workers}, num_wafer: {TC.num_wafer}')

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

    # Forward timing
    forward_tuple = None
    forward_time = None
    forward_red = 0.0
    if TC.kp1 == 1 and TC.kp2 == 1:  # no parallelism
        forward_tuple = TC.getCf(TC.M, TC.K, TC.N)
        forward_time = forward_tuple[0]
    elif TC.t == "CR":
        forward_tuple = TC.getDistGEMM_f_kp1(TC.M, TC.K, TC.N, TC.kp1, "Cf_CR")
        forward_time = forward_tuple[0]
        # Reduction for CR forward: RS over kp1 rows
        forward_red = TC.getR(
            Dim0=TC.M,
            Dim1=TC.N,
            p=TC.kp1,
            ib=TC.IBK1,
            ll=TC.LLK1,
            partial=True,
            allReduce=True,
            name="Cf_CR",
        )
    elif TC.t == "RC":
        forward_tuple = TC.getDistGEMM_f_kp2(TC.M, TC.K, TC.N, TC.kp1, TC.kp2, "Cf_RC")
        forward_time = forward_tuple[0]
        # Reduction for RC forward: AG over kp2 columns
        forward_red = TC.getR(
            Dim0=TC.M // TC.kp1 if TC.kp1 else TC.M,
            Dim1=TC.N,
            p=TC.kp2,
            ib=TC.IBK2,
            ll=TC.LLK2,
            partial=False,
            allReduce=False,
            name="Cf_RC",
        )
    else:
        print("Incorrect parallelism type, CR: Column-Row, RC: Row-Column")
        sys.exit(1)

    # Optional backward timing
    backward_time = 0.0
    dp_reduction_time = 0.0
    backward_red = 0.0
    if getattr(TC.model, "backward", False):
        if TC.kp1 == 1 and TC.kp2 == 1:
            # Two GEMMs: grad wrt act and weights
            grad_act_time, _, _, _ = TC.getGEMMTime(TC.M, TC.N, TC.K, "Cb_act")
            grad_wt_time, _, _, _ = TC.getGEMMTime(TC.K, TC.M, TC.N, "Cb_wt")
            backward_time = grad_act_time + grad_wt_time
        elif TC.t == "CR":
            bg_gemm, bg_red = TC.getDistGEMM_b_kp1(TC.M, TC.K, TC.N, TC.kp1, "Cb_CR")
            backward_time = bg_gemm + bg_red
            backward_red = bg_red
        elif TC.t == "RC":
            bg_gemm, bg_red = TC.getDistGEMM_b_kp2(TC.M, TC.K, TC.N, TC.kp1, TC.kp2, "Cb_RC")
            backward_time = bg_gemm + bg_red
            backward_red = bg_red
        # Data-parallel reduction of weight gradients (follow LSTM policy)
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
    # Emit a single reduction line for astra_test parsing (sum of fwd+bwd reductions)
    print("Reduction_time: {}".format(forward_red + backward_red))
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
    TC = TimeCalculation(exp_hw_config, exp_model_config, mode)
    gemm_3d=process_gemm_shapes(TC.batch_size, TC.seq_len, TC.hidden_dim, TC.num_heads, TC.ffn_dim,  option="multiply_batch_into_m")
    print(gemm_3d) #m,k,n
    for i, (m, k, n) in enumerate(gemm_3d):
        print(f"Running main for GEMM dimensions: M={m}, K={k}, N={n} (Layer {i + 1})")
        
        output_file = exp_dir + "/summary_m%s_n%s_k%s_layer%s.txt" %(m, n, k, i+1) ##Output dir should be created manually

        TC = TimeCalculation(exp_hw_config, exp_model_config, mode)
        TC.validating_GEMM = True

        if TC.kp1 == 1 and TC.kp2 ==1: #no parallelism
            gemm_time = TC.getCf(m, k, n)
        elif TC.t == "CR":
            gemm_time = TC.getDistGEMM_f_kp1(m, k, n, TC.kp1, "Cf_CR")
        elif TC.t == "RC":
            gemm_time = TC.getDistGEMM_f_kp2(m, k, n, TC.kp1, TC.kp2, "Cf_RC")
        else:
            print("Incorrect parallelism type, CR: Column-Row, RC: Row-Column")
            sys.exit()

        with open(output_file, "w") as f:
            f.write("Best Order: {}\n".format(gemm_time[1]))
            f.write("Best Tile: {}\n".format(gemm_time[2]))
            f.write("Time: {}\n".format(gemm_time[0]))
            
    # output_dir_LLM = os.path.join(exp_dir, "output_LLM")
    caltime(TC.num_layers ,TC.batch_size, TC.seq_len,TC.n_tokens, TC.communication_time, TC.N_PP, exp_dir, exp_dir)
    
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
