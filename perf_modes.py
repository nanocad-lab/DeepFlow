#!/tools/lm-venv/py3.6-tf-1.3.0-svail/bin/python

import os
import sys
import config
import pandas as pd

from tile import TiledGEMM, formatBytes
from time_calculation import TimeCalculation
from transformer_util import  process_gemm_shapes, caltime

algByte = False  # algorithmic ops false
proj = False  # consider projection layer, turn off for end-2-end validation, as baeline model does not have projection layer
validating_v100 = True



    
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
    
    
    # exp_path = os.path.expandvars(os.path.expanduser(exp_config))
    # exp_config = config.parse_config(exp_path)
    # output_file = exp_dir + "/summary_GEMM_m%s_n%s_k%s.txt" % (
    #     m,
    #     n,
    #     k,
    # ) 
    # create output dir if it doesn't exist
    # if not os.path.exists(exp_dir):
    #     os.makedirs(exp_dir)

    TC = TimeCalculation(exp_hw_config, exp_model_config, mode)
    
    TC.validating_GEMM = True
    # Report GEMM time on fw path

    if TC.kp1 == 1 and TC.kp2 == 1:  # no parallelism
        gemm_time = TC.getCf(TC.M, TC.K, TC.N)
    elif TC.t == "CR":
        gemm_time = TC.getDistGEMM_f_kp1(TC.M, TC.K, TC.N, TC.kp1, "Cf_CR")
    elif TC.t == "RC":
        gemm_time = TC.getDistGEMM_f_kp2(TC.M, TC.K, TC.N, TC.kp1, TC.kp2, "Cf_RC")
    else:
        print("Incorrect parallelism type, CR: Column-Row, RC: Row-Column")
        sys.exit()
        
    output_file = exp_dir + "/summary_mode%s_M%s_K%s_N%s.txt" % (
        mode, TC.M, TC.K, TC.N
    ) 
    with open(output_file, "w") as f:
        f.write("Best Order: {}\n".format(gemm_time[1]))
        f.write("Best Tile: {}\n".format(gemm_time[2]))
        f.write("Time: {}\n".format(gemm_time[0]))
        for i in range(len(gemm_time[3])):
            f.write(f"L{i}: {formatBytes(gemm_time[3][i])}\n")
    print("Performance Results written to {}".format(output_file))
    return


def run_Transformer(
    exp_hw_config_path,
    exp_model_config_path,
    exp_dir,
    mode):
    exp_hw_path = os.path.expandvars(os.path.expanduser(exp_hw_config_path))
    exp_model_path = os.path.expandvars(os.path.expanduser(exp_model_config_path))
    exp_hw_config = config.parse_config(exp_hw_path, config_type="hardware")
    exp_model_config = config.parse_config(exp_model_path, config_type=mode)
    TC = TimeCalculation(exp_hw_config, exp_model_config, mode)
    gemm_shapes_4d,gemm_3d=process_gemm_shapes(TC.batch_size, TC.seq_len, TC.hidden_dim, TC.num_heads, TC.h_MLP1, output_file="mat_dims_llm.txt", option="multiply_batch_into_m")
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
            
            
    caltime(TC.num_layers ,TC.batch_size, TC.seq_len,TC.n_tokens, TC.communication_time, TC.N_PP, exp_dir)
    return


if __name__ == "__main__":

    config_hardware_path = "configs/hardware-config/waferscale_20v100_80hbm.yaml"
    config_Transformer_model_path = "configs/model-config/Transformer.yaml"
    config_LSTM_model_path = "configs/model-config/LSTM.yaml"
    config_GEMM_path = "configs/model-config/GEMM.yaml"

    # mode= "Transformer"  
    # mode= "GEMM"  
    mode= "LSTM"  #ONLY MODIFY THIS LINE TO CHANGE THE MODEL TYPE
    
    script_dir = os.path.dirname(os.path.abspath(__file__))  
    output_dir = os.path.join(script_dir, "output")
    exp_dir = os.path.join(output_dir, mode)
    os.makedirs(exp_dir, exist_ok=True)
    
    
    if mode == "Transformer":
        print("Using Transformer parameters for computation...")
        run_Transformer(
            exp_hw_config_path=config_hardware_path,
            exp_model_config_path=config_Transformer_model_path,
            exp_dir=exp_dir,
            mode=mode,
        )
    
    elif mode == "LSTM":
        print("Using LSTM parameters for computation...")
        run_LSTM(
            exp_hw_config_path=config_hardware_path,
            exp_model_config_path=config_LSTM_model_path,
            exp_dir=exp_dir,
            mode=mode,
        
        )
    elif mode == "GEMM":
        print("Using GEMM parameters for computation...")
        run_GEMM(
            exp_hw_config_path=config_hardware_path,
            exp_model_config_path=config_GEMM_path,
            exp_dir=exp_dir,
            mode=mode,
        )
    else:
        print("Invalid mode selected. Please choose 'Transformer', 'LSTM', or 'GEMM'.")

