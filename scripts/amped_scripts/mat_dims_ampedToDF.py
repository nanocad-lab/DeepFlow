import numpy as np
import argparse
import os
import pandas as pd

def mmm_breakup(B, D, S, h, nheads, h_MLP1, h_MLP2, N_DP, N_PP):
    mmm =  {}
    dims = {}
    numlevels = 6
    levels = ["X.W=KQV", "Q.K=R", "R.V=Z", "Z.W=O", "O.WL1=O1", "O1.WL2=O2"]
    #print("matrix dimensions accounting for all heads & batched dimension")
    dims[str(levels[0])]=[int(3*B*S/N_DP/N_PP), D, h*nheads] #factor 3 due to K+Q+V
    dims[str(levels[1])]=[int(B*S/N_DP/N_PP), h, S*nheads]
    dims[str(levels[2])]=[int(B*S/N_DP/N_PP), S, h*nheads]
    dims[str(levels[3])]=[int(B*S/N_DP/N_PP), D, D]
    dims[str(levels[4])]=[int(B*S/N_DP/N_PP), D, h_MLP1]
    dims[str(levels[5])]=[int(B*S/N_DP/N_PP), h_MLP1, h_MLP2]

    #print("levels:",levels)
    #print("writting the matrix dimensions ...")
    file = open("mat_dims_amped.txt","w")
    #file.write('#'+str(levels)+'\n')
    for i in range(numlevels):
        mmm[str(levels[i])]=[]
        mmm[str(levels[i])].append(dims[str(levels[i])])
        tmp = str(mmm[str(levels[i])]).replace("[", "").replace("]","").replace(",", " ")
        print(tmp)
        file.write(tmp+'\n')

def main():
    print("**** Creating GEMMs from AMPeD ****")
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', type=str, required=True)
    args = parser.parse_args()
    config_filename = args.config_filename

    #print("****", config_filename)

    #set the PATH to the amped dir
    amped_dir = os.path.dirname('/imec/other/csainfra/kundu16/public/AMPeD/')
    #Run AMPeD and generate the output file first
    out_file_path = amped_dir +'/'+ 'output_files/'+ config_filename

    #out_file_path = amped_dir +'/'+ 'output_files/'+ '2023-09-20_16-35-36_config_summary.txt'

    #B D S h nheads h_MLP1 h_MLP2
    llm_configs = ["batch_size", "dimensionality", "context", "hidden_layer_dimension_for_attention_sublayers",\
                 "attention_heads", "hidden_layer_dimension_MLP_1", "hidden_layer_dimension_MLP_2", "data_parallel_degree",
"pipeline_parallel_degree"]
    llm_params = {}
    df = pd.read_csv(out_file_path)
    for i, conf in enumerate(llm_configs):
        llm_params[conf]=[]
        for index, row in df.iterrows():
            if conf in row.str.split().values[0]:
                llm_params[conf].append(row.str.split().values[0][2])
                #print(conf, row.str.split().values[0][2])

    #print(llm_params)
    #print(llm_params["batch_size"][0])

    B = int(llm_params[llm_configs[0]][0])
    D = int(llm_params[llm_configs[1]][0])
    S = int(llm_params[llm_configs[2]][0])
    h = int(llm_params[llm_configs[3]][0])
    nheads = int(llm_params[llm_configs[4]][0])
    h_MLP1 = int(llm_params[llm_configs[5]][0])
    h_MLP2 = int(llm_params[llm_configs[6]][0])
    N_DP = int(llm_params[llm_configs[7]][0])
    N_PP = int(llm_params[llm_configs[8]][0])
    mmm_breakup(B, D, S, h, nheads, h_MLP1, h_MLP2, N_DP, N_PP)

if __name__ == '__main__':
    main()
