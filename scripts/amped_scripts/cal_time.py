import os
import pandas as pd
import argparse

def time_from_GEMM():
    # Directory containing the files
    directory = "/imec/other/csainfra/kundu16/DeepFlow/results/output/LLM/"

    t_elapsed =0.0
    #print("Time spent in different GEMMs")
    # Loop through files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):  # You can adjust the file extension as needed
            file_path = os.path.join(directory, filename)

            data = pd.read_csv(file_path, sep=':', header=None, names=['Field', 'Value'], skipinitialspace=True)
            # Extract the time value
            time_row = data[data['Field'] == 'Time']
            if not time_row.empty:
                time_value = float(time_row['Value'].iloc[0])
                #print(time_value)
            else:
                print("Time value not found in the file.")

            t_elapsed += time_value

    t_elapsed = t_elapsed*2.0 #FW pass + BW pass (weight update time is added seperately)
    #print("********** FW-BW pass time ************", t_elapsed)
    return t_elapsed


def main():
    print("**** Computing GEMM timings using DeepFlow *****")
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', type=str, required=True)
    parser.add_argument('--breakdown_filename', type=str, required=True)

    args = parser.parse_args()
    config_filename = args.config_filename
    breakdown_filename = args.breakdown_filename

    #print("*******", config_filename, breakdown_filename, "*********")

    #------------------------reading configs from AMPeD ------------------------------------------------
    amped_dir = os.path.dirname('/imec/other/csainfra/kundu16/public/AMPeD/')
    #config_out_file_path = amped_dir +'/'+ 'output_files/'+ '2023-09-20_10-07-00_config_summary.txt'
    config_out_file_path = amped_dir +'/'+ 'output_files/'+ config_filename

    #N_L B S ntokens comm_time N_PP
    llm_configs = ["layers", "batch_size", "context", "tokens_to_train", \
                "data_parallel_degree", "tensor_parallel_degree", "pipeline_parallel_degree"]
    llm_params_ext = {}
    df = pd.read_csv(config_out_file_path)
    for i, conf in enumerate(llm_configs):
        llm_params_ext[conf]=[]
        for index, row in df.iterrows():
            if conf in row.str.split().values[0]:
                llm_params_ext[conf].append(row.str.split().values[0][2])
                #print(conf, row.str.split().values[0][2])

    #print(llm_params)
    #print(llm_params_ext["tokens_to_train"][0])

    #------------------------reading timings from AMPeD------------------------------------------------------------
    #breakdown_out_file_path = amped_dir +'/'+ 'output_files/'+ '2023-09-20_10-07-00_training_time_breakdown.txt'
    breakdown_out_file_path = amped_dir +'/'+ 'output_files/' + breakdown_filename
    components = ["Total communication time forward pass (s)", \
              "Total communication time backward pass (s)", \
             "Computation time weight updates (s)", "Waiting Time due to pipeline bubbles (s)"]
    time_spent={}
    df = pd.read_csv(breakdown_out_file_path)

    #print(components)
    timeFromAmped = {}
    for i, comp in enumerate(components):
        timeFromAmped[comp] = []
        for index, row in df.iterrows():
            #print(row.str.split('-----').values[0])
            if comp in row.str.split('-----').values[0]:
                timeFromAmped[comp].append(row.str.split().values[0][-1])
                #print(comp, ":", row.str.split().values[0][-1])

    print(timeFromAmped)    

    t_FW_BW = time_from_GEMM()
    #print("t_FW_BW:", t_FW_BW)
    nbatch = int(llm_params_ext["tokens_to_train"][0])/(int(llm_params_ext["context"][0])\
                                           *int(llm_params_ext["batch_size"][0]))
    time = int(llm_params_ext["layers"][0])*nbatch*t_FW_BW + float(timeFromAmped["Total communication time forward pass (s)"][0])\
        +float(timeFromAmped["Total communication time backward pass (s)"][0])\
        +float(timeFromAmped["Computation time weight updates (s)"][0]) \
        +float(timeFromAmped["Waiting Time due to pipeline bubbles (s)"][0])

    print("total time:", time)

if __name__ == '__main__':
    main()

