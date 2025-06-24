import os
import pandas as pd
import argparse

# Directory containing the files
directory = "../output/LLM/"

t_elapsed =0.0
print("Time spent in different GEMMs")
# Loop through files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".txt"):  # You can adjust the file extension as needed
        file_path = os.path.join(directory, filename)

        data = pd.read_csv(file_path, sep=':', header=None, names=['Field', 'Value'], skipinitialspace=True)
        # Extract the time value
        time_row = data[data['Field'] == 'Time']
        if not time_row.empty:
            time_value = float(time_row['Value'].iloc[0])
            print(time_value)
        else:
            print("Time value not found in the file.")
        
        t_elapsed += time_value

t_elapsed = t_elapsed*3.0 #FW pass + BW pass (~ 2x FW pass)
comp_time = t_elapsed
#comm_time = 8.85 # (hours) comes from AMPED
parser = argparse.ArgumentParser(description='Generate Matrix dimensions')

parser.add_argument('N_L', type=int, help='nlayers')
parser.add_argument('B', type=int, help='batch size')
parser.add_argument('S', type=int, help='sequence length')
parser.add_argument('ntokens', type=int, help='number of tokens to train')
parser.add_argument('comm_time', type=int, help='comm overhead')
parser.add_argument('N_PP', type=int, help='pipeline degree')

args = parser.parse_args()

N_L = args.N_L
B = args.B
S = args.S
ntokens = args.ntokens
comm_time = args.comm_time
N_PP = args.N_PP

nbatch = ntokens/(S*B)
time = N_L*nbatch*t_elapsed/N_PP + comm_time

print("number of tokens:", ntokens, " | time to exhaust all tokens:", time, "(s)", " or ", time/3600.0/24.0, " days")
