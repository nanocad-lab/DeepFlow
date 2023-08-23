import os
import pandas as pd
import argparse

# Directory containing the files
directory = "/imec/other/csainfra/kundu16/DeepFlow/results/output/LLM/"

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

parser = argparse.ArgumentParser(description='Generate Matrix dimensions')

parser.add_argument('S', type=int, help='sequence length')
parser.add_argument('ntokens', type=int, help='number of tokens to train')

args = parser.parse_args()

S = args.S
ntokens = args.ntokens
nbatch = ntokens/S

print("number of tokens:", ntokens, " | time to exhaust all tokens:", nbatch*t_elapsed, "(s)", " or ", nbatch*t_elapsed/3600.0, " days")
