import os
import pandas as pd

# Directory containing the files
directory = "/imec/other/csainfra/kundu16/DeepFlow/results/output/LLM/"

t_elapsed =0.0
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

print("t_elapsed:", t_elapsed)
