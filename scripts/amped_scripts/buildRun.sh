#!/bin/bash

# Running AMPeD to spit out the configs
# Go to AMPeD
cd /imec/other/csainfra/kundu16/public/AMPeD
conda activate amped #look at https://github.com/CSA-infra/AMPeD to create your own conda env("amped")
python -m amped.main
# This generates to files: 2023-09-20_16-35-36_config_summary.txt, 2023-09-20_16-35-36_training_time_breakdown.txt

# creating GEMMs from LLM by parsing AMPeD outputs
# Go to DeepFlow
cd /imec/other/csainfra/kundu16/DeepFlow/scripts/amped_scripts
python mat_dims_ampedToDF.py --config_filename 2023-09-20_16-35-36_config_summary.txt
conda deactivate

# run DeepFlow; change parallelization in run.sh if needed
source /imec/other/csainfra/kundu16/DF/bin/activate #look at https://github.com/nanocad-lab/DeepFlow for the virtual env "DF"
bash run.sh

# calculate total time
python cal_time.py --config_filename 2023-09-20_16-35-36_config_summary.txt --breakdown_filename 2023-09-20_16-35-36_training_time_breakdown.txt
deactivate
