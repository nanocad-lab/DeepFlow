#!/bin/bash
#-----------------------------------------------------------------------------------
# Running AMPeD to spit out the configs
# Go to AMPeD repo
#-----------------------------------------------------------------------------------
dir="/imec/other/csainfra/kundu16/public/AMPeD"
rm $dir/output_files/*.txt #removing the old txt files

cd $dir
conda activate amped #look at https://github.com/CSA-infra/AMPeD to create your own conda env("amped")
#git checkout connect_DF #checkout to "connect_DF" branch
python -m amped.main
# This generates to files: 2023-09-20_16-35-36_config_summary.txt, 2023-09-20_16-35-36_training_time_breakdown.txt

#-----------------------------------------------------------------------------------
# parsing amped outputs to create the taskgraph and GEMM dimensions
#-----------------------------------------------------------------------------------
cd $dir/output_files
#saving the output filenames
array=(a,b)
array_counter=0
for f in "$dir"/output_files/*.txt; do
  #echo "$f"
  full_name=$f
  base_name=$(basename ${full_name})
  array[$array_counter]="$base_name"
  array_counter=$(($array_counter + 1))
done
echo ${array[0]}
# creating GEMMs from LLM by parsing AMPeD outputs
# Go to DeepFlow
cd /imec/other/csainfra/kundu16/DeepFlow/scripts/amped_scripts
python mat_dims_ampedToDF.py --config_filename ${array[0]}
conda deactivate

#--------------------------------------------------------------------------------------
# run DeepFlow; change parallelization in run.sh if needed
#--------------------------------------------------------------------------------------
source /imec/other/csainfra/kundu16/DF/bin/activate #look at https://github.com/nanocad-lab/DeepFlow for the virtual env "DF"
source run.sh $dir/output_files/${array[0]} #passing the config file obtained from AMPeD to extract the TP_intra degree

#--------------------------------------------------------------------------------------
# stitches the elapsed times to calculate total time
#--------------------------------------------------------------------------------------
python cal_time.py --config_filename ${array[0]} --breakdown_filename ${array[1]} 
deactivate
