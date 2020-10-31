import subprocess
import time
import re
from datetime import datetime
import os
import shutil
import ruamel as _ruamel
import ruamel.yaml as _yaml
from main import call_arch_search


#base_config: path to the base_config
def create_config(base_config, out_dir, vol_scaling, area_scaling, power_scaling):
  config_dict = {}
  #read the base config
  with open(base_config, "r") as f:
    config_dict = _yaml.load(f, Loader=_ruamel.yaml.Loader)
    nominal_power_per_mcu = config_dict['tech_param']['core']['nominal_power_per_mcu']
    nominal_area_per_mcu = config_dict['tech_param']['core']['nominal_area_per_mcu']
    operating_area_per_mcu = config_dict['tech_param']['core']['operating_area_per_mcu']
    nominal_voltage = config_dict['tech_param']['core']['nominal_voltage']
  
    
    if os.path.exists(out_dir):
      shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)
    
    modified_config = "{}/exp_config.yaml".format(out_dir)
    config_dict['tech_param']['core']['nominal_power_per_mcu'] = nominal_power_per_mcu * power_scaling
    config_dict['tech_param']['core']['nominal_area_per_mcu'] = nominal_area_per_mcu * area_scaling
    config_dict['tech_param']['core']['operating_area_per_mcu'] = operating_area_per_mcu * area_scaling
    config_dict['tech_param']['core']['nominal_voltage'] = nominal_voltage * vol_scaling
    
    with open(modified_config, 'w') as f:
        _yaml.dump(config_dict, f, default_flow_style=False)

    vol_scaling   = vol_scaling * 0.9
    area_scaling  = area_scaling * 0.67
    power_scaling = power_scaling * 0.54

    return vol_scaling, area_scaling, power_scaling


#################################################
root="/mnt/home/newsha/baidu/developement/MechaFlow"
exp_dir="/mnt/scratch/newsha/baidu/developement/MechaFlow/case_study/arch_search/energy_per_flop"
base_config = "{}/configs/exp_config.yaml".format(root)
lp = 1
debug=False
no_launch=True
num_search=1 #20
num_wafer=64
wafer_dim=1
num_worker=64
batch_size=4096
hidden_dim=19968
    
now = datetime.now()
current_time = now.strftime("%d:%m:%Y-%H:%M:%S")
cmdf = 'commands' + current_time

tech_nodes = [14, 10, 7, 5, 3, 2, 1]
for node in tech_nodes:
  out_dir = "{}/node{}".format(exp_dir, node)
  vol_scaling, area_scaling, power_scaling = create_config(base_config, out_dir,  1, 1, 1)
  exp_config = "{}/exp_config.yaml".format(out_dir)
  #call_arch_search(exp_config, out_dir, debug, no_launch, num_search, num_wafer, wafer_dim, num_worker, batch_size, hidden_dim)
  cmd = "python3 main.py arch_search --exp_dir {exp_dir} --exp_config configs/exp_config.yaml --no_launch True --num_search 10 --wafer_dim 8 --num_wafer 1 --batch_size 4096 --hidden_dim 19968 | grep sbatch >> {cmdf}".format(exp_dir=out_dir, cmdf=cmdf)
  cmd_output = subprocess.check_output(cmd, shell=True)

with open(cmdf) as f:
    my_jobs = f.readlines()
job_stack = [x.strip() for x in my_jobs]

#now = datetime.now()
#current_time = now.strftime("%H:%M:%S")
#_hour=(current_time.split(':')[0]
#_min=(current_time.split(':')[1])
#_sec=(current_time.split(':')[2]

while len(job_stack) > 0:
    command = job_stack.pop(0)
    print(command)
    output = subprocess.check_output(
        command, stderr=subprocess.STDOUT, shell=True,
    ).decode("utf-8")
    job_id = re.search(r"[Jj]ob [0-9]+", output).group(0)
    job_id = int(job_id.split(' ')[1])
    
    print("JOB {} Submitted!".format(job_id), flush=True)

    tot_jobs = int(subprocess.check_output("squeue | grep ' R ' | wc -l", shell=True).decode('utf-8').strip('\n'))
    my_jobs  = int(subprocess.check_output("squeue | grep 'newsha' | grep ' R ' | wc -l", shell=True).decode('utf-8').strip('\n'))
   
    #tot_gpu = int(subprocess.check_output("/usr/bin/sinfo --format='%n %t %G' | grep gpu | awk -F':' '{sum+=$2}END{print sum}'", shell=True).decode("utf-8").strip('\n'))
    #busy_gpu = int(subprocess.check_output("/usr/bin/squeue --format=' %u %.8T %.14R %.7b' | grep RUNNING | awk -F':' '{sum+=$2}END{print sum}'", shell=True).decode("utf-8").strip('\n'))
    #free_gpu = tot_gpu - busy_gpu
    #threshold = free_gpu * 0.15

    print(my_jobs, tot_jobs)
    while my_jobs > 200:
        print(my_jobs, tot_jobs)
        time.sleep(1)
        my_jobs = int(subprocess.check_output("squeue | grep 'newsha' | grep ' R ' | wc -l", shell=True).decode('utf-8').strip('\n'))
 
