import subprocess
import time
import re
from datetime import datetime

now = datetime.now()
current_time = now.strftime("%d:%m:%Y-%H:%M:%S")
cmdf = 'commands' + current_time


################### Batch SiIF
#Batch: 64#
exp_name='batch64-packSiIF-tech14-worker64-wafer1-' + current_time
exp_dir='/mnt/scratch/newsha/MechaFlow/' + exp_name

cmd = "python3 main.py arch_search --exp_dir {exp_dir} --exp_config configs/exp_config_SiIF.yaml --no_launch True --num_search 10 --wafer_dim 8 --num_wafer 1 --batch_size 64 --hidden_dim 19968 | grep sbatch >> {cmdf}".format(exp_dir=exp_dir, cmdf=cmdf)
cmd_output = subprocess.check_output(cmd, shell=True)

#Batch: 256#
#exp_name='batch256-packSiIF-tech14-worker64-wafer1-' + current_time
#exp_dir='/mnt/scratch/newsha/MechaFlow/' + exp_name
#
#cmd = "python3 main.py arch_search --exp_dir {exp_dir} --exp_config configs/exp_config_SiIF.yaml --no_launch True --num_search 10 --wafer_dim 8 --num_wafer 1 --batch_size 256 --hidden_dim 19968 | grep sbatch >> {cmdf}".format(exp_dir=exp_dir, cmdf=cmdf)
#cmd_output = subprocess.check_output(cmd, shell=True)

#Batch: 1024#
exp_name='batch1024-packSiIF-tech14-worker64-wafer1-' + current_time
exp_dir='/mnt/scratch/newsha/MechaFlow/' + exp_name

cmd = "python3 main.py arch_search --exp_dir {exp_dir} --exp_config configs/exp_config_SiIF.yaml --no_launch True --num_search 10 --wafer_dim 8 --num_wafer 1 --batch_size 1024 --hidden_dim 19968 | grep sbatch >> {cmdf}".format(exp_dir=exp_dir, cmdf=cmdf)
cmd_output = subprocess.check_output(cmd, shell=True)

#Batch: 4096#
exp_name='batch4096-packSiIF-tech14-worker64-wafer1-' + current_time
exp_dir='/mnt/scratch/newsha/MechaFlow/' + exp_name

cmd = "python3 main.py arch_search --exp_dir {exp_dir} --exp_config configs/exp_config_SiIF.yaml --no_launch True --num_search 10 --wafer_dim 8 --num_wafer 1 --batch_size 4096 --hidden_dim 19968 | grep sbatch >> {cmdf}".format(exp_dir=exp_dir, cmdf=cmdf)
cmd_output = subprocess.check_output(cmd, shell=True)

#Batch: 8192#
exp_name='batch8192-packSiIF-tech14-worker64-wafer1-' + current_time
exp_dir='/mnt/scratch/newsha/MechaFlow/' + exp_name

cmd = "python3 main.py arch_search --exp_dir {exp_dir} --exp_config configs/exp_config_SiIF.yaml --no_launch True --num_search 10 --wafer_dim 8 --num_wafer 1 --batch_size 8192 --hidden_dim 19968 | grep sbatch >> {cmdf}".format(exp_dir=exp_dir, cmdf=cmdf)
cmd_output = subprocess.check_output(cmd, shell=True)

#Batch: 16384#
exp_name='batch16384-packSiIF-tech14-worker64-wafer1-' + current_time
exp_dir='/mnt/scratch/newsha/MechaFlow/' + exp_name

cmd = "python3 main.py arch_search --exp_dir {exp_dir} --exp_config configs/exp_config_SiIF.yaml --no_launch True --num_search 10 --wafer_dim 8 --num_wafer 1 --batch_size 16384 --hidden_dim 19968 | grep sbatch >> {cmdf}".format(exp_dir=exp_dir, cmdf=cmdf)
cmd_output = subprocess.check_output(cmd, shell=True)

################### Hidden SiIF
#Hidden_dim: 1024#
exp_name='Hidden1024-packSiIF-tech14-worker64-wafer1-' + current_time
exp_dir='/mnt/scratch/newsha/MechaFlow/' + exp_name

cmd = "python3 main.py arch_search --exp_dir {exp_dir} --exp_config configs/exp_config_SiIF.yaml --no_launch True --num_search 10 --wafer_dim 8 --num_wafer 1 --batch_size 256 --hidden_dim 1024 | grep sbatch >> {cmdf}".format(exp_dir=exp_dir, cmdf=cmdf)
cmd_output = subprocess.check_output(cmd, shell=True)

#Hidden_dim: 4096#
exp_name='Hidden4096-packSiIF-tech14-worker64-wafer1-' + current_time
exp_dir='/mnt/scratch/newsha/MechaFlow/' + exp_name

cmd = "python3 main.py arch_search --exp_dir {exp_dir} --exp_config configs/exp_config_SiIF.yaml --no_launch True --num_search 10 --wafer_dim 8 --num_wafer 1 --batch_size 256 --hidden_dim 4096 | grep sbatch >> {cmdf}".format(exp_dir=exp_dir, cmdf=cmdf)
cmd_output = subprocess.check_output(cmd, shell=True)

#Hidden_dim: 8192#
exp_name='Hidden8192-packSiIF-tech14-worker64-wafer1-' + current_time
exp_dir='/mnt/scratch/newsha/MechaFlow/' + exp_name

cmd = "python3 main.py arch_search --exp_dir {exp_dir} --exp_config configs/exp_config_SiIF.yaml --no_launch True --num_search 10 --wafer_dim 8 --num_wafer 1 --batch_size 256 --hidden_dim 8192 | grep sbatch >> {cmdf}".format(exp_dir=exp_dir, cmdf=cmdf)
cmd_output = subprocess.check_output(cmd, shell=True)

#Hidden_dim: 16384#
exp_name='Hidden16384-packSiIF-tech14-worker64-wafer1-' + current_time
exp_dir='/mnt/scratch/newsha/MechaFlow/' + exp_name

cmd = "python3 main.py arch_search --exp_dir {exp_dir} --exp_config configs/exp_config_SiIF.yaml --no_launch True --num_search 10 --wafer_dim 8 --num_wafer 1 --batch_size 256 --hidden_dim 16384 | grep sbatch >> {cmdf}".format(exp_dir=exp_dir, cmdf=cmdf)
cmd_output = subprocess.check_output(cmd, shell=True)

#Hidden_dim: 19968#

#Hidden_dim: 32768#
exp_name='Hidden32768-packSiIF-tech14-worker64-wafer1-' + current_time
exp_dir='/mnt/scratch/newsha/MechaFlow/' + exp_name

cmd = "python3 main.py arch_search --exp_dir {exp_dir} --exp_config configs/exp_config_SiIF.yaml --no_launch True --num_search 10 --wafer_dim 8 --num_wafer 1 --batch_size 256 --hidden_dim 32768 | grep sbatch >> {cmdf}".format(exp_dir=exp_dir, cmdf=cmdf)
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
        #status = subprocess.check_output("sacct -u newsha -S {}:{}:{} -s F,CA".format(_hour,_min,_sec), shell=True)
