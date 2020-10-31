import subprocess
import time
import re
from datetime import datetime

now = datetime.now()
current_time = now.strftime("%d:%m:%Y-%H:%M:%S")
cmdf = 'commands' + current_time

########################
##scale-out-single-chip-wafer#
########################
exp_name='scaleout-batch4096-packVolta-tech14-worker32-wafer32' + current_time
exp_dir='/mnt/scratch/newsha/MechaFlow/' + exp_name

cmd = "python3 main.py arch_search --exp_dir {exp_dir} --exp_config configs/normalized-v100.yaml --no_launch True --num_search 10 --wafer_dim 1 --num_wafer 32 --num_worker 32 --batch_size 4096 --hidden_dim 19968 | grep sbatch >> {cmdf}".format(exp_dir=exp_dir, cmdf=cmdf)
cmd_output = subprocess.check_output(cmd, shell=True)

#exp_name='scaleout-batch4096-packVolta-tech14-worker64-wafer64' + current_time
#exp_dir='/mnt/scratch/newsha/MechaFlow/' + exp_name

#cmd = "python3 main.py arch_search --exp_dir {exp_dir} --exp_config configs/normalized-v100.yaml --no_launch True --num_search 10 --wafer_dim 1 --num_wafer 64 --num_worker 64 --batch_size 4096 --hidden_dim 19968 | grep sbatch >> {cmdf}".format(exp_dir=exp_dir, cmdf=cmdf)
#cmd_output = subprocess.check_output(cmd, shell=True)

exp_name='scaleout-batch4096-packVolta-tech14-worker128-wafer128' + current_time
exp_dir='/mnt/scratch/newsha/MechaFlow/' + exp_name

cmd = "python3 main.py arch_search --exp_dir {exp_dir} --exp_config configs/normalized-v100.yaml --no_launch True --num_search 10 --wafer_dim 1 --num_wafer 128 --num_worker 128 --batch_size 4096 --hidden_dim 19968 | grep sbatch >> {cmdf}".format(exp_dir=exp_dir, cmdf=cmdf)
cmd_output = subprocess.check_output(cmd, shell=True)



#####################
##Core granularity-per-wafer#
#####################
# #1 x (2 x 2)   = 16
#exp_name='chip_gran-batch256-packSiIF-tech14-worker4-dim2-w1-' + current_time
#exp_dir='/mnt/scratch/newsha/MechaFlow/' + exp_name
#
#cmd = "python3 main.py arch_search --exp_dir {exp_dir} --exp_config configs/exp_config_SiIF_4.yaml --no_launch True --num_wafer 1 --wafer_dim 2 --num_search 10 | grep sbatch >| {cmdf}".format(exp_dir=exp_dir, cmdf=cmdf)
#cmd_output = subprocess.check_output(cmd, shell=True)
#
#
##1 x (4 x 4)   = 16
#exp_name='chip_gran-batch256-packSiIF-tech14-worker16-dim4-w1-' + current_time
#exp_dir='/mnt/scratch/newsha/MechaFlow/' + exp_name
#
#cmd = "python3 main.py arch_search --exp_dir {exp_dir} --exp_config configs/exp_config_SiIF_16.yaml --no_launch True --num_wafer 1 --wafer_dim 4 --num_search 10 | grep sbatch >> {cmdf}".format(exp_dir=exp_dir, cmdf=cmdf)
#cmd_output = subprocess.check_output(cmd, shell=True)
#
##1 x (8 x 8)   = 64
#
##1 x (16 x 16) = 256
#exp_name='chip_gran-batch256-packSiIF-tech14-worker256-dim16-w1-' + current_time
#exp_dir='/mnt/scratch/newsha/MechaFlow/' + exp_name
#
#cmd = "python3 main.py arch_search --exp_dir {exp_dir} --exp_config configs/exp_config_SiIF_256.yaml --no_launch True --num_wafer 1 --wafer_dim 16 --num_search 10 | grep sbatch >> {cmdf}".format(exp_dir=exp_dir, cmdf=cmdf)
#cmd_output = subprocess.check_output(cmd, shell=True)
#
##1 x (32 x 32) = 1024
#exp_name='chip_gran-batch256-packSiIF-tech14-worker1024-dim32-w1-' + current_time
#exp_dir='/mnt/scratch/newsha/MechaFlow/' + exp_name
#
#cmd = "python3 main.py arch_search --exp_dir {exp_dir} --exp_config configs/exp_config_SiIF_1024.yaml --no_launch True --num_wafer 1 --wafer_dim 32 --num_search 10 | grep sbatch >> {cmdf}".format(exp_dir=exp_dir, cmdf=cmdf)
#cmd_output = subprocess.check_output(cmd, shell=True)
#
########################
##scale-out-multi-wafer#
########################
#1 x (8 x 8) = 64 GPUs but assign only 2 workers
#exp_name='scaleout-batch256-packSiIF-tech14-worker2-dim8-w1-' + current_time
#exp_dir='/mnt/scratch/newsha/MechaFlow/' + exp_name

#cmd = "python3 main.py arch_search --exp_dir {exp_dir} --exp_config configs/exp_config_SiIF.yaml --no_launch True --num_wafer 1 --wafer_dim 8 --num_worker 2 --num_search 10 | grep sbatch >> {cmdf}".format(exp_dir=exp_dir, cmdf=cmdf)
#cmd_output = subprocess.check_output(cmd, shell=True)

##1 x (8 x 8) = 64 GPUs but assign only 4 workers
#exp_name='scaleout-batch256-packSiIF-tech14-worker4-dim8-w1-' + current_time
#exp_dir='/mnt/scratch/newsha/MechaFlow/' + exp_name
#
#cmd = "python3 main.py arch_search --exp_dir {exp_dir} --exp_config configs/exp_config_SiIF.yaml --no_launch True --num_wafer 1 --wafer_dim 8 --num_worker 4 --num_search 10 | grep sbatch >> {cmdf}".format(exp_dir=exp_dir, cmdf=cmdf)
#cmd_output = subprocess.check_output(cmd, shell=True)
#
##1 x (8 x 8) = 64 GPUs but assign only 16 workers
#exp_name='scaleout-batch256-packSiIF-tech14-worker16-dim8-w1-' + current_time
#exp_dir='/mnt/scratch/newsha/MechaFlow/' + exp_name
#
#cmd = "python3 main.py arch_search --exp_dir {exp_dir} --exp_config configs/exp_config_SiIF.yaml --no_launch True --num_wafer 1 --wafer_dim 8 --num_worker 16 --num_search 10 | grep sbatch >> {cmdf}".format(exp_dir=exp_dir, cmdf=cmdf)
#cmd_output = subprocess.check_output(cmd, shell=True)
#
##1 x (8 x 8) = 64 workers
#
#2 x (8 x 8) = 128 workers
#exp_name='scaleout-batch256-packSiIF-tech14-worker128-dim8-w2-' + current_time
#exp_dir='/mnt/scratch/newsha/MechaFlow/' + exp_name

#cmd = "python3 main.py arch_search --exp_dir {exp_dir} --exp_config configs/exp_config_SiIF.yaml --no_launch True --num_wafer 2 --wafer_dim 8 --num_search 10 | grep sbatch >> {cmdf}".format(exp_dir=exp_dir, cmdf=cmdf)
#cmd_output = subprocess.check_output(cmd, shell=True)

##4 x (8 x 8) = 256 workers
#exp_name='scaleout-batch256-packSiIF-tech14-worker256-dim8-w4-' + current_time
#exp_dir='/mnt/scratch/newsha/MechaFlow/' + exp_name
#
#cmd = "python3 main.py arch_search --exp_dir {exp_dir} --exp_config configs/exp_config_SiIF.yaml --no_launch True --num_wafer 4 --wafer_dim 8 --num_search 10 | grep sbatch >> {cmdf}".format(exp_dir=exp_dir, cmdf=cmdf)
#cmd_output = subprocess.check_output(cmd, shell=True)
#
##16 x (8 x 8) = 1024 workers
#exp_name='scaleout-batch256-packSiIF-tech14-worker1024-dim8-w16-' + current_time
#exp_dir='/mnt/scratch/newsha/MechaFlow/' + exp_name
#
#cmd = "python3 main.py arch_search --exp_dir {exp_dir} --exp_config configs/exp_config_SiIF.yaml --no_launch True --num_wafer 16 --wafer_dim 8 --num_search 10 | grep sbatch >> {cmdf}".format(exp_dir=exp_dir, cmdf=cmdf)
#cmd_output = subprocess.check_output(cmd, shell=True)

#Next: Launch same as above with Volta and Volta+SiIF intra bandwidth

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
