import subprocess
import time
import re
from datetime import datetime

now = datetime.now()
current_time = now.strftime("%d:%m:%Y-%H:%M:%S")
cmdf = 'commands' + current_time

##################### Technology: Volta ######################
##TDP: 300
#exp_name='power300_packVolta-tech14-worker64-wafer64-' + current_time
#exp_dir='/mnt/scratch/newsha/MechaFlow/' + exp_name
#
#cmd = "python3 main.py arch_search --exp_dir {exp_dir} --exp_config configs/power/power300/exp_config.yaml --no_launch True --num_search 10 --wafer_dim 1 --num_wafer 64 --batch_size 2048 --hidden_dim 19968 | grep sbatch >> {cmdf}".format(exp_dir=exp_dir, cmdf=cmdf)
#cmd_output = subprocess.check_output(cmd, shell=True)
#
##TDP: 600
#exp_name='power600_packVolta-tech14-worker64-wafer64-' + current_time
#exp_dir='/mnt/scratch/newsha/MechaFlow/' + exp_name
#
#cmd = "python3 main.py arch_search --exp_dir {exp_dir} --exp_config configs/power/power600/exp_config.yaml --no_launch True --num_search 10 --wafer_dim 1 --num_wafer 64 --batch_size 2048 --hidden_dim 19968 | grep sbatch >> {cmdf}".format(exp_dir=exp_dir, cmdf=cmdf)
#cmd_output = subprocess.check_output(cmd, shell=True)
#
##TDP: 1200
#exp_name='power1200_packVolta-tech14-worker64-wafer64-' + current_time
#exp_dir='/mnt/scratch/newsha/MechaFlow/' + exp_name
#
#cmd = "python3 main.py arch_search --exp_dir {exp_dir} --exp_config configs/power/power1200/exp_config.yaml --no_launch True --num_search 10 --wafer_dim 1 --num_wafer 64 --batch_size 2048 --hidden_dim 19968 | grep sbatch >> {cmdf}".format(exp_dir=exp_dir, cmdf=cmdf)
#cmd_output = subprocess.check_output(cmd, shell=True)
#
#
##################### Technology: MCM ########################
##TDP: 300
#exp_name='power300_packMCM-tech14-worker64-wafer16-' + current_time
#exp_dir='/mnt/scratch/newsha/MechaFlow/' + exp_name
#
#cmd = "python3 main.py arch_search --exp_dir {exp_dir} --exp_config configs/power/power300/exp_config_MCM.yaml --no_launch True --num_search 10 --wafer_dim 2 --num_wafer 16 --batch_size 2048 --hidden_dim 19968 | grep sbatch >> {cmdf}".format(exp_dir=exp_dir, cmdf=cmdf)
#cmd_output = subprocess.check_output(cmd, shell=True)
#
##TDP: 600
#exp_name='power600_packMCM-tech14-worker64-wafer16-' + current_time
#exp_dir='/mnt/scratch/newsha/MechaFlow/' + exp_name
#
#cmd = "python3 main.py arch_search --exp_dir {exp_dir} --exp_config configs/power/power600/exp_config_MCM.yaml --no_launch True --num_search 10 --wafer_dim 2 --num_wafer 16 --batch_size 2048 --hidden_dim 19968 | grep sbatch >> {cmdf}".format(exp_dir=exp_dir, cmdf=cmdf)
#cmd_output = subprocess.check_output(cmd, shell=True)
#
##TDP: 1200
#exp_name='power1200_packMCM-tech14-worker64-wafer16-' + current_time
#exp_dir='/mnt/scratch/newsha/MechaFlow/' + exp_name
#
#cmd = "python3 main.py arch_search --exp_dir {exp_dir} --exp_config configs/power/power1200/exp_config_MCM.yaml  --no_launch True --num_search 10 --wafer_dim 2 --num_wafer 16 --batch_size 2048 --hidden_dim 19968 | grep sbatch >> {cmdf}".format(exp_dir=exp_dir, cmdf=cmdf)
#cmd_output = subprocess.check_output(cmd, shell=True)
#

#################### Technology: SiIF #########################
#TDP: 300
#vdd: 1.8
exp_name='power300_vdd1.8_packSiIF-tech10-worker64-wafer1-' + current_time
exp_dir='/mnt/scratch/newsha/MechaFlow/' + exp_name

cmd = "python3 main.py arch_search --exp_dir {exp_dir} --exp_config configs/power/vdd1.8/power300/exp_config_SiIF.yaml --no_launch True --num_search 10 --wafer_dim 8 --num_wafer 1 --batch_size 2048 --hidden_dim 19968 | grep sbatch >> {cmdf}".format(exp_dir=exp_dir, cmdf=cmdf)
cmd_output = subprocess.check_output(cmd, shell=True)

#TDP: 400
#vdd: 1.8
exp_name='power400_vdd1.8_packSiIF-tech10-worker64-wafer1-' + current_time
exp_dir='/mnt/scratch/newsha/MechaFlow/' + exp_name

cmd = "python3 main.py arch_search --exp_dir {exp_dir} --exp_config configs/power/vdd1.8/power400/exp_config_SiIF.yaml --no_launch True --num_search 10 --wafer_dim 8 --num_wafer 1 --batch_size 2048 --hidden_dim 19968 | grep sbatch >> {cmdf}".format(exp_dir=exp_dir, cmdf=cmdf)
cmd_output = subprocess.check_output(cmd, shell=True)

#TDP: 500
#vdd: 1.8
exp_name='power500_vdd1.8_packSiIF-tech10-worker64-wafer1-' + current_time
exp_dir='/mnt/scratch/newsha/MechaFlow/' + exp_name

cmd = "python3 main.py arch_search --exp_dir {exp_dir} --exp_config configs/power/vdd1.8/power500/exp_config_SiIF.yaml --no_launch True --num_search 10 --wafer_dim 8 --num_wafer 1 --batch_size 2048 --hidden_dim 19968 | grep sbatch >> {cmdf}".format(exp_dir=exp_dir, cmdf=cmdf)
cmd_output = subprocess.check_output(cmd, shell=True)

#TDP: 600
#vdd: 1.8
exp_name='power600_vdd1.8_packSiIF-tech10-worker64-wafer1-' + current_time
exp_dir='/mnt/scratch/newsha/MechaFlow/' + exp_name

cmd = "python3 main.py arch_search --exp_dir {exp_dir} --exp_config configs/power/vdd1.8/power600/exp_config_SiIF.yaml --no_launch True --num_search 10 --wafer_dim 8 --num_wafer 1 --batch_size 2048 --hidden_dim 19968 | grep sbatch >> {cmdf}".format(exp_dir=exp_dir, cmdf=cmdf)
cmd_output = subprocess.check_output(cmd, shell=True)

#TDP: 800
#vdd: 1.8
exp_name='power800_vdd1.8_packSiIF-tech10-worker64-wafer1-' + current_time
exp_dir='/mnt/scratch/newsha/MechaFlow/' + exp_name

cmd = "python3 main.py arch_search --exp_dir {exp_dir} --exp_config configs/power/vdd1.8/power800/exp_config_SiIF.yaml --no_launch True --num_search 10 --wafer_dim 8 --num_wafer 1 --batch_size 2048 --hidden_dim 19968 | grep sbatch >> {cmdf}".format(exp_dir=exp_dir, cmdf=cmdf)
cmd_output = subprocess.check_output(cmd, shell=True)


#TDP: 1000
#vdd: 1.8
exp_name='power1000_vdd1.8_packSiIF-tech10-worker64-wafer1-' + current_time
exp_dir='/mnt/scratch/newsha/MechaFlow/' + exp_name

cmd = "python3 main.py arch_search --exp_dir {exp_dir} --exp_config configs/power/vdd1.8/power1000/exp_config_SiIF.yaml --no_launch True --num_search 10 --wafer_dim 8 --num_wafer 1 --batch_size 2048 --hidden_dim 19968 | grep sbatch >> {cmdf}".format(exp_dir=exp_dir, cmdf=cmdf)
cmd_output = subprocess.check_output(cmd, shell=True)

##TDP: 300
##vdd: 3.0
#exp_name='power300_vdd3.0_packSiIF-tech14-worker64-wafer1-' + current_time
#exp_dir='/mnt/scratch/newsha/MechaFlow/' + exp_name
#
#cmd = "python3 main.py arch_search --exp_dir {exp_dir} --exp_config configs/power/vdd3.0/power300/exp_config_SiIF.yaml --no_launch True --num_search 10 --wafer_dim 8 --num_wafer 1 --batch_size 2048 --hidden_dim 19968 | grep sbatch >> {cmdf}".format(exp_dir=exp_dir, cmdf=cmdf)
#cmd_output = subprocess.check_output(cmd, shell=True)
#
##TDP: 400
##vdd: 3.0
#exp_name='power400_vdd3.0_packSiIF-tech14-worker64-wafer1-' + current_time
#exp_dir='/mnt/scratch/newsha/MechaFlow/' + exp_name
#
#cmd = "python3 main.py arch_search --exp_dir {exp_dir} --exp_config configs/power/vdd3.0/power400/exp_config_SiIF.yaml --no_launch True --num_search 10 --wafer_dim 8 --num_wafer 1 --batch_size 2048 --hidden_dim 19968 | grep sbatch >> {cmdf}".format(exp_dir=exp_dir, cmdf=cmdf)
#cmd_output = subprocess.check_output(cmd, shell=True)
#
##TDP: 500
##vdd: 3.0
#exp_name='power500_vdd3.0_packSiIF-tech14-worker64-wafer1-' + current_time
#exp_dir='/mnt/scratch/newsha/MechaFlow/' + exp_name
#
#cmd = "python3 main.py arch_search --exp_dir {exp_dir} --exp_config configs/power/vdd3.0/power500/exp_config_SiIF.yaml --no_launch True --num_search 10 --wafer_dim 8 --num_wafer 1 --batch_size 2048 --hidden_dim 19968 | grep sbatch >> {cmdf}".format(exp_dir=exp_dir, cmdf=cmdf)
#cmd_output = subprocess.check_output(cmd, shell=True)
#
##TDP: 600
##vdd: 3.0
#exp_name='power600_vdd3.0_packSiIF-tech14-worker64-wafer1-' + current_time
#exp_dir='/mnt/scratch/newsha/MechaFlow/' + exp_name
#
#cmd = "python3 main.py arch_search --exp_dir {exp_dir} --exp_config configs/power/vdd3.0/power600/exp_config_SiIF.yaml --no_launch True --num_search 10 --wafer_dim 8 --num_wafer 1 --batch_size 2048 --hidden_dim 19968 | grep sbatch >> {cmdf}".format(exp_dir=exp_dir, cmdf=cmdf)
#cmd_output = subprocess.check_output(cmd, shell=True)
#
##TDP: 800
##vdd: 3.0
#exp_name='power800_vdd3.0_packSiIF-tech14-worker64-wafer1-' + current_time
#exp_dir='/mnt/scratch/newsha/MechaFlow/' + exp_name
#
#cmd = "python3 main.py arch_search --exp_dir {exp_dir} --exp_config configs/power/vdd3.0/power800/exp_config_SiIF.yaml --no_launch True --num_search 10 --wafer_dim 8 --num_wafer 1 --batch_size 2048 --hidden_dim 19968 | grep sbatch >> {cmdf}".format(exp_dir=exp_dir, cmdf=cmdf)
#cmd_output = subprocess.check_output(cmd, shell=True)
#
#
##TDP: 1000
##vdd: 3.0
#exp_name='power1000_vdd3.0_packSiIF-tech14-worker64-wafer1-' + current_time
#exp_dir='/mnt/scratch/newsha/MechaFlow/' + exp_name
#
#cmd = "python3 main.py arch_search --exp_dir {exp_dir} --exp_config configs/power/vdd3.0/power1000/exp_config_SiIF.yaml --no_launch True --num_search 10 --wafer_dim 8 --num_wafer 1 --batch_size 2048 --hidden_dim 19968 | grep sbatch >> {cmdf}".format(exp_dir=exp_dir, cmdf=cmdf)
#cmd_output = subprocess.check_output(cmd, shell=True)
#


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
