import subprocess
import os
import shutil
import re

def findMultipliers(n, curr_depth, results, result, max_depth):
    if curr_depth == max_depth:
        result.append(n)
        results.append(result)
        return
    i = 1
    while i <= n:
        r = result[:]
        r.append(i)
        findMultipliers(n//i, curr_depth+1, results, r, max_depth)
        i = i * 2

def create_sbatch_command(t, N, kp1, kp2, m, n, k, exp_dir, exp_config):
  partition='1080Ti,TitanXx8,M40x8,2080Ti,P100,TitanXx8_short,TitanXx8_mlong,TitanXx8_slong,M40x8_slong,M40x8_mlong,1080Ti_mlong,1080Ti_slong,1080Ti_short,2080Ti_mlong'
  gpus_per_node = 1
  outdir = ''
  script_args = 'NULL'
  job_name='NULL'
  if t == 'RC':
    job_name="{}_N{}_K{}_K{}_m{}_n{}_k{}".format(t, N, kp1, kp2, m, n, k)
    out_dir = "{}/{}/{}".format(exp_dir, exp_config, job_name)
    script_args = ('--exp_config configs/{exp_config}.yaml ' 
                   '--exp_dir {exp_dir} '
                   '--gemm True '
                   '--m {m} --n {n} --k {k} '
                   '--t {t} --kp1 {kp1} --kp2 {kp2}'.format(
                      exp_dir=out_dir,
                      exp_config=exp_config,
                      m=m,
                      n=n,
                      k=k,
                      t=t,
                      kp1=kp1,
                      kp2=kp2)
                   )
  elif t == 'CR':
    job_name="{}_N{}_K{}_m{}_n{}_k{}".format(t, N, kp1, m, n, k)
    out_dir = "{}/{}/{}".format(exp_dir, exp_config, job_name)
    script_args = ('--exp_config configs/{exp_config}.yaml ' 
                   '--exp_dir {exp_dir} '
                   '--gemm True '
                   '--m {m} --n {n} --k {k} '
                   '--t {t} --kp1 {kp1} --kp2 {kp2}'.format(
                      exp_dir=out_dir,
                      exp_config=exp_config,
                      m=m,
                      n=n,
                      k=k,
                      t=t,
                      kp1=kp1,
                      kp2=kp2)
                   )
  else:
    print("Type can be either CR or RC")
    exit(0)



  if os.path.exists(out_dir):
    shutil.rmtree(out_dir, ignore_errors=True)
  os.makedirs(out_dir)

  script = 'perf.py'
  command = (
      'sbatch --job-name={job_name} '
      '--gres=gpu:{gpus_per_node} '
      '--partition={partition} '
      '--wrap="srun stdbuf -i0 -o0 -e0 '
      'python {script} ' +
      script_args + 
      '" ' + 
      '-o {out_dir}/slurm-%j.out').format(
        job_name=job_name,
        gpus_per_node=gpus_per_node,
        partition=partition,
        script=script,
        out_dir=out_dir)

  print(command)
  return command
  
def run_slurm_command(command):
    output = subprocess.check_output(
             command, stderr=subprocess.STDOUT, shell=True,
             ).decode("utf-8")
    job_id = re.search(r"[Jj]ob [0-9]+", output).group(0)
    job_id = int(job_id.split(' ')[1])
    
    print("JOB {} Submitted!".format(job_id), flush=True)

def create_gemm_dims():
  a = [2048]
  dims= []
  for i in range(1,9):
    a.append(i*4096)
  
  for m in a:
    for n in a:
      for k in a:
        dims.append((m,n,k))
  return dims



  
num_gpus = [1, 2, 4, 8]
exp_dir = '/mnt/home/newsha/baidu/developement/MechaFlow/validation/results/predictions/gemm/v100/parallel-core_0.9_l2_0.63'
configs = ["v100-gemm-validation"] #, "v100_tensorcore_disabled_half_precision", "v100_tensorcore_disabled_float_precision"]

dims = create_gemm_dims()

for config in configs:
  for t in ['RC','CR']:
    for N in num_gpus: 
      for dim in dims:
        result = []
        pss = []
        m, n, k = dim
        findMultipliers(N, 1, pss, result, 2)
        for ps in pss:
          kp1 = ps[0]
          kp2 = ps[1]
          command = create_sbatch_command(t, N, kp1, kp2, m, n, k, exp_dir, config)
          run_slurm_command(command)
  
