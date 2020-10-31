
import subprocess
import os
import shutil
import re
import ruamel as _ruamel
import ruamel.yaml as _yaml

def create_sbatch_command(exp_dir, feature, tech_node, exp_config, N, kp1, kp2, kp_type, dp, lp):
  partition='1080Ti,M40x8,2080Ti,P100,TitanXx8_short,TitanXx8_mlong,TitanXx8_slong,M40x8_slong,M40x8_mlong,1080Ti_mlong,1080Ti_slong,1080Ti_short,2080Ti_mlong'
  gpus_per_node = 1
  batch_size = 4096 
  seq_len = 20
  hidden_dim = 19968 
  vocab_size = 800000
  num_layer = 2
  job_name="{}/{}_N{}_k{}_k{}_d{}_l{}/{}".format(feature, kp_type, N, kp1, kp2, dp, lp, tech_node)
  out_dir = "{}/{}".format(exp_dir, job_name)
  script_args = ('--exp_config {exp_config} ' 
                 '--exp_dir {exp_dir} '
                 '--debug True '
                 '--batch_size {batch_size} '
                 '--seq_len {seq_len} '
                 '--hidden_dim {hidden_dim} '
                 '--vocab_size {vocab_size} '
                 '--num_layer {num_layer} '
                 '--dp {dp} ' 
                 '--lp {lp} '
                 '--t {kp_type} '
                 '--kp1 {kp1} '
                 '--kp2 {kp2}'.format(
                    exp_config=exp_config,
                    exp_dir=out_dir,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    hidden_dim=hidden_dim,
                    vocab_size=vocab_size,
                    num_layer=num_layer,
                    dp=dp,
                    lp=lp,
                    kp1=kp1,
                    kp2=kp2,
                    kp_type=kp_type)
                 )


  script = 'perf.py'
  command = (
      'sbatch --job-name={job_name} '
      '--gres=gpu:{gpus_per_node} '
      '--partition={partition} '
      '--exclude asimov-135 '
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



def create_command(exp_dir, feature, tech_node, exp_config, N, kp1, kp2, kp_type, dp, lp):
  partition='debug_1080Ti' #'1080Ti,M40x8,2080Ti,P100,TitanXx8_short,TitanXx8_mlong,TitanXx8_slong,M40x8_slong,M40x8_mlong,1080Ti_mlong,1080Ti_slong,1080Ti_short,2080Ti_mlong'
  gpus_per_node = 1
  batch_size = 4096 
  seq_len = 20
  hidden_dim = 19968 
  vocab_size = 800000
  num_layer = 2
  job_name="{}/{}_N{}_k{}_k{}_d{}_l{}/{}".format(feature, kp_type, N, kp1, kp2, dp, lp, tech_node)
  out_dir = "{}/{}".format(exp_dir, job_name)
  script_args = ('--exp_config {exp_config} ' 
                 '--exp_dir {exp_dir} '
                 '--debug True '
                 '--batch_size {batch_size} '
                 '--seq_len {seq_len} '
                 '--hidden_dim {hidden_dim} '
                 '--vocab_size {vocab_size} '
                 '--num_layer {num_layer} '
                 '--dp {dp} ' 
                 '--lp {lp} '
                 '--t {kp_type} '
                 '--kp1 {kp1} '
                 '--kp2 {kp2}'.format(
                    exp_config=exp_config,
                    exp_dir=out_dir,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    hidden_dim=hidden_dim,
                    vocab_size=vocab_size,
                    num_layer=num_layer,
                    dp=dp,
                    lp=lp,
                    kp1=kp1,
                    kp2=kp2,
                    kp_type=kp_type)
                 )


  script = 'perf.py'
  command = "python {script} {script_args}".format(script=script,
                                            script_args=script_args) 

  print(command)
  return command
 


def run_slurm_command(command):
    output = subprocess.check_output(
             command, stderr=subprocess.STDOUT, shell=True,
             ).decode("utf-8")
    job_id = re.search(r"[Jj]ob [0-9]+", output).group(0)
    job_id = int(job_id.split(' ')[1])
    
    print("JOB {} Submitted!".format(job_id), flush=True)

def run_command(command):
    output = subprocess.check_output(
             command, stderr=subprocess.STDOUT, shell=True,
             ).decode("utf-8")


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


def sweep_core(base_config, exp_dir, N, kp1, kp2, kp_type, dp, lp):
  config_dict = {}
  #read the base config
  with open(base_config, "r") as f:
    config_dict = _yaml.load(f, Loader=_ruamel.yaml.Loader)
    
    energy_per_link = config_dict['tech_param']['network']['inter_node']['nominal_energy_per_link']
    area_per_link    = config_dict['tech_param']['network']['inter_node']['nominal_area_per_link']
  
    
    
  area_scaling  = 1
  energy_scaling = 1

  tech_nodes = [i for i in range(1,20)]
  for node in tech_nodes:
      #modify the feature
      out_dir = "{}/{}/{}_N{}_k{}_k{}_d{}_l{}/{}".format(exp_dir, "energy_per_xbit", kp_type, N, kp1, kp2, dp, lp, node)
      
      if os.path.exists(out_dir):
        shutil.rmtree(out_dir, ignore_errors=True)
      os.makedirs(out_dir)
      
      modified_config = "{}/exp_config.yaml".format(out_dir)
      config_dict['tech_param']['network']['inter_node']['nominal_area_per_link'] = area_per_link * area_scaling 
      config_dict['tech_param']['network']['inter_node']['nominal_energy_per_link'] = energy_per_link * energy_scaling 

      
      with open(modified_config, 'w') as f:
        _yaml.dump(config_dict, f, default_flow_style=False)
      #launch batch job
      command = create_sbatch_command(exp_dir, "energy_per_xbit", node, modified_config, N, kp1, kp2, kp_type, dp, lp)
      run_slurm_command(command)
      #command = create_command(exp_dir, "energy_per_xbit_debug_true", node, modified_config, N, kp1, kp2, kp_type, dp, lp)
      #run_command(command)

      area_scaling    = area_scaling * 0.75
      energy_scaling  = energy_scaling * 0.5




#################################################
root="/mnt/home/newsha/baidu/developement/MechaFlow"
exp_dir="/mnt/home/newsha/baidu/developement/MechaFlow/case_study/perf_model"
#base_config = "{}/configs/exp_config_SiIF-large.yaml".format(root)
base_config = "{}/configs/v100-large.yaml".format(root)
num_gpus = [64]
lp = 1

for t in ['RC', 'CR']:
  for N in num_gpus:
    if t == 'RC':
      result = []
      pss = []
      findMultipliers(N, 1, pss, result, 3)
      for ps in pss:
        kp1 = ps[0]
        kp2 = ps[1]
        dp  = ps[2]
        
        sweep_core(base_config, exp_dir, N, kp1, kp2, t, dp, lp)

    elif t == 'CR':
      result = []
      pss = []
      findMultipliers(N, 1, pss, result, 2)
      for ps in pss:
        kp1 = ps[0]
        kp2 = 1
        dp  = ps[1]

        sweep_core(base_config, exp_dir, N, kp1, kp2, t, dp, lp)
        
