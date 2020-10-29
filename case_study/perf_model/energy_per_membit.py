
import subprocess
import os
import shutil
import re
import ruamel as _ruamel
import ruamel.yaml as _yaml

def create_sbatch_command(exp_dir, feature, tech_node, exp_config, N, kp1, kp2, kp_type, dp, lp):
  partition='1080Ti,TitanXx8,M40x8,2080Ti,P100,TitanXx8_short,TitanXx8_mlong,TitanXx8_slong,M40x8_slong,M40x8_mlong,1080Ti_mlong,1080Ti_slong,1080Ti_short,2080Ti_mlong'
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


def sweep_DRAM(base_config, exp_dir, N, kp1, kp2, kp_type, dp, lp):
  config_dict = {}
  #read the base config
  with open(base_config, "r") as f:
    config_dict = _yaml.load(f, Loader=_ruamel.yaml.Loader)
    dynamic_energy_per_bit = config_dict['tech_param']['DRAM']['dynamic_energy_per_bit']
    static_power_per_bit = config_dict['tech_param']['DRAM']['static_power_per_bit']
    nominal_voltage = config_dict['tech_param']['DRAM']['nominal_voltage']
  

  vol_scaling   = 1
  static_power_scaling = 1
  energy_scaling = 1

  tech_nodes = ['HBM1', 'HBM2','HBM2e-8','HBM2e-12','HBM2e-16']
  area_per_stack = [100, 100, 110, 110, 110]
  num_links_per_stack = [1024, 1024, 1024, 1536, 2048]
  capacity_per_stack = [i * 1024 * 1024 for i in [4, 8, 16, 24, 32]]

  freq = [1.38e9, 2e9, 2.4e9, 3.2e9, 4e9]

  for i, node in enumerate(tech_nodes):
      #modify the feature
      out_dir = "{}/{}/{}_N{}_k{}_k{}_d{}_l{}/{}".format(exp_dir, "energy_per_membit", kp_type, N, kp1, kp2, dp, lp, node)
      
      if os.path.exists(out_dir):
        shutil.rmtree(out_dir, ignore_errors=True)
      os.makedirs(out_dir)
      
      modified_config = "{}/exp_config.yaml".format(out_dir)
      config_dict['tech_param']['DRAM']['dynamic_energy_per_bit'] = dynamic_energy_per_bit * energy_scaling
      config_dict['tech_param']['DRAM']['static_power_per_bit']   = static_power_per_bit * static_power_scaling
      config_dict['tech_param']['DRAM']['nominal_voltage']        = nominal_voltage * vol_scaling
      config_dict['tech_param']['DRAM']['freq']                   = freq[i]
      config_dict['tech_param']['DRAM']['area_per_stack']         = area_per_stack[i]
      config_dict['tech_param']['DRAM']['capacity_per_stack']     = capacity_per_stack[i]
      config_dict['tech_param']['DRAM']['num_links_per_stack']    = num_links_per_stack[i]
      
      with open(modified_config, 'w') as f:
          _yaml.dump(config_dict, f, default_flow_style=False)
      #launch batch job
      command = create_sbatch_command(exp_dir, "energy_per_membit", node, modified_config, N, kp1, kp2, kp_type, dp, lp)
      run_slurm_command(command)

      vol_scaling   = vol_scaling * 0.95
      staic_power_scaling  = static_power_scaling * 1.2
      energy_scaling = energy_scaling * 0.7




#################################################
root="/mnt/home/newsha/baidu/developement/MechaFlow"
exp_dir="/mnt/home/newsha/baidu/developement/MechaFlow/case_study/perf_model"
num_gpus = [64]
configs = ['v100'] #,'SiIF', 'MCM']
lp = 1


for config in configs:
  base_config = "{}/configs/{}-large.yaml".format(root, config)
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
          
          sweep_DRAM(base_config, exp_dir, N, kp1, kp2, t, dp, lp)
  
      elif t == 'CR':
        result = []
        pss = []
        findMultipliers(N, 1, pss, result, 2)
        for ps in pss:
          kp1 = ps[0]
          kp2 = 1
          dp  = ps[1]
  
          sweep_DRAM(base_config, exp_dir, N, kp1, kp2, t, dp, lp)
        
