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

  tech_nodes = [14, 10, 7, 5, 3, 2, 1]
  time = [0] * len(tech_nodes) 
  for i, node in enumerate(tech_nodes):
      par = "{}_N{}_k{}_k{}_d{}_l{}".format(kp_type, N, kp1, kp2, dp, lp)
      out_dir = "{}/{}/{}/{}".format(exp_dir, "siif-energy_per_flop", par, node)
      summary_file = "{}/summary.txt".format(out_dir)
      #read the summary file
      valid = False
      with open(summary_file, "r") as f:
          for line in f:
            if "Total Memory Required per Data Shard Per Model Shard" in line:
              overflow_rate = float(line.split()[-1]) 
              if overflow_rate <= 1:
                valid = True
            if "Time" in line and valid:
              time[i] = float(line.split()[1])

  print("{} {} {} {} {} {} {} {}".format(par, time[0], time[1], time[2], time[3], time[4], time[5], time[6]))
    
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
 
