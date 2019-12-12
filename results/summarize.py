import os

cid=['c', 'mbw', 'ms', 'L2bw', 'L2s', 'SMbw', 'SMs', 'Rbw', 'Rs', 'i', 'o']

flist=['exp_config_14nm', 'exp_config_SiIF_14nm', 'exp_config_SiIF_10nm_NOSTATIC']
exp_root='/mnt/scratch/newsha/MechaFlow/arch_search'
for fname in flist:
  print("===================================================================================================")
  print(fname)
  print("===================================================================================================")
  dp_list=[1, 32]
  batch_list=[32, 512]
  
  for dp in dp_list:
    for b in batch_list:
      minT=float("inf")
      minLine=''
      with open(fname, 'r') as f:
        for cnt, line in enumerate(f):
            sline=line.strip().split()
            time=float(sline[1])
            if ("dp{}/b{}".format(dp, b) in sline[0] and time < minT):
              minT = time
              minLine = line

      print("{}".format(minLine))
      best_design=minLine.strip().split()[0]
      best_file=exp_root+"/"+fname+"_Sep24/"+best_design+"/best.txt"
      best_dir=''
      if os.path.isfile(best_file):
          with open(best_file, 'r') as b:
               lines=b.readlines()
               line=lines[2]
               assert("Best Dir" in line)
               best_dir=line.strip().split()[2]

      if os.path.isfile(best_dir+"/summary.txt"):
          with open(best_dir+"/summary.txt", 'r') as c:
              lines=c.readlines()
              config=best_design
              for i in range(3,14):
                  conf=lines[i].strip().split(':')[1]
                  config = config + " \t" + cid[i-3] + ": " + conf
              print(config)
              
