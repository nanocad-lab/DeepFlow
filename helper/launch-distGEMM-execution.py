import subprocess

RC=True
CR=False

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

a = []
dims= []
for i in range(1,9):
  a.append(i*4096)

for m in a:
  for n in a:
    for k in a:
      dims.append((m,n,k))

 
num_gpus = [1, 2, 4, 8]
output_file = "../results/measured_time/gemm/v100/parallel/v100_tensorcore_enabled_half_precision/RC"

with open(output_file, "w") as f:
  f.write("type ps(num_gpus,kp1,kp2) dim(m,n,k) time(sec.)\n\n")

#RC
if RC:
  for N in num_gpus: 
    for dim in dims:
      result = []
      pss = []
      m, n, k = dim
      findMultipliers(N, 1, pss, result, 2)
      for ps in pss:
        kp1 = ps[0]
        kp2 = ps[1]
        if kp2 == 1:
          command = "/tools/cuda/cuda-10.2.89/bin/nvprof --print-gpu-trace python dist_gemm_tf2_v9.py -kp1 {} -kp2 {} -N {} -t RC -m {} -n {} -k {} 2>&1 | grep gemm | tail -n+3 | awk '{{sum+=$2}}END{{print sum/NR}}'".format(kp1, kp2, N, m, n, k)
          output = subprocess.check_output(
                   command, stderr=subprocess.STDOUT, shell=True,
                   ).decode("utf-8")
        else:
          command = "python dist_gemm_tf2_v7.py -m {} -n {} -k {} -t RC -kp1 {} -kp2 {} -N {} 2>&1 | grep Step | tail -n 8 | awk '{{sum+=$2}}END{{print sum/NR}}'".format(m, n, k, kp1, kp2, N)
          output = subprocess.check_output(
                   command, stderr=subprocess.STDOUT, shell=True,
                   ).decode("utf-8")
        with open(output_file, "a+") as f:
          f.write("RC ps(N{},{},{}) dim({},{},{}) {}".format(N, kp1, kp2, m, n, k, output))

#CR
if CR:
  for N in num_gpus: 
    for dim in dims:
      kp1 = N
      kp2 = '-'
      m, n, k = dim
      command = "python dist_gemm_tf2_v7.py -m {} -n {} -k {} -t CR -kp1 {} -N {} 2>&1 | grep Step | tail -n 8 | awk '{{sum+=$2}}END{{print sum/NR}}'".format(m, n, k, kp1, N)
      output = subprocess.check_output(
               command, stderr=subprocess.STDOUT, shell=True,
               ).decode("utf-8")
      with open(output_file, "a+") as f:
        f.write("CR ps(N{},{},{}) dim({},{},{}) {}".format(N, kp1, kp2, m, n, k, output))
  
