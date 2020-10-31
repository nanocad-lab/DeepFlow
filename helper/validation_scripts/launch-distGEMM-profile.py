#!/tools/python/python3.8.3/bin/python
import os
import shutil
import subprocess
import numpy as np


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

def run_command(cmd, var, result):
  try:
    output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True).decode("utf-8")
    output = output.strip().replace(',','')
    result[var] = float(output) if output != "" else output
  except:
    print("command for {} did not work".format(var))


a = []
dims= []
for i in range(1,5):
  a.append(i*8192)

for m in a:
  for n in a:
    for k in a:
      dims.append((m,n,k))

 
num_gpus = [1]
mech_root = "/mnt/home/newsha/baidu/developement/MechaFlow" 
output_dir = "{}/validation/results/profiled/gemm/v100/parallel".format(mech_root)
result_file="{}/result.csv".format(output_dir)

if os.path.exists(output_dir):
  shutil.rmtree(output_dir, ignore_errors=True)
os.makedirs(output_dir)
print("Created {}".format(output_dir))

with open(result_file, "w") as f:
  f.write("Type N kp1 kp2 m n k core_util dram_util l2_util dram_read dram_write l2_access fp16_inst fma_inst\n\n")
print("Type N kp1 kp2 m n k core_util dram_util l2_util dram_read dram_write l2_access fp16_inst fma_inst\n\n")



for t in ['RC']:
  for N in num_gpus: 
    for dim in dims:
      if t == "RC":
        result = []
        pss = []
        m, n, k = dim
        findMultipliers(N, 1, pss, result, 2)
        for ps in pss:
          kp1 = ps[0]
          kp2 = ps[1]
          output_file="{}/{}_N{}_K{}_K{}_m{}_n{}_k{}".format(output_dir, t, N, kp1, kp2, m, n, k)
          command1 = "/tools/cuda/cuda-11.0.1/bin/ncu --metrics \"regex:.*\" -k volta_fp16_s884gemm_fp16_... -s 0 -c 1 '/tools/venvs/tensorflow/tensorflow-2.2.0/bin/python' dist_gemm_tf2_v9.py -kp1 {} -kp2 {} -N {} -t RC -m {} -n {} -k {} > {} 2>&1".format(kp1, kp2, N, m, n, k, output_file)
      #CR
      if t == "CR":
        kp1 = N
        kp2 = '-'
        m, n, k = dim
        output_file="{}/{}_N{}_K{}_K{}_m{}_n{}_k{}".format(output_dir, t, N, kp1, kp2, m, n, k)
        command1 = "/tools/cuda/cuda-11.0.1/bin/ncu --metrics \"regex:.*\" -k volta_fp16_s884gemm_fp16_... -s 0 -c 1 '/tools/venvs/tensorflow/tensorflow-2.2.0/bin/python' dist_gemm_tf2_v9.py -kp1 {} -N {} -t CR -m {} -n {} -k {} > {} 2>&1".format(kp1, N, m, n, k, output_file)
 

      command2 = "cat {} | grep \"sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active\"| awk {{'print $3'}}".format(output_file) #unit
      command3 = "cat {} | grep \"dram__throughput.avg.pct_of_peak_sustained_active\"| awk {{'print $3'}}".format(output_file) #unit
      command4 = "cat {} | grep lts__t_sectors.avg.pct_of_peak_sustained_active | awk {{'print $3'}}".format(output_file) #unit
      command5 = "cat {} | grep dram_read_bytes | grep sum | head -n 1 | awk {{'print $3'}}".format(output_file) #unit
      command6 = "cat {} | grep dram_write_bytes | grep sum | head -n 1 | awk {{'print $3'}}".format(output_file) #unit
      command7 = "cat {} | grep lts__t_bytes.sum | head -n 1 | awk {{'print $3'}}".format(output_file) #unit
      command8 = "cat {} | grep sm__sass_thread_inst_executed_op_fp16_pred_on.sum | head -n 1 | awk {{'print $3'}}".format(output_file) #unit
      command9 = "cat {} | grep sm__sass_thread_inst_executed_ops_fadd_fmul_ffma_pred_on.sum | head -n 1 | awk {{'print $3'}}".format(output_file) #unit
      
      result = {'ncu':-1, 'core_util':-1, 'dram_util':-1, 
                'l2_util':-1, 'dram_read':-1, 'dram_write':-1, 
                'l2_access':-1, 'fp16_inst':-1, 'fma_inst':-1}
      
      run_command(command1, 'ncu', result)
      run_command(command2, 'core_util', result)
      run_command(command3, 'dram_util', result)
      run_command(command4, 'l2_util', result)
      run_command(command5, 'dram_read', result)
      run_command(command6, 'dram_write', result)
      run_command(command7, 'l2_access', result)
      run_command(command8, 'fp16_inst', result)
      run_command(command9, 'fma_inst', result)
      
      with open(result_file, "a+") as f:
        f.write("{0:s} N{1:d} {2:d} {3:d} {4:d} {5:d} {6:d} {7:.2f} {8:.2f} {9:.2f} {10:,} {11:,} {12:,} {13:,} {14:,}\n".format(t, N, kp1, kp2, m, n, k, result['core_util'], result['dram_util'], result['l2_util'], result['dram_read'], result['dram_write'], result['l2_access'], int(result['fp16_inst']), int(result['fma_inst'])))
      print("{0:s} N{1:d} {2:d} {3:d} {4:d} {5:d} {6:d} {7:.2f} {8:.2f} {9:.2f} {10:,} {11:,} {12:,} {13:,} {14:,}\n".format(t, N, kp1, kp2, m, n, k, result['core_util'], result['dram_util'], result['l2_util'], result['dram_read'], result['dram_write'], result['l2_access'], int(result['fp16_inst']), int(result['fma_inst'])))
