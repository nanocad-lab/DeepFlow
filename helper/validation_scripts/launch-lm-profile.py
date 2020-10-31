#!/tools/python/python3.8.3/bin/python
import os
import shutil
import subprocess
import numpy as np

batch_list=[i*1024 for i in range(2,7)]
seq_list=[10]
hidden_list=[i*1024 for i in range(2,7)]
vocab_list=[2048] #[int(i) for i in (2**np.linspace(10,13,20)//2*2)]
layer_list=[1]
bpe_list=[10]
epoch_list=[3]



def run_command(cmd, var, result):
  try:
    output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True).decode("utf-8")
    output = output.strip().replace(',','')
    result[var] = float(output) if output != "" else output
  except:
    print("command for {} did not work".format(var))

output_dir="/mnt/home/newsha/baidu/developement/MechaFlow/validation/benchmarks/rnnlm/profile_gemm"
result_file="{}/result.csv".format(output_dir)

if os.path.exists(output_dir):
  shutil.rmtree(output_dir, ignore_errors=True)
os.makedirs(output_dir)
print("Created {}".format(output_dir))

with open(result_file, "w") as f:
  f.write("Batch Seq Hidden Vocab Layers Epoch BPE core_util dram_util l2_util dram_read dram_write l2_access fp16_inst fma_inst\n\n")
print("Batch Seq Hidden Vocab Layers Epoch BPE core_util dram_util l2_util dram_read dram_write l2_access fp16_inst fma_inst\n\n")

for b in batch_list:
  for s in seq_list:
    for d in hidden_list:
      for v in vocab_list:
        for l in layer_list:
          for bpe in bpe_list:
            for e in epoch_list:
              bpe = min(bpe, 25000//b)
              fname = "B{}-S{}-D{}-V{}-L{}-E{}-P{}".format(b,s,d,v,l,e,bpe)
              output_file = "{}/{}.out".format(output_dir, fname)

              command1="/tools/cuda/cuda-11.0.1/bin/ncu --metrics \"regex:.*\" -k volta_fp16_s884gemm_fp16_... -s 0 -c 1 '/tools/venvs/tensorflow/tensorflow-2.2.0/bin/python' lm-fp16.py -m train -train data/test-index.txt -test data/test-index.txt -valid data/test-index.txt -b{} -s{} -d{} -v{} -l{} -p{} -e{} > {} 2>&1".format(b, s, d, v, l, bpe, e, output_file)
              #command1 = "/tools/cuda/cuda-11.0.1/bin/nsys profile -t cuda,osrt,cudnn,cublas,nvtx,mpi -o profile/{} --stats=true -f true python lm-fp16.py -b{} -s{} -d{} -v{} -l{} -p{} -e{} -m train -train data/test-index.txt -test data/test-index.txt -valid data/test-index.txt > {} 2>&1".format(fname, b, s, d, v, l, bpe, e, output_file)
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
                f.write("{0:d} {1:d} {2:d} {3:d} {4:d} {5:d} {6:d} {7:.2f} {8:.2f} {9:.2f} {10:,} {11:,} {12:,} {13:,} {14:,}\n".format(b, s, d, v, l, e, bpe, result['core_util'], result['dram_util'], result['l2_util'], result['dram_read'], result['dram_write'], result['l2_access'], int(result['fp16_inst']), int(result['fma_inst'])))
              print("{0:d} {1:d} {2:d} {3:d} {4:d} {5:d} {6:d} {7:.2f} {8:.2f} {9:.2f} {10:,} {11:,} {12:,} {13:,} {14:,}\n".format(b, s, d, v, l, e, bpe, result['core_util'], result['dram_util'], result['l2_util'], result['dram_read'], result['dram_write'], result['l2_access'], int(result['fp16_inst']), int(result['fma_inst'])))
