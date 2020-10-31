#!/tools/python/python3.8.3/bin/python
import os
import shutil
import subprocess
import numpy as np

batch_list=[i*1024 for i in range(2,7)] #[int(i) for i in (2**np.linspace(10,13,20)//2*2)]
seq_list=[10]
hidden_list=[i*1024 for i in range(2,7)] #[int(i) for i in (2**np.linspace(10,13,20)//2*2)]
vocab_list=[i*1024 for i in range(2,7)] #[int(i) for i in (2**np.linspace(10,13,20)//2*2)]
layer_list=[1]
bpe_list=[10]
epoch_list=[5]

output_dir="/mnt/home/newsha/baidu/developement/MechaFlow/validation/benchmarks/rnnlm/profile"
result_file="{}/result.csv".format(output_dir)

if os.path.exists(output_dir):
  shutil.rmtree(output_dir, ignore_errors=True)
os.makedirs(output_dir)
print("Created {}".format(output_dir))

with open(result_file, "w") as f:
  f.write("Batch Seq Hidden Vocab Layers Epoch BPE Time\n\n")
print("Batch Seq Hidden Vocab Layers Epoch BPE Time\n\n")

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
              command1 = "/tools/cuda/cuda-11.0.1/bin/nsys profile -t cuda,osrt,cudnn,cublas,nvtx,mpi -o profile/{} --stats=true -f true python lm-fp16-timehistory.py -b{} -s{} -d{} -v{} -l{} -p{} -e{} -m train -train data/test-index.txt -test data/test-index.txt -valid data/test-index.txt > {} 2>&1".format(fname, b, s, d, v, l, bpe, e, output_file)
              #command2 = "cat {} | grep step | tr '\' '\n' | grep step | tail -n 1 |awk {{'print $5'}} | sed 's#/step##'".format(output_file) #unit
              command2 = "cat {} | grep Avg | awk -v bpe={} {{'print $2/bpe'}}".format(output_file, bpe) #unit
              try:
                output1 = subprocess.check_output(
                            command1, stderr=subprocess.STDOUT, shell=True
                            ).decode("utf-8")
              except:
                continue

              try:
                output2 = subprocess.check_output(
                            command2, stderr=subprocess.STDOUT, shell=True
                            ).decode("utf-8")
                avg_time = output2.strip()
              except:
                avg_time = -1
                continue

              with open(result_file, "a+") as f:
                f.write("{} {} {} {} {} {} {} {}\n".format(b,s,d,v,l,e,bpe, avg_time))
              print("{} {} {} {} {} {} {} {}\n".format(b,s,d,v,l,e,bpe, avg_time))
