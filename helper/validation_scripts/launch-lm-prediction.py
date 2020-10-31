#!/tools/python/python3.8.3/bin/python
import os
import shutil
import subprocess

batch_list=[i*1024 for i in range(2,7)]
seq_list=[10]
hidden_list=[i*1024 for i in range(2,7)]
vocab_list=[i*1024 for i in range(2,7)]
layer_list=[1]
bpe_list=[10]
epoch_list=[3]

output_dir="/mnt/home/newsha/baidu/developement/MechaFlow/predictions/lm-core0.85-l2-0.45-h2d"

result_file="{}/result.csv".format(output_dir)

if os.path.exists(output_dir):
  print("path exists")
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
              dir_name = "{}/B{}-S{}-D{}-V{}-L{}-E{}-P{}".format(output_dir,b,s,d,v,l,e,bpe)
              if os.path.exists(dir_name):
                shutil.rmtree(dir_name, ignore_errors=True)
              os.makedirs(dir_name)
              debug_file = "{}/debug.txt".format(dir_name)
              command1 = "python perf.py --exp_config configs/v100.yaml --debug True --exp_dir {} --batch_size {} --seq_len {} --hidden_dim {} --vocab_size {} --num_layer {} | tail -n 2 > {} 2>&1".format(dir_name, b, s, d, v, l, debug_file)
              command2 = "cat {}/summary.txt | grep Time | awk '{{print $2}}'".format(dir_name)
              try:
                output1 = subprocess.check_output(
                            command1, stderr=subprocess.STDOUT, shell=True
                            ).decode("utf-8")
                print("output1")
              except:
                print("!!!!!")
                continue

              try:
                output2 = subprocess.check_output(
                            command2, stderr=subprocess.STDOUT, shell=True
                            ).decode("utf-8")
                avg_time = float(output2.strip())
                print(command2)
                print(output2)
                print(avg_time)
              except:
                print(command2)
                avg_time = -1
                continue

              with open(result_file, "a+") as f:
                f.write("{} {} {} {} {} {} {} {}\n".format(b,s,d,v,l,e,bpe, avg_time))
              print("{} {} {} {} {} {} {} {}\n".format(b,s,d,v,l,e,bpe, avg_time))
