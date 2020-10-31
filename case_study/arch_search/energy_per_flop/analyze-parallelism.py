import os.path
import re
from os import listdir
import sys

root='/mnt/scratch/newsha/MechaFlow'
exp=sys.argv[1]
exp='scaleout-batch256-packSiIF-tech14-worker1024-dim8-w16-17:02:2020-21:02:53'
#exp='packaging-value_mom-100steps-batch256-packSiIF-tech14-worker64-wafer1-13:02:2020-16:42:23'
#exp='packaging-value_mom-100steps-batch256-packVolta-tech14-worker64-wafer64-18:02:2020-21:49:15'
#exp='chip_gran-batch256-packSiIF-tech14-worker4-dim2-w1-17:02:2020-21:02:53' 
exp_root=root + "/" + exp
#exp_root='/mnt/scratch/newsha/MechaFlow/batch256-packSiIF-tech14-worker64-03:01:2020-18:15:26'
index_dict={'area_core': 2, 'area_DRAM': 4, 'area_L2': 6, 'area_shared': 8, 'area_reg': 10, 'area_intra': 12, 'area_inter': 14, 
        'power_core': 17, 'power_DRAM': 19, 'power_L2': 21, 'power_shared': 23, 'power_reg': 25, 'power_intra': 27, 'power_inter': 29, 
        'perimeter_DRAM': 32, 'perimeter_intra': 34, 'perimeter_inter': 36}



def get_stats(parallel_strategy, layout, batch='b256', start_point=0):
  sums={'area_core': 0, 'area_DRAM': 0, 'area_L2': 0, 'area_shared': 0, 'area_reg': 0, 'area_intra': 0, 'area_inter': 0, 
        'power_core': 0, 'power_DRAM': 0, 'power_L2': 0, 'power_shared': 0, 'power_reg': 0, 'power_intra': 0, 'power_inter': 0, 
        'perimeter_DRAM': 0, 'perimeter_intra': 0, 'perimeter_inter': 0}

  exp_dir='{exp_root}/{parallel_strategy}/{layout}/{batch}/r{start_point}'.format(exp_root=exp_root, parallel_strategy=parallel_strategy, 
                                                                          layout=layout, batch=batch, start_point=start_point)
  config=''
  best_time=''
  best_file='{exp_dir}/best_dir/summary.txt'.format(exp_dir=exp_dir)
  config=''
  best_time=-1
  time_limit=0

  #print(best_file)

  if os.path.isfile(best_file):
    with open(best_file, 'r') as f:
      for line in f:
        if re.search('area_breakdown', line):
            config = line.strip('\n')
            conf_list = config.split()
            for key, val in index_dict.items():
              sums[key] = sums[key] + float(conf_list[val])
        if re.search('Time', line):
            best_time=float(line.split(': ')[1].strip('\n'))
  return best_time, sums
  #print('{0:2d} {1:.4f} {2:.4f} {3}'.format(i, best_time, time_limit, config))  
  
  #for key, val in sum.items():
  #  print('{0:16s}: {1:.2f}'.format(key, val/100))


best_time = float('inf')
best_config = ''
for parallel_strategy in listdir(exp_root):
  best_time = float('inf')
  time_limit=-1
  best_config = ''
  for layout in listdir(exp_root + '/' + parallel_strategy):
    for batch in listdir(exp_root + '/' + parallel_strategy + '/' + layout):
      for start_point in range(0, 10):
          time, _= get_stats(parallel_strategy, layout, batch, start_point)
          if time < best_time and time != -1:
            best_time = time
            best_config=(parallel_strategy, layout, batch, start_point)
  print(parallel_strategy, best_time, best_config)

#print(best_time, best_config)
