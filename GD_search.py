import click
import os
import shutil
import re
import collections
import random
import numpy as np
import copy
import sys

import config
import util
import perf

import ruamel as _ruamel
import ruamel.yaml as _yaml

import math

import subprocess
import re

#exp_root = os.getcwd() + '/hw_architecture/batch_sweep'

class GradientDescentSearch:
    def __init__(self, exp_dir, exp_config, debug, **kwargs):
        #Initial Values to start the GD search with
        self.parameters = {}
        #Starting TDP
        #Play with this number to start from a reasonable point for search
        #self.parameters['TDP'] = 300
        ###############
        self.parameters['area_breakdown'] = {}
        self.parameters['power_breakdown'] = {}
        self.parameters['perimeter_breakdown'] = {}
        self.parameters['area_breakdown']['core'] = 0.14
        self.parameters['area_breakdown']['DRAM'] = 0.14
        self.parameters['area_breakdown']['L2'] = 0.14
        self.parameters['area_breakdown']['L1'] = 0.14
        self.parameters['area_breakdown']['reg_mem'] = 0.14
        self.parameters['area_breakdown']['intra_node'] = (0.14 if int(kwargs.get('intra_derate', 1)) > 0 else 0)
        self.parameters['area_breakdown']['inter_node'] = (0.14 if int(kwargs.get('inter_derate', 1)) > 0 else 0)
        self.parameters['power_breakdown']['core'] = 0.14
        self.parameters['power_breakdown']['DRAM'] = 0.14
        self.parameters['power_breakdown']['L2'] = 0.14
        self.parameters['power_breakdown']['L1'] = 0.14
        self.parameters['power_breakdown']['reg_mem'] = 0.14
        self.parameters['power_breakdown']['intra_node'] = (0.14 if int(kwargs.get('intra_derate', 1)) > 0 else 0)
        self.parameters['power_breakdown']['inter_node'] = (0.14 if int(kwargs.get('inter_derate', 1)) > 0 else 0)
        self.parameters['perimeter_breakdown']['DRAM'] = 0.5
        self.parameters['perimeter_breakdown']['inter_node'] = (0.3 if int(kwargs.get('inter_derate', 1)) > 0 else 0)
        self.parameters['perimeter_breakdown']['intra_node'] = (0.2 if int(kwargs.get('intra_derate', 1)) > 0 else 0)
        
        self.model_level_params = {}
        self.model_level_params['data_scale'] = int(kwargs.get('data_scale', 1))
        self.model_level_params['batch_size'] = int(kwargs.get('batch_size', 256))
        self.model_level_params['layer_size'] = int(kwargs.get('hidden_dim', 19968))
        
        self.sch_level_params = {}
        self.sch_level_params['dp'] = int(kwargs.get('dp', 1))
        self.sch_level_params['lp'] = int(kwargs.get('lp', 1))
        
        self.kp_level_params = {}
        self.kp_level_params['kp_type'] = int(kwargs.get('kp_type', -1))
        self.kp_level_params['kp1'] = int(kwargs.get('kp1', 1))
        self.kp_level_params['kp2'] = int(kwargs.get('kp2', 1))

        self.sch_level_params['kp'] = (1 if self.kp_level_params['kp_type'] == -1 else 
                                       (self.kp_level_params['kp1'] if self.kp_level_params['kp_type'] == 1 else 
                                        self.kp_level_params['kp1'] * self.kp_level_params['kp2']))
        
        self.chip_area_budget = int(kwargs.get('chip_area_budget', -1))   

        self.system_hierarchy_params = {}
        self.system_hierarchy_params['num_workers'] = (self.sch_level_params['kp'] * 
                                                     self.sch_level_params['dp'] * 
                                                     self.sch_level_params['lp'])
        wafer_dim = int(kwargs.get('wafer_dim', 1))
        self.system_hierarchy_params['num_nodes_per_wafer'] = wafer_dim * wafer_dim  
        self.system_hierarchy_params['inter_derate'] = int(kwargs.get('inter_derate', 1))
        self.system_hierarchy_params['intra_derate'] = int(kwargs.get('intra_derate', 1))
        self.system_hierarchy_params['kp1_inter'] = bool(kwargs.get('kp1_inter', False))
        self.system_hierarchy_params['kp2_inter'] = bool(kwargs.get('kp2_inter', False))
        self.system_hierarchy_params['dp_inter'] = bool(kwargs.get('dp_inter', False))
        self.system_hierarchy_params['lp_inter'] = bool(kwargs.get('lp_inter', False))

        #Refine the search_parameters list to exclude excluded_params
        self.search_params = copy.deepcopy(self.parameters)
        #self.excluded = {'area_breakdown':['DRAM', 'inter_node'], 'power_breakdown':['inter_node'], 'perimeter_breakdown':['inter_node']}
        #self.excluded = {'area_breakdown':['DRAM']}
        self.excluded = {}
        if  self.system_hierarchy_params['inter_derate'] == 0:
            for param_class in self.parameters:
              if param_class in self.excluded:
                if 'inter_node' not in self.excluded[param_class]:
                  self.excluded[param_class].append('inter_node')
              else:
                self.excluded[param_class]=['inter_node']
        if  self.system_hierarchy_params['intra_derate'] == 0:
           for param_class in self.parameters:
             if param_class in self.excluded:
               if 'intra_node' not in self.excluded[param_class]:
                 self.excluded[param_class].append('intra_node')
             else:
               self.excluded[param_class]=['intra_node']
        

        #for c,p in zip(self.excluded_param_class, self.excluded_param):
        for c, plist in self.excluded.items():
          for p in plist:
            self.search_params[c].pop(p)

        self.index      = int(kwargs.get('index', 1))
        self.exp_dir    = exp_dir
        self.exp_config = exp_config

        self.debug      = debug
        self.best_time  = float('inf')
        self.lr = 1

        self.initialize()
    
    def printParams(self, params, message='', f=None):
        line=''
        for param_class in self.parameters:
            line = line + " " + param_class
            for param in self.parameters[param_class]:
                param_abv = param[0]
                if 'inter' in param:
                    param_abv = 'o'
                val = params[param_class][param] if (param_class in params and param in params[param_class]) else self.parameters[param_class][param]
                line = line + " " + param_abv + ": " + "{0:.3f}".format(val)
        if f is None:
            print(message + " " + line + "\n", flush=True)
        else:
            f.write(message + " " + line + "\n")

    def collect_time(self, params, iteration):
        exp_dir = self.create_config_dir(params, iteration)
        self.populate_config(params, exp_dir)
        ##Performance Finder
        #print("python3 perf.py --exp_config {exp_dir}/exp_config.yaml --exp_dir {exp_dir} --debug {debug}".format(exp_dir=exp_dir, debug=False))
        #self.printParams(params)
        #print("Config file: {}".format(exp_dir + "/exp_config.yaml"))
        #os.system("python3 perf.py --exp_config {exp_dir}/exp_config.yaml --exp_dir {exp_dir} --debug {debug}".format(exp_dir=exp_dir, debug=False))
        perf.callPerf(exp_config='{}/exp_config.yaml'.format(exp_dir), exp_dir=exp_dir, debug=False)

        ##Time Limit to compute a step
        #os.system("bash search_scripts/time_limit.sh " + str(self.model_level_params['data_scale']) + " " + str(self.model_level_params['batch_size']) + ' | grep "time_per_step" >> ' + exp_dir+"/summary.txt")

        exec_time = float("inf")
        time_limit = 1e15
        mem_overflow_rate = -1

        for line in open(exp_dir+'/summary.txt', 'r'):
            if re.search('Time:', line):
                exec_time = float(line.split(': ')[1])
            if re.search('time_per_step:', line):
                time_limit= float(line.split(': ')[1].split('\n')[0])
            if re.search('Throughput: ',line):
                core_throughput = line.split(': ')[1]
            if re.search('Memory Bandwidth: ',line):
                mem_bw = line.split(': ')[1]
            if re.search('Model Shard/',line):
                mem_overflow_rate = float(line.split(': ')[1])

        assert(mem_overflow_rate > 0)

        found = False
        if exec_time < time_limit:
            found = True

        if self.debug:
            print("Config file: {}".format(exp_dir + "/exp_config.yaml"))
            self.printParams(params)
            print("Time: {}\n".format(exec_time))
        
        #print("{}/summary.txt".format(exp_dir))

        return exec_time, time_limit, found, exp_dir, mem_overflow_rate
 
    def initialize(self):
        random_seed = self.index
        random.seed(random_seed)

        with open(self.exp_config, "r") as f:
            config_dict = _yaml.load(f, Loader=_ruamel.yaml.Loader)

        for param_class in self.parameters:
          if param_class == 'perimeter_breakdown': 
            for param in self.parameters[param_class]:
              self.parameters[param_class][param] = config_dict[param_class][param]
          else:
            for param in self.parameters[param_class]:
              if 'node' in param:
                self.parameters[param_class][param] = config_dict[param_class]['network'][param]
              else:
                self.parameters[param_class][param] = config_dict[param_class][param]


        print("Initializing...", flush=True)
        for param_class in self.search_params:
            random_params = random.sample(range(1,100),len(self.search_params[param_class]))
              
            expected_sum = 1
            if param_class in self.excluded:
              for param in self.excluded[param_class]:
                expected_sum = expected_sum - self.parameters[param_class][param]
            tot = float(sum(random_params)) 
            scale_factor = tot/expected_sum
            random_params = [x/scale_factor for x in random_params]
            for i, param in enumerate(self.search_params[param_class]):
              self.search_params[param_class][param] = random_params[i]
        
        self.printParams(self.search_params)
        
        iteration = 0
        t = self.collect_time(self.search_params, iteration)
        new_exec_time = t[0]
        time_limit = t[1]
        best_dir = t[3]
        self.best_time = new_exec_time
        self.best_dir = best_dir
        print("Step: {}, New_time: {}, Best_time: {}, lr: {}".format(iteration, new_exec_time, self.best_time,  self.lr), flush=True)

    def create_config_dir(self, params, iteration):
        exp_dir=[]
        exp_file='s' + str(iteration) + "_"
        
        exp_root = self.exp_dir
        exp_config = self.exp_config
        start_id = self.index

        for i, param_class in enumerate(params):
            class_abv = param_class.split('_')[0]
            exp_file = exp_file + ('' if i==0 else '_') + class_abv
            for param in params[param_class]:
                param_abv = param[0]
                if 'inter' in param:
                    param_abv = 'o'
                exp_file = exp_file + "_" + param_abv + "{0:.3f}".format(params[param_class][param])
        exp_dir = [exp_root, exp_file]
        exp_dir = '/'.join(exp_dir)
        
        try:
            shutil.rmtree(exp_dir)
        except:
            pass

        os.makedirs(exp_dir)
        #if(self.debug):
        #    print("Created directory at", exp_dir)
        return exp_dir


    def populate_config(self, params, exp_dir):
        exp_config  = self.exp_config #template to copy from
        config_file = exp_dir + "/exp_config.yaml" #config file to to run
       
        
        #print("cp " + exp_config + " " + config_file)
        os.system("cp " + exp_config + " " + config_file)
        
        config_dict = {}
        with open(exp_config, "r") as f:
            config_dict = _yaml.load(f, Loader=_ruamel.yaml.Loader)
        
        config_dict['area_breakdown']['proc_chip_area_budget'] = self.chip_area_budget if self.chip_area_budget != -1 else config_dict['area_breakdown']['proc_chip_area_budget']#in mm^2
        for param_class in params:
            for param in params[param_class]:
                if param_class != 'perimeter_breakdown' and ('node' in param):
                    try:
                        config_dict[param_class]['network'][param] = params[param_class][param]
                    except:
                        config_dict[param_class]['network'] ={}
                        config_dict[param_class]['network'][param] = params[param_class][param]
                else:
                    config_dict[param_class][param] = params[param_class][param]
      
        #model_param
        for param in self.model_level_params:
            try:
                config_dict['model_param'][param] = self.model_level_params[param]
            except:
                config_dict['model_param'] = {}
                config_dict['model_param'][param] = self.model_level_params[param]
        
        #scheduling_param
        for param in self.sch_level_params:
            if "kp" not in param:
                config_dict['scheduling_param'][param] = self.sch_level_params[param]

        kp_type = self.kp_level_params['kp_type']
        for param in config_dict['scheduling_param']:
            if 'kp' in param  and 'dim1' in param:
                config_dict['scheduling_param'][param] = (1 if kp_type == -1 
                                        else self.kp_level_params['kp1'])
            elif 'kp' in param  and 'dim2' in param:
                config_dict['scheduling_param'][param] = (1 if kp_type == -1 or kp_type == 1 else self.kp_level_params['kp2'])
            elif 'type' in param:
                config_dict['scheduling_param'][param] = kp_type

        #system_hierarchy_param
        for param in self.system_hierarchy_params:
            config_dict['system_hierarchy'][param] = self.system_hierarchy_params[param]
        
        with open(config_file, 'w') as yaml_file:
            _yaml.dump(config_dict, yaml_file, default_flow_style=False)
      
        #print(config_file)
        #if self.debug:
        #    for param_class in params:
        #        print("{} parameters sum up to {}".format(config_dict[param_class], sum(params[param_class].values())))

    def do_GDSearch(self):
        saturated = False
        min_value_params = 0
        alpha = 1.5 #size of perturbation(should be >1)
        lr = self.lr
        beta1 = 0.9
        beta2 = 0.999
        eps   = 1e-8
        iteration = 1
        best_time = self.best_time
        prev_exec_time = best_time
        prev_ckpt_time = best_time
        best_iteration = 0
        best_params = {}
        search_params = self.search_params
        #acc_grad = {}

        #for param_class in search_params:
        #    acc_grad[param_class] = {}
        #    for param in search_params[param_class]:
        #        acc_grad[param_class][param] = 0
        
        M = {}
        R = {}

        for param_class in search_params:
          M[param_class] = {}
          R[param_class] = {}
          for param in search_params[param_class]:
            M[param_class][param] = 0
            R[param_class][param] = 0


        best_dir = self.best_dir
        while (saturated == False):
            #print("\n")
            print("******************************************************************************Iteration: {}".format(iteration))
            #print("\n")

            random_seed = self.index
            random.seed(random_seed)
            
            gradient_list = {}
            beta = {}
            old_params = copy.deepcopy(search_params)
            exec_time_tuple = self.collect_time(search_params, iteration)
            exec_time       = exec_time_tuple[0]
            mem_overflow_rate = exec_time_tuple[4]

            #if self.debug:
            #    print("Execution time %4f with TDP parameters" %(exec_time), search_params)
            temp_params = copy.deepcopy(search_params)

            for param_class in search_params:
                gradient_list[param_class] = {}
                for param in search_params[param_class]:
                    temp_params[param_class][param] = search_params[param_class][param] * alpha #increase each param by alpha
                    #if self.debug:
                    #    print("PARAM_CLASS: {} PARAM: {}, search_param_value: {}".format((param_class),(param),(temp_params[param_class][param])))
                    new_exec_time = self.collect_time(temp_params, iteration)[0]
                    
                    if exec_time == float('Inf') and new_exec_time == float('Inf'):
                        gradient = 0
                    elif search_params[param_class][param] == 0:
                        gradient = 0
                    else:
                        #gradient = (exec_time - new_exec_time) / ((alpha - 1) * search_params[param_class][param])
                        gradient = (new_exec_time - exec_time) / ((alpha - 1) * search_params[param_class][param])
                    #if self.debug:
                    #    print("Exec time improv =", gradient, exec_time, new_exec_time)
                    gradient_list[param_class][param] = gradient
                    #acc_grad[param_class][param] += abs(gradient)
                    temp_params = copy.deepcopy(search_params)
                    #print("{:} {:}: {:,} -> {:,} , {}".format(param_class, param, exec_time, new_exec_time, gradient_list[param_class][param]))
                    #print()

            if self.debug:
                print("***grad_list: {}".format(gradient_list))
                #print("***acc_grad: {}".format(acc_grad))

            #for param_class in search_params:
            #  for param in search_params[param_class]:
            #    print("{:} {:} {:,}".format(param_class, param, gradient_list[param_class][param]))
              
            clip_max = 10
            for param_class in search_params:
              grad_norm = math.sqrt(np.sum([i**2 for i in gradient_list[param_class].values()]))
              for i, param in enumerate(search_params[param_class]):
                  #Adam optimizer
                  #M[param_class][param] = beta1 * M[param_class][param] + (1. - beta1) * gradient_list[param_class][param]
                  #R[param_class][param] = beta2 * R[param_class][param] + (1. - beta2) * gradient_list[param_class][param]**2

                  #m_hat = M[param_class][param] / (1. - beta1**(iteration))
                  #r_hat = R[param_class][param] / (1. - beta2**(iteration))
                  
                  #gradient_update = lr * m_hat / (np.sqrt(r_hat) + eps)
                  
                  gradient_update = (gradient_list[param_class][param] * lr / grad_norm) if grad_norm > 0 else (gradient_list[param_class][param] * lr)
                  
                  #gradient_clipped = (clip_max if gradient_update > clip_max else gradient_update)

                  #search_params[param_class][param] += gradient_clipped
                  search_params[param_class][param] = search_params[param_class][param] - gradient_update
                  #search_params[param_class][param] = search_params[param_class][param] - gradient_clipped
                  search_params[param_class][param] = search_params[param_class][param] if search_params[param_class][param] > 0 else 1e-2
                  #search_params[param_class][param] = search_params[param_class][param] if search_params[param_class][param] > 0 else 1e-1
                  M[param_class][param] = beta1 * M[param_class][param] + (1. - beta1) * search_params[param_class][param]
                  search_params[param_class][param] = M[param_class][param]


              if mem_overflow_rate > 1:
                mem_percentage = search_params[param_class]['DRAM']
                if 'area' in param_class: 
                    mem_percentage = search_params[param_class]['DRAM'] * mem_overflow_rate
                    print("Increasing DRAM area to meet the memory capacity requirement", flush=True)
                if mem_percentage > 1: 
                  saturated = True
                  print("Memory capacity is not sufficient to support the given model + parallelism strategy and there is not enough area on chip to support a larger controller", flush=True)
                else:
                  search_params[param_class]['DRAM'] = mem_percentage

              feat_vector = list(search_params[param_class].values())
              feat_sum = np.sum(feat_vector)
              
              expected_sum = 1
              if param_class in self.excluded:
                for param in self.excluded[param_class]:
                  expected_sum = expected_sum - self.parameters[param_class][param]
              
              #Only scale things down if their sum is above 100% otherwise it is ok to sum up to less than 100%
              if True: 
              #if feat_sum > 1:
                scale_factor = feat_sum / expected_sum
              else:
                scale_factor = 1
              for param in search_params[param_class]:
                  old_v =  search_params[param_class][param]
                  search_params[param_class][param] = search_params[param_class][param] if (scale_factor == 0) else search_params[param_class][param]/scale_factor
                  #adam_update = (old_params[param_class][param] - old_v)
                  #print("{:} {:}: {:,} -> {:,} -> {:,} , ({})".format(param_class, param, old_params[param_class][param], old_v, search_params[param_class][param], gradient_list[param_class][param]))
                  #print()
              
            self.printParams(old_params, "old_params")
            self.printParams(search_params, "new_params")
            t = self.collect_time(search_params, iteration)

            new_exec_time = t[0]
            time_limit = t[1]
            ratio = new_exec_time / prev_exec_time
            if new_exec_time < best_time:
                best_time = new_exec_time
                best_params = copy.deepcopy(search_params)
                best_dir = t[3]
                best_iteration = iteration
            
            #print("Step: {}, New_time: {}, Best_time: {}, Time_limit: {}".format(iteration, new_exec_time, best_time, time_limit))
            print("Step: {}, New_time: {}, Best_time: {}, lr: {}, bit: {}".format(iteration, new_exec_time, best_time, lr, best_iteration), flush=True)
            #curr_dir='{}/summary.txt'.format(t[3])
            #print("{}".format(curr_dir))     
           
            if iteration == 100 or new_exec_time == float('inf'):
              saturated = True
            #if iteration % 10 == 0 or new_exec_time == float('inf'):
            #    if ((prev_ckpt_time - new_exec_time < threshold) or (prev_ckpt_time==float('inf') and new_exec_time == float('inf'))):
            #        saturated = True
            #        if t[2] == True:
            #            print("Saturated. Best time: {}, Best architecture: {}".format(best_time, best_params))
            #        else:
            #            print("Saturated at {} but no architecture meets the **Time Limit**: {}".format(best_time, time_limit))
            #            print("Best architecture: {}".format(best_params))
            #    prev_ckpt_time = new_exec_time


            #if new_exec_time >= prev_exec_time:
            #    beta_factor = beta_factor * 0.5
            #    updated_alpha = (alpha - 1) * 0.8 + 1
            #    if updated_alpha > 1.001:
            #        alpha = updated_alpha
            #    search_params = copy.deepcopy(old_params)
            #    print("Previous move results in ascending direction. Try again with smaller perturabtions (alpha) and learning_rate (beta)")

            iteration = iteration + 1    
            prev_exec_time = new_exec_time
        
        best_config='{}/summary.txt'.format(best_dir)
        print("Best_config: {}".format(best_config), flush=True)     
        return best_params, best_time, time_limit, best_dir
    
def get_slurm_job_info():
    """Get information about the current job using `scontrol show job`.

    Returns a dict mapping parameter names (e.g. "UserId", "RunTime", etc) to
    their values, both as strings.
    """
    info = {}

    if "SLURM_JOB_ID" in os.environ:
      job_id  = int(os.environ["SLURM_JOB_ID"])

      command = ["scontrol", "show", "job", str(job_id)]
      output  = subprocess.check_output(command).decode("utf-8")

      # Use a regex to extract the parameter names and values
      pattern = "([A-Za-z/]*)=([^ \t\n]*)"
      info    = dict(re.findall(pattern, output))
    return info

@click.command("arch_search")        
@click.option("--exp_config", help="Path to experiment config", required=True)
@click.option("--exp_dir", help="Checkpoint/log directory", required=True)
@click.option("--debug", help="Debug", default=False, type=bool)
@click.option("--batch_size", help="Batch size", default=256)
@click.option("--hidden_dim", help="Dimension of Hidden Layer", default=19968)
@click.option("--data_scale", help="Data scale", default=1)
@click.option("--index", help="Search index", required=True)
@click.option("--dp", help="Number of data parallel workers", required=True)
@click.option("--lp", help="Number of layer parallel workers", default=1)
@click.option("--kp_type", help="Number of kernel parallel workers", default=-1)
@click.option("--kp1", help="Number of kernel parallel workers along input dimension in RC or inner dimesnion in CR", default=1)
@click.option("--kp2", help="Number of kernel parallel workers along output dimension in RC or should be 1 for CR", default=1)
@click.option("--inter_derate", help="derate factor for inter(cross)-wafer communication", default=1)
@click.option("--intra_derate", help="derate factor for intra(within)-wafer communication", default=1)
@click.option("--kp1_inter", help="Does parallelism along kp1 dimension cross the wafers?", default=False, type=bool)
@click.option("--kp2_inter", help="Does parallelism along kp2 dimension cross the wafers?", default=False, type=bool)
@click.option("--dp_inter", help="Does parallelism along dp dimension cross the wafers?", default=False, type=bool)
@click.option("--lp_inter", help="Does parallelism along lp dimension cross the wafers?", default=False, type=bool)
@click.option("--wafer_dim", help="wafer dimension (num of accelerator along x-axis, assuming wafer is square form)", default=1)
def main(exp_config, 
         exp_dir, 
         debug, 
         index, 
         batch_size,
         hidden_dim,
         data_scale, 
         dp, 
         lp, 
         kp_type, 
         kp1, 
         kp2, 
         inter_derate,
         intra_derate,
         kp1_inter,
         kp2_inter,
         dp_inter,
         lp_inter,
         wafer_dim):

    info = get_slurm_job_info()
    if ('NodeList' in info):
      print("JobId: {}".format(info['JobId']), flush=True)
      print("Node: {}".format(info['NodeList']), flush=True)
      print(subprocess.check_output('uptime').decode("utf-8"), flush=True)
      #command=["ps","-aux"]
      #print(subprocess.check_output(command).decode("utf-8"))
    
    chip_area_budget = util.getChipArea(exp_config, 
                                        batch_size=batch_size,
                                        hidden_dim=hidden_dim,
                                        dp=dp, 
                                        lp=lp, 
                                        kp_type=kp_type, 
                                        kp1=kp1, 
                                        kp2=kp2)
    
    if chip_area_budget < 0:
        print("Node area budget is not large enough to accomedate memory footprint, either increase node area budget or stack capacity") 
        return

    GDS = GradientDescentSearch(exp_dir=exp_dir, 
                                exp_config=exp_config, 
                                debug=debug, 
                                batch_size=batch_size,
                                hidden_dim=hidden_dim,
                                data_scale=data_scale,
                                dp=dp,
                                lp=lp,
                                kp_type=kp_type,
                                kp1=kp1,
                                kp2=kp2,
                                inter_derate=inter_derate,
                                intra_derate=intra_derate,
                                kp1_inter=kp1_inter,
                                kp2_inter=kp2_inter,
                                dp_inter=dp_inter,
                                lp_inter=lp_inter,
                                wafer_dim=wafer_dim,
                                chip_area_budget=chip_area_budget,
                                index=index)

    output_file = exp_dir + "/best.txt"
    best_params, best_time, time_limit, best_dir = GDS.do_GDSearch()
    with open(output_file, "a+") as f:
        f.write("Best Time: {}\n".format(best_time))
        f.write("Time Limit: {}\n".format(time_limit))
        f.write("Best Dir: {}\n".format(best_dir))
        GDS.printParams(best_params, f=f)
    
    try:
      output_dir = re.sub("/tmp","/mnt/scratch", exp_dir)
      shutil.copyfile(output_file, output_dir + "/best.txt")
      cmd=["cp","-r", best_dir, output_dir + "/best_dir"]
      subprocess.check_output(cmd).decode("utf-8")
      shutil.rmtree(exp_dir)
      if ('JobId' in info):
        job_id = info['JobId']
        slurm_dir = re.sub("/tmp","/mnt/home", exp_dir)
        slurm_output = "{}/slurm-{}.out".format(slurm_dir, job_id)
        shutil.copyfile(slurm_output, output_dir + "/slurm.txt")
    except:
      pass

    
    info = get_slurm_job_info()
    if ('RunTime' in info):
      print("Runtime: {}".format(info['RunTime']), flush=True)

if __name__ == "__main__":
    main()
