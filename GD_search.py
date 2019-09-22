import click
import os
import shutil
import re
import collections
import random
import numpy as np
import copy

import config

import ruamel as _ruamel
import ruamel.yaml as _yaml

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
        self.parameters['area_breakdown']['shared_mem'] = 0.14
        self.parameters['area_breakdown']['reg_mem'] = 0.14
        self.parameters['area_breakdown']['intra_node'] = 0.14
        self.parameters['area_breakdown']['inter_node'] = 0.14
        self.parameters['power_breakdown']['core'] = 0.14
        self.parameters['power_breakdown']['DRAM'] = 0.14
        self.parameters['power_breakdown']['L2'] = 0.14
        self.parameters['power_breakdown']['shared_mem'] = 0.14
        self.parameters['power_breakdown']['reg_mem'] = 0.14
        self.parameters['power_breakdown']['intra_node'] = 0.14
        self.parameters['power_breakdown']['inter_node'] = 0.14
        self.parameters['perimeter_breakdown']['DRAM'] = 0.5
        self.parameters['perimeter_breakdown']['inter_node'] = 0.3
        self.parameters['perimeter_breakdown']['intra_node'] = 0.2
        
        self.model_level_params = {}
        self.model_level_params['data_scale'] = int(kwargs.get('data_scale', 1))
        self.model_level_params['batch_size'] = int(kwargs.get('batch_size', 32))
        
        self.sch_level_params = {}
        self.sch_level_params['dp'] = int(kwargs.get('dp', 1))
        self.sch_level_params['lp'] = int(kwargs.get('lp', 1))
        self.sch_level_params['kp'] = int(kwargs.get('kp', 1))

        self.search_params = {}
        self.search_params = self.parameters

        self.index      = int(kwargs.get('index', 1))
        self.exp_dir    = exp_dir
        self.exp_config = exp_config

        self.debug      = debug
        
        self.initialize()
    
    def printParams(self, params, message='', f=None):
        line=''
        for param_class in params:
            line = line + " " + param_class
            for param in params[param_class]:
                param_abv = param[0]
                if 'inter' in param:
                    param_abv = 'o'
                line = line + " " + param_abv + ": " + "{0:.3f}".format(params[param_class][param])
        if f is None:
            print(message + " " + line + "\n")
        else:
            f.write(message + " " + line + "\n")

    def collect_time(self, params, iteration):
        exp_dir = self.create_config_dir(params, iteration)
        self.populate_config(params, exp_dir)
        ##Performance Finder
        #print("python3 perf.py --exp_config {exp_dir}/exp_config.yaml --exp_dir {exp_dir} --debug {debug}".format(exp_dir=exp_dir, debug=False))
        #self.printParams(params)
        #print("Config file: {}".format(exp_dir + "/exp_config.yaml"))
        os.system("python3 perf.py --exp_config {exp_dir}/exp_config.yaml --exp_dir {exp_dir} --debug {debug}".format(exp_dir=exp_dir, debug=False))

        ##Time Limit to compute a step
        os.system("bash search_scripts/time_limit.sh " + str(self.model_level_params['data_scale']) + " " + str(self.model_level_params['batch_size']) + ' | grep "time_per_step" >> ' + exp_dir+"/summary.txt")

        exec_time = 100000000
        time_limit = 1e15

        for line in open(exp_dir+'/summary.txt', 'r'):
            if re.search('Time:', line):
                exec_time = float(line.split(': ')[1])
            if re.search('time_per_step:', line):
                time_limit= float(line.split(': ')[1].split('\n')[0])
            if re.search('Throughput: ',line):
                core_throughput = line.split(': ')[1]
            if re.search('Memory Bandwidth: ',line):
                mem_bw = line.split(': ')[1]

        found = False
        if exec_time < time_limit:
            found = True

        if self.debug:
            print("Config file: {}".format(exp_dir + "/exp_config.yaml"))
            self.printParams(params)
            print("Time: {}\n".format(exec_time))

        return exec_time, time_limit, found
 
    def initialize(self):
        random_seed = self.index
        random.seed(random_seed)
        print("Initializing...")
        for param_class in self.search_params:
            random_params = random.sample(range(1,100),len(self.search_params[param_class]))
            tot = float(sum(random_params))
            random_params = [x/tot for x in random_params]
            for i, param in enumerate(self.search_params[param_class]):
                self.search_params[param_class][param] = random_params[i]
        self.printParams(self.search_params)

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
       
        for param in self.model_level_params:
            try:
                config_dict['model_param'][param] = self.model_level_params[param]
            except:
                config_dict['model_param'] = {}
                config_dict['model_param'][param] = self.model_level_params[param]

        tot_nodes = 1
        for param in self.sch_level_params:
            #TODO: This needs to change when fixing kp, for now assuming kp is fixed
            tot_nodes *= self.sch_level_params[param]
            if "kp" not in param:
                config_dict['scheduling_param'][param] = self.sch_level_params[param]
            #else:
                #TODO: figure out which type of kp and how different layers are divided

        config_dict['system_hierarchy']['tot_nodes'] = tot_nodes
        
        with open(config_file, 'w') as yaml_file:
            _yaml.dump(config_dict, yaml_file, default_flow_style=False)
       
        #if self.debug:
        #    for param_class in params:
        #        print("{} parameters sum up to {}".format(config_dict[param_class], sum(params[param_class].values())))

    def do_GDSearch(self):
        saturated = False
        min_value_params = 0
        alpha = 1.1 #size of perturbation(should be >1)
        lr = 0.001
        iteration = 0
        best_time = float('inf')
        prev_exec_time = float('inf')
        prev_ckpt_time = float('inf')
        best_params = {}
        search_params = self.search_params

        while (saturated == False):
            #print("\n")
            print("******************************************************************************Iteration: {}".format(iteration))
            #print("\n")

            random_seed = self.index
            random.seed(random_seed)
            
            gradient_list = {}
            beta = {}
            old_params = copy.deepcopy(search_params)
            exec_time = self.collect_time(search_params, iteration)[0]
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
                    gradient = (exec_time - new_exec_time) / ((alpha - 1) * search_params[param_class][param])
                    #if self.debug:
                    #    print("Exec time improv =", gradient, exec_time, new_exec_time)
                    gradient_list[param_class][param] = gradient
                    temp_params = copy.deepcopy(search_params)

            if self.debug:
                print("Raw gradient_list: {}".format(gradient_list))

            for param_class in search_params:
              random_noise = np.random.normal(0, 1e-6, len(search_params[param_class])) 
              for i, param in enumerate(search_params[param_class]):
                  search_params[param_class][param] += gradient_list[param_class][param] * lr + random_noise[i]
                  search_params[param_class][param] = search_params[param_class][param] if search_params[param_class][param] > 0 else 0
              param_class_l2 = np.linalg.norm(list(search_params[param_class].values()), ord=1)
              for param in search_params[param_class]:
                  search_params[param_class][param] = search_params[param_class][param] if (param_class_l2 == 0) else search_params[param_class][param]/param_class_l2

              
            self.printParams(old_params, "old_params")
            self.printParams(search_params, "new_params")
            t = self.collect_time(search_params, iteration)

            new_exec_time = t[0]
            time_limit = t[1]
            
            if new_exec_time < best_time:
                best_time = new_exec_time
                best_params = copy.deepcopy(search_params)

            print("Step: {}, New_time: {}, Best_time: {}, Time_limit: {}".format(iteration, new_exec_time, best_time, time_limit))

            if iteration % 10 == 0:
                if (prev_ckpt_time - new_exec_time < 0.0001):
                    saturated = True
                    if t[2] == True:
                        print("Saturated. Best time: {}, Best architecture: {}".format(best_time, best_params))
                    else:
                        print("Saturated at {} but no architecture meets the **Time Limit**: {}".format(new_exec_time, time_limit))
                prev_ckpt_time = new_exec_time


            #if new_exec_time >= prev_exec_time:
            #    beta_factor = beta_factor * 0.5
            #    updated_alpha = (alpha - 1) * 0.8 + 1
            #    if updated_alpha > 1.001:
            #        alpha = updated_alpha
            #    search_params = copy.deepcopy(old_params)
            #    print("Previous move results in ascending direction. Try again with smaller perturabtions (alpha) and learning_rate (beta)")

            iteration = iteration + 1    
            prev_exec_time = new_exec_time
            
        return best_params, best_time, time_limit
    
   
@click.command("arch_search")        
@click.option("--exp_config", help="Path to experiment config", required=True)
@click.option("--exp_dir", help="Checkpoint/log directory", required=True)
@click.option("--debug", help="Debug", default=False, type=bool)
@click.option("--batch_size", help="Batch size", default=32)
@click.option("--data_scale", help="Data scale", default=1)
@click.option("--index", help="Search index", required=True)
@click.option("--dp", help="Number of data parallel workers", required=True)
@click.option("--lp", help="Number of layer parallel workers", required=True)
@click.option("--kp", help="Number of kernel parallel workers", required=True)
def main(exp_config, exp_dir, debug, index, batch_size, data_scale, dp, lp, kp):
    GDS = GradientDescentSearch(exp_dir=exp_dir, 
                                exp_config=exp_config, 
                                debug=debug, 
                                batch_size=batch_size, 
                                data_scale=data_scale,
                                dp=dp,
                                lp=lp,
                                kp=kp,
                                index=index)

    best_params, best_time, time_limit = GDS.do_GDSearch()
    output_file = exp_dir + "/best.txt"
    with open(output_file, "a+") as f:
        f.write("Best Time: {}\n".format(best_time))
        f.write("Time Limit: {}\n".format(time_limit))
        GDS.printParams(best_params, f=f)

if __name__ == "__main__":
    main()
