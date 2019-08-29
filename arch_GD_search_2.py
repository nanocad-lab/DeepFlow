import os
import shutil
import re
import collections
import random
import numpy as np

import config

import ruamel as _ruamel
import ruamel.yaml as _yaml

exp_root = os.getcwd() + '/hw_architecture/membw_batch'

class GradientDescentSearch:
    def __init__(self, _exp_root):
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
        
        self.top_level_params = {}
        self.top_level_params['data_scale'] = 100
        self.top_level_params['batch_size'] = 32
        self.exp_root = _exp_root

        self.search_params = {}
        self.search_params = self.parameters

        self.random_starts = 1
        self.debug = False

    def populate_config(self, params, config_dict):
        
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
       
        for param in self.top_level_params:
            try:
                config_dict['model_param'][param] = self.top_level_params[param]
            except:
                config_dict['model_param'] = {}
                config_dict['model_param'][param] = self.top_level_params[param]

        if self.debug:
            for param_class in params:
                print(config_dict[param_class], sum(params[param_class].values()))

    def collect_time(self, params):
        exp_dir_inputs_p = []
        for item in params:
            exp_dir_inputs_p.append(item + '-')
            #for i in params[item]:
                #exp_dir_inputs_p.append(i + str(params[item][i]))
        exp_dir_inputs_p = ['test']
        exp_dir_inputs = [self.exp_root, 'data_scale'+str(self.top_level_params['data_scale']), '_'.join(exp_dir_inputs_p)]
        exp_dir = '/'.join(exp_dir_inputs)
        
        try:
            shutil.rmtree(exp_dir)
        except:
            pass
            #print("Directory not there to be removed, Moving on ....")

        os.makedirs(exp_dir)
        #if(self.debug):
        #    print("Created directory at", exp_dir)

        os.system("cp configs/exp_config.yaml "+ exp_dir+"/exp_config.yaml")

        filename = 'configs/exp_config.yaml'
        with open(filename, "r") as f:
            config_dict = _yaml.load(f, Loader=_ruamel.yaml.Loader)

        #print(config_dict)
        self.populate_config(params, config_dict)
        with open(exp_dir+'/exp_config.yaml', 'w') as yaml_file:
            _yaml.dump(config_dict, yaml_file, default_flow_style=False)

        ##Area Finder
        #os.system

        ##Performance Finder
        os.system("python perf.py --exp_config " + exp_dir+"/exp_config.yaml --debug False >| " + exp_dir + "/summary.txt")

        ##Time Limit to compute a step
        os.system("bash search_scripts/time_limit.sh " + str(self.top_level_params['data_scale']) + " " + str(self.top_level_params['batch_size']) + ' | grep "time_per_step" >> ' + exp_dir+"/summary.txt")

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

        return exec_time, time_limit, found

    def multi_start_search(self):

        params_list = []
        exec_time_list = [] #np.zeros(self.random_starts)

        for i in range(self.random_starts):
            for param_class in self.search_params:
                random_params = random.sample(range(1,100),len(self.search_params[param_class]))
                tot = sum(random_params)
                random_params = [x/tot for x in random_params]
                for i, param in enumerate(self.search_params[param_class]):
                    self.search_params[param_class][param] = random_params[i]

            tdp_breakdown = self.do_GDSearch(self.search_params)

            params_list.append(tdp_breakdown[0])
            exec_time_list.append(tdp_breakdown[1])
        
        index = exec_time_list.index(min(exec_time_list))
        return params_list[index], exec_time_list[index]
        

    def batch_size_sweep(self,batch_sizes):
        
        init_parameters = self.search_params.copy()
        for batch_size in batch_sizes:
            self.top_level_params['batch_size'] = batch_size
            #self.parameters['tot_batch_size'] = batch_size
            self.search_params = init_parameters.copy()
            tdp_breakdown = self.multi_start_search()
            print('For Batch-Size: %d, the TDP breakdown is as follows: ' %(batch_size), tdp_breakdown)
    
    def do_GDSearch(self, search_params):
       
        saturated = False
        min_value_params = 0
        prev_exec_time = 1000000000
        alpha = 1.1 #size of perturbation(should be >1)
        beta_init = 0.005 #step size
        while (saturated == False):
            gradient_list = {}
            beta = {}
            old_params = search_params.copy()
            exec_time = self.collect_time(search_params)[0]
            if self.debug:
                print("Execution time %4f with TDP parameters" %(exec_time), search_params)
            temp_params = search_params.copy()

            for param_class in search_params:
                gradient_list[param_class] = {}
                for param in search_params[param_class]:
                    temp_params[param_class][param] = search_params[param_class][param]*alpha #increase each param by alpha
                    if self.debug:
                        print("PARAM_CLASS: {} PARAM: {}, search_param_value: {}".format((param_class),(param),(temp_params[param_class][param])))
                    new_exec_time = self.collect_time(temp_params)[0]
                    gradient = (exec_time-new_exec_time)/((alpha-1)*search_params[param_class][param])
                    if self.debug:
                        print("Exec time improv =", gradient, exec_time, new_exec_time)
                    gradient_list[param_class][param] = gradient
                    temp_params = search_params.copy()

            for param_class in search_params:
                try:
                    beta[param_class] = beta_init*(sum(gradient_list[param_class].values())/(sum(x for i in gradient_list.values() for x in i.values())))
                except:
                    beta[param_class] = 0
            #sorted_improv = sorted(gradient_list.items(), key=lambda kv: kv[1], reverse=True)
            #gradient_list = collections.OrderedDict(sorted_improv)

                if self.debug:
                    print(gradient_list[param_class])

            frac_to_increase = {}
            for param_class in search_params:
                amount_to_increase = 0
                tot_gradient = sum(gradient_list[param_class].values())
            
                for param in search_params[param_class]:
                    if temp_params[param_class][param] > 0.01:
                        temp_params[param_class][param] = search_params[param_class][param] - beta[param_class]
                        amount_to_increase += beta[param_class]
                    if tot_gradient != 0:
                        frac_to_increase[param] = gradient_list[param_class][param]/tot_gradient
                    else:
                        frac_to_increase[param] = 0
                
                if self.debug:
                    print("Amount Increased: ", amount_to_increase)
                    print(search_params, temp_params, frac_to_increase)
           
                for param in search_params[param_class]:
                    search_params[param_class][param] = temp_params[param_class][param] + frac_to_increase[param]*amount_to_increase

            t = self.collect_time(search_params)
            #if t[2] == True:
            #    print ("Found arch to meet timing requirement")
            #    return search_params, exec_time

            new_exec_time = t[0]
            #print("New Execution Time: {}".format(new_exec_time))
            if (prev_exec_time - new_exec_time < 0.000005):
                saturated = True
                if t[2] == False:
                    print("Saturated but the architecture doesn't meet **Time Limit**")
            
            hyper_parameter_scaling = False
            
            if new_exec_time >= prev_exec_time:
                hyper_parameter_scaling = True
            else:
                prev_exec_time = new_exec_time

            if hyper_parameter_scaling:
                for param_class in search_params:
                    beta[param_class] = beta[param_class]*0.5
                    alpha = alpha*0.5
                search_params = old_params.copy()

            if saturated:
                if self.debug:
                    print('Saturated, done with search', search_params, exec_time)
                return search_params, exec_time
            

def main():

    GDS = GradientDescentSearch(exp_root)
    GDS.debug=True
    GDS.random_starts = 1
    
    batch_sizes = [32, 512]
    GDS.batch_size_sweep(batch_sizes)



if __name__ == "__main__":
    main()
