import os
import shutil
import re
import collections

exp_root = os.getcwd() + '/hw_architecture/membw_batch'

class GradientDescentSearch:
    def __init__(self, _exp_root):
        #Initial Values to start the GD search with
        self.parameters = {}
        #Starting TDP
        #Play with this number to start from a reasonable point for search
        #self.parameters['TDP'] = 300
        ###############
        self.parameters['core'] = 0.3
        self.parameters['DRAM'] = 0.3
        self.parameters['L2'] = 0.2
        self.parameters['shared_mem'] = 0.2
        self.parameters['IB'] = 0.0
        self.data_scale = 100
        self.parameters['tot_batch_size'] = 32
        self.exp_root = _exp_root

        search_params_list = ['core', 'DRAM', 'L2', 'shared_mem']

        self.search_params = {}
        for item in search_params_list:
            self.search_params[item] = self.parameters[item]

        self.debug = False

    def collect_time(self, params):
        exp_dir_inputs_p = []
        for i in params:
            exp_dir_inputs_p.append(i + str(params[i]))
        exp_dir_inputs = [self.exp_root, 'data_scale'+str(self.data_scale), '_'.join(exp_dir_inputs_p)]
        exp_dir = '/'.join(exp_dir_inputs)
        
        try:
            shutil.rmtree(exp_dir)
        except:
            pass
            #print("Directory not there to be removed, Moving on ....")

        os.makedirs(exp_dir)
        #if(self.debug):
        #    print("Created directory at", exp_dir)

        os.system("cp ../configs/exp_config.yaml "+ exp_dir+"/exp_config.yaml")

        for i in params:
            param_name = i
            if 'size' in i:
                param_value = str(params[i]) + " GB"
            elif 'bw' in i:
                param_value = str(params[i]) + " GB/s"
            else:
                param_value = str(params[i])
            command = ['sed -i "s/', param_name, ': .*/', param_name, ': ""', param_value, '""/" ', exp_dir+'/exp_config.yaml']
            command = "".join(command)
            os.system(command)

        ##Area Finder
        #os.system

        ##Performance Finder
        os.system("python ../perf.py --exp_config " + exp_dir+"/exp_config.yaml --debug False >| " + exp_dir + "/summary.txt")

        ##Time Limit to compute a step
        os.system("bash time_limit.sh " + str(self.data_scale) + " " + str(self.parameters['tot_batch_size']) + ' | grep "time_per_step" >> ' + exp_dir+"/summary.txt")

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


    def do_GDSearch(self, search_params):
       
        tot_params = len(search_params)
        saturated = False
        min_value_params = 0
        prev_exec_time = 1000000000
        while (saturated == False):
            alpha = 1.0002 #size of perturbation(should be >1)
            beta = 0.00005 #step size
            gradient_list = {}
            old_params = search_params.copy()
            exec_time = self.collect_time(search_params)[0]
            print("Execution time %4f with TDP parameters" %(exec_time), search_params)
            temp_params = search_params.copy()

            for param in search_params:
                temp_params[param] = search_params[param]*alpha #increase each param by alpha
                if self.debug:
                    print("param: {}, search_param_value: {}".format((param),(temp_params[param])) )
                new_exec_time = self.collect_time(temp_params)[0]
                gradient = (exec_time-new_exec_time)/((alpha-1)*search_params[param])
                if self.debug:
                    print("Exec time improv =", gradient, exec_time, new_exec_time)
                gradient_list[param] = gradient
                temp_params = search_params.copy()

            sorted_improv = sorted(gradient_list.items(), key=lambda kv: kv[1], reverse=True)
            gradient_list = collections.OrderedDict(sorted_improv)

            if self.debug:
                print(gradient_list)

            num_items_to_increase = 2
            count = 0
            amount_increased = 0

            for param in gradient_list:
                if (count == 0 and gradient_list[param] < 0):
                    saturated = True

                elif gradient_list[param] > 0:
                    new_param = search_params[param] + beta*gradient_list[param]  #*search_params[param]
                    if (new_param > 1.0):
                        new_param = 1.0
                    search_params[param] = new_param
                    amount_increased += beta*gradient_list[param] #*search_params[param]
                elif gradient_list[param] == 0:
                    print(param)
                count += 1
                if count >= num_items_to_increase:
                    break

            if self.debug:
                print("Amount Increased: ", amount_increased)

            num_items_to_decrease = 2
            count = 0
            amount_to_decrease = amount_increased/num_items_to_decrease

            if self.debug:
                print(amount_to_decrease)

            gradient_list = collections.OrderedDict(reversed(list(gradient_list.items())))

            if self.debug:
                print(gradient_list)

            for param in gradient_list:
                if (count == 0 and gradient_list[param] < 0):
                    new_param = search_params[param] - amount_increased
                    if (new_param < 0.01):
                        new_param = 0.01
                        amount_to_decrease = (amount_increased - search_params[param]+new_param)/(num_items_to_decrease - count + 1)
                    search_params[param] = new_param
                elif gradient_list[param] >= 0:
                    new_param = search_params[param] - amount_to_decrease
                    if (new_param < 0.01):
                        new_param = 0.01
                        amount_to_decrease += (amount_to_decrease - search_params[param] + new_param)/(num_items_to_decrease -count +1)
                    search_params[param] = new_param
                count += 1
                if count >= num_items_to_decrease:
                    break

            t = self.collect_time(search_params)
            if t[2] == True:
                print ("Found arch to meet timing requirement")
                return search_params, exec_time

            new_exec_time = t[0]
            if prev_exec_time - new_exec_time < 0.00005:
                saturated = True
            
            if new_exec_time >= prev_exec_time:
                search_params = old_params.copy()
                alpha = alpha*0.5
                beta = beta*0.5
            else:
                prev_exec_time = new_exec_time

            agg_TDP = 0
            for param in search_params:
                if (search_params[param] < 0.05):
                    min_value_params += 1
                agg_TDP += search_params[param]
            if agg_TDP > 1:
                beta = beta*0.5
                alpha = alpha*0.5
                search_params = old_params.copy()

            if saturated:
                print('Saturated, done with search', search_params, exec_time)
                return search_params, exec_time
            

def main():

    GDS = GradientDescentSearch(exp_root)
    GDS.debug=False
    tdp_breakdown = GDS.do_GDSearch(GDS.search_params)

    print('The TDP breakdown is as follows: ', tdp_breakdown)


if __name__ == "__main__":
    main()
