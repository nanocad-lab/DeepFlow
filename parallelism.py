import config
import util

class Parallelism():
    def __init__(self, exp_config):
        self.autoPar = exp_config.sch_config.auto
        self.lp    = exp_config.sch_config.lp
        self.kp_hidden_dim1    = exp_config.sch_config.kp_hidden_dim1
        self.kp_softmax_dim1   = exp_config.sch_config.kp_softmax_dim1
        self.kp_embedding_dim1 = exp_config.sch_config.kp_embedding_dim1
        self.kp_projection_dim1 = exp_config.sch_config.kp_projection_dim1
        self.kp_hidden_dim2    = exp_config.sch_config.kp_hidden_dim2
        self.kp_softmax_dim2   = exp_config.sch_config.kp_softmax_dim2
        self.kp_embedding_dim2 = exp_config.sch_config.kp_embedding_dim2
        self.kp_projection_dim2 = exp_config.sch_config.kp_projection_dim2
        self.dp    = exp_config.sch_config.dp
        self.kp_hidden_type  =  exp_config.sch_config.kp_hidden_type #1: CR, 2: RC
        self.kp_softmax_type  =  exp_config.sch_config.kp_softmax_type #1: CR, 2: RC
        self.kp_embedding_type  =  exp_config.sch_config.kp_embedding_type #1: CR, 2: RC
        self.kp_projection_type  =  exp_config.sch_config.kp_projection_type #1: CR, 2: RC
        self.exp_config = exp_config

    def findParallelStrategy(self):
        if (self.autoPar == None or self.autoPar == False):
            pass
        elif (self.autoPar == "greedy"):
            self.greedyScheduler()
        elif (self.autoPar=="dynamic"):#dynamic programming solution
            self.dynamic()
        else:
            print("Scheduling types supported are None/False/greedy/dynamic")
            return NotImplemented

    
    def dynamic(self):
        print("Dynamic Programming Scheduling Not Implemented yet!")
        return NotImplemented
    
    def greedyScheduler(self):
        #Find minimum number of parallel workers based on 
        #application's memory footprint and 
        #memory capacity per accelerator
        #Step 1. Try to fit everything on one GPU
        #Step 2. If it does not fit, check if individual items fit on one GPU,
        #if it does allocate a  seperate GPU per item
        #Step 3. If it does not, try layer parallelism across hidden layers
        #Step 4. If it does not, try kernel parallelism across hidden layers
        #Step 5. For softmax and embedding layers, try kernel parallelism
        tot_mem, embedding_mem, hidden_mem, softmax_mem, projection_mem, wt_mem, act_mem, point_mem = util.getTotMemReq(self.exp_config)
        #tot_mem, embedding_mem, hidden_mem, softmax_mem, projection_mem = util.getTotMemReq(self.exp_config)
        print("Value of M is {.1d}\n"
               .format(self.M))
        if (tot_mem < self.M):
            self.lp = 1
            self.kp_hidden_type = -1
            self.kp_softmax_type = -1
            self.kp_embedding_type = -1
            self.kp_projection_type = -1
        
        else:
            self.kp_hidden_dim1 = 1
            self.kp_hidden_dim2 = 1
            self.kp_softmax_dim1 = 1
            self.kp_softmax_dim2 = 1
            self.kp_projection_dim1 = 1
            self.kp_projection_dim2 = 1
            
            L = self.L
            lp = math.ceil(hidden_mem / M)
            self.lp = (L if lp > L else lp)
            self.kp_hidden     = (1 if lp <= L else math.ceil(hidden_mem / L / M))
            self.kp_softmax    = math.ceil(softmax_mem / M)
            self.kp_embedding  = math.ceil(embedding_mem / M)
            self.kp_projection = math.ceil(projection_mem / M)
            
            self.findlp()
            
            if (self.kp_hidden == 1):
                self.kp_hidden_type = -1
    
            if (self.kp_softmax == 1):
                self.kp_softmax_type = -1
    
            if (self.kp_projection == 1):
                self.kp_projection_type = -1
            
            if (self.kp_embedding == 1):
                self.kp_embedding_type = -1
           
            if self.kp_hidden_type == 1:
                self.kp_hidden_dim1 = self.kp_hidden
                self.kp_hidden_dim2 = 1
            elif self.kp_hidden_type == 2:
                #This is an arbiotrary choice until I have more insight
                self.kp_hidden_dim1 = self.findDiv(math.ceil(math.sqrt(self.kp_hidden)), self.miniB)
                self.kp_hidden_dim2 = self.kp_hidden_dim1 
             
            if self.kp_softmax_type == 1:
                self.kp_softmax_dim1 = self.kp_softmax
                self.kp_softmax_dim2 = 1
            elif self.kp_softmax_type == 2:
                #This is an arbitrary choice until I have more insight
                self.kp_softmax_dim1 = self.findDiv(math.ceil(math.sqrt(self.kp_softmax)), self.miniB * self.S)
                self.kp_softmax_dim2 = self.kp_softmax_dim1 
                
            if self.kp_projection_type == 1:
                self.kp_projection_dim1 = self.kp_projection
                self.kp_projection_dim2 = 1
            elif self.kp_projection_type == 2:
                #This is an arbitrary choice until I have more insight
                self.kp_projection_dim1 = self.findDiv(math.ceil(math.sqrt(self.kp_projection)), self.miniB * self.S)
                self.kp_projection_dim2 = self.kp_projection_dim1 
    
    
    #################Helper functions################
    def bag(self, bag):
          #Not implemented, backpacking stratgey
          return 1
    
    #Find how to use minimum number of layers for layer parallelism
    def findlp(self):
            tot_mem, embedding_mem, hidden_mem, softmax_mem, projection_mem = self.getTotMemReq()
            self.M = self.mem_size
    
            #If Kernel parallelism is one for all components then finding how to group
            #layers together is a knap-sack problem with some dependency constraints st.
            #neighbouring layers are bagged together
            if (self.lp == 1 and self.kp_hidden == 1 and self.kp_embedding == 1 and 
                self.kp_projection == 1 and self.kp_softmax == 1 and tot_mem > M):
                value = [embedding_mem, hidden_mem, projection_mem, softmax_mem]
                num_bags = self.bag(value)
                
            elif (self.lp > 1 and self.kp_hidden == 1 and self.kp_embedding == 1 and 
                  self.kp_projection == 1 and self.kp_softmax == 1 and tot_mem > M):
                  value1 = [embedding_mem, hidden_mem / self.lp] 
                  value2 = [hidden_mem / self.lp, projection_mem, softmax_mem] 
                  num_bags1 = self.bag(value1)
                  num_bags2 = self.bag(value2)
                  num_bags = num_bags1 + num_bags2
            self.lp = num_bags
    
    
    #Find minimum value between A and B that B is divisible by
    def findDiv(self, A, B):
            smallestFactor = -1
            for i in range(A,B+1):
              if (B % i == 0):
                  smallestFactor = i
                  break
            assert(smallestFactor != -1)
            return smallestFactor


