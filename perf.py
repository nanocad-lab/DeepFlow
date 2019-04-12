#!/tools/lm-venv/py3.6-tf-1.3.0-svail/bin/python

import click
import math

import os
import sys
import config

th_scale=0.7
mem_scale=0.7
algByte=False #algorithmic ops false
proj=False #consider projection layer, turn off for validation
mem_latency=50e-9 #50 nano sec
tensorflow_overhead=750e-6 #0.75 u-seconds

G=1 #1024.0*1024.0*1024.0

class TimeCalculation:
    def __init__(self, exp_config):
        #Model Parameters
        self.B = exp_config.model_config.batch_size
        self.V = exp_config.model_config.vocab_size
        self.L = exp_config.model_config.num_layers
        self.D = exp_config.model_config.layer_size
        self.projection = exp_config.model_config.projection
        self.S = exp_config.model_config.seq_len
        self.G = exp_config.model_config.num_gates
        self.NL = exp_config.model_config.num_non_linear
        self.A = exp_config.model_config.num_add
        self.P = self.NL + self.A
        
        #Architecture Parameters
        self.O         = exp_config.arch_config.kernel_launch_overhead
        self.precision = exp_config.arch_config.precision
       
        #Technology Parameters
        self.dram_energy_per_byte         = exp_config.tech_config.DRAM_energy_per_bit_trans * 8
        self.L2_energy_per_byte           = exp_config.tech_config.L2_energy_per_bit * 8
        self.shared_mem_energy_per_byte   = exp_config.tech_config.shared_mem_energy_per_bit * 8
        self.energy_per_flop              = exp_config.tech_config.core_energy_per_flop
        self.internode_energy_per_byte    = exp_config.tech_config.internode_energy_per_bit * 8
        self.ll                           = exp_config.tech_config.line_latency

        #Scheduling Parameters
        self.lp    = exp_config.sch_config.lp
        self.hlp    = exp_config.sch_config.hlp
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

        #Power Breakdown Parameters
        self.TDP       = exp_config.power_breakdown.TDP
        self.DRAMPower = exp_config.power_breakdown.DRAM * self.TDP
        self.HBM_stack_bw  = exp_config.tech_config.HBM_stack_bw
        self.HBM_stack_capacity = exp_config.tech_config.HBM_stack_capacity
        self.mem_bw = self.DRAMPower / self.dram_energy_per_byte * mem_scale
        self.mem_size = (self.mem_bw / mem_scale / self.HBM_stack_bw) * self.HBM_stack_capacity

        #FIXME: Do we anymore need this?
        setPar = False
        if (self.lp is not None and self.hlp is not None and 
            self.kp_hidden_dim1 is not None and self.kp_hidden_dim2 is not None and 
            self.kp_softmax_dim1 is not None and self.kp_softmax_dim2 is not None and
            self.kp_embedding_dim1 is not None and self.kp_embedding_dim2 is not None and 
            self.dp is not None and 
            self.kp_hidden_type is not None and 
            self.kp_softmax_type is not None and
            self.kp_embedding_type is not None):
            setPar = True
        
        self.miniB = math.ceil(self.B / self.dp)
        self.setParallelism(setPar)

        #Calculate architecture configurations based on power breakdown
        #and technology parameters
        self.corePower = exp_config.power_breakdown.core * self.TDP
        self.L2Power   = exp_config.power_breakdown.L2 * self.TDP
        self.shared_mem_power = exp_config.power_breakdown.shared_mem * self.TDP
        self.IBPower  = exp_config.power_breakdown.IB * self.TDP
        self.th     = self.corePower / self.energy_per_flop * th_scale
        
        self.L2_bank_bw  = exp_config.tech_config.L2_bank_bw
        self.L2_bank_capacity = exp_config.tech_config.L2_bank_capacity
        self.L2_bw    = (self.L2Power / self.L2_energy_per_byte)
        self.L2_size  = (self.L2_bw / self.L2_bank_bw) * self.L2_bank_capacity 
      
        self.shared_mem_bank_bw = exp_config.tech_config.shared_mem_bank_bw
        self.shared_mem_bank_capacity = exp_config.tech_config.shared_mem_bank_capacity
        self.shared_mem_bw = (self.shared_mem_power / self.shared_mem_energy_per_byte)
        self.shared_mem_size = (self.shared_mem_bw / self.shared_mem_bank_bw) * self.shared_mem_bank_capacity
       
        print("Shared_Mem_Size: {} MB".format(self.shared_mem_size/(1024 * 1024)))

        #TODO: Do we anymore need this?
        #Assuming Homogenous Cores, i.e. if we have 
        #3 links out of one core within one layers, 
        #we have 3 links out of every core regardless of if that layers need it.
        grid_dim = 0

        if self.dp > 1:
          grid_dim += 1
        
        if self.lp > 1:
          grid_dim += 1

        h1 = self.kp_hidden_dim1
        h2 = self.kp_hidden_dim2
        
        s1 = self.kp_softmax_dim1
        s2 = self.kp_softmax_dim2
      
        p1 = self.kp_projection_dim1
        p2 = self.kp_projection_dim2

        e1 = self.kp_embedding_dim1
        e2 = self.kp_embedding_dim2

        if(h1 > 1 or s1 > 1 or p1 > 1 or e1 > 1 or h2 > 1 or s2 > 1 or p2 > 1 or e2 > 1):
            grid_dim += 1
       
        print("grid_dim: {}".format(grid_dim))

        #FIXME: hack! Fix after ISCA deadline.
        num_links = 1
        if (grid_dim == 1):
            num_links = self.dp * h1
        elif (grid_dim == 2):
            num_links = 2 * self.dp * h1
        else:
            num_links = 3 * self.dp * h1 * self.lp
        
        print("num_links: {}".format(num_links))

        self.IB = (0 if grid_dim == 0 else self.IBPower / (num_links * self.internode_energy_per_byte)) 
        #NOTE:Assuming power across all links are similar
        self.IBK = self.IB
        self.IBM = self.IB
        self.IBD = self.IB

        self.attached = True
        self.debug = False
        self.tot_time = 0
        self.L2_tile_dim = 0
        self.sm_tile_dim = 0
        self.setTileDim()
    
        self.tot_flop = 0
        self.tot_mem = 0

    def tot_par(self):
        embedding = self.V * self.D
        hidden = (2 * self.D + 1) * (self.G * self.D) * self.L
        projection = self.D * self.projection
        softmax = ((self.projection if proj else self.D) + 1) * self.V

        tot_par = embedding + hidden + projection + softmax
        return tot_par


    def setParallelism(self, ParallelParams):
      print("----kp_hidden_type: {}, kp_softmax_type: {}, kp_projection_type: {}," 
             "kp_embedding_type: {}\n".format(self.kp_hidden_type, 
              self.kp_softmax_type, self.kp_projection_type, self.kp_embedding_type))
      
      M = self.mem_size
      tot_mem, embedding_mem, hidden_mem, softmax_mem, projection_mem = self.getTotMemReq()
      
      print("Overall Memory Requirement per Data Shard:")
      print("req_mem: {}, embedding: {}, hidden: {}, softmax: {}, projection: {}".format(tot_mem/1e9, embedding_mem/1e9, hidden_mem/1e9, softmax_mem/1e9, projection_mem/1e9))
      print("mem_size: {} GB, ratio: {}\n".format(M/(1024 * 1024 * 1024), tot_mem/M))
      
      if (ParallelParams): #and tot_mem < M):
        #TODO: ensure everyone fits in memory by given arguments
        print("Manual Mode\n")

      
      else: 
        print("Finding best parallelism parameters")
        #Step 1. Try to fit everything on one GPU
        #Step 2. If it does not fit, check if individual items fit in one GPU,
        #if it does allocate a  seperate GPU per item
        #Step 3. If it does not fit, try layer parallelism across hidden layers
        #Step 4. If it does not fit, try kernel parallelism across hidden layers
        #Step 5. For softmax and embedding try kernel parallelism
        tot_mem, embedding_mem, hidden_mem, softmax_mem, projection_mem = self.getTotMemReq()

        if (tot_mem < M):
            self.lp = 1
            self.hlp = 1
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
            hlp = math.ceil(hidden_mem / M)
            self.hlp = (L if hlp > L else hlp)
            self.kp_hidden     = (1 if hlp <= L else math.ceil(hidden_mem / L / M))
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

      print("kp_hidden_type: {}, kp_softmax_type: {}, kp_projection_type: {}," 
             "kp_embedding_type: {}\n".format(self.kp_hidden_type, 
              self.kp_softmax_type, self.kp_projection_type, self.kp_embedding_type))

      print("dp: {}, lp: {}, hlp: {}, kp_hidden_dim1: {}, kp_hidden_dim2: {}," 
             "kp_softmax_dim1: {}, kp_softmax_dim2: {}, kp_embedding: {}," 
             "kp_projection_dim1: {}, kp_proejction_dim2: {}\n".format(self.dp,
            self.lp, self.hlp, self.kp_hidden_dim1, self.kp_hidden_dim2, 
            self.kp_softmax_dim1, self.kp_softmax_dim2, self.kp_embedding_dim1, 
            self.kp_projection_dim1, self.kp_projection_dim2))   

      embedding_mem, hidden_mem, softmax_mem, projection_mem = self.getMemUsagePerCore()
      print("Memory Usage per Core:")
      print("embedding: {}, hidden: {}, softmax: {}, projection: {}".
            format(embedding_mem/(1024*1024*1024), hidden_mem/(1024*1024*1024), softmax_mem/(1024*1024*1024), 
            projection_mem/(1024*1024*1024)))
      #assert(tot_mem < M)
   
    def bag(self, bag):
      return 1

    def findlp(self):
        tot_mem, embedding_mem, hidden_mem, softmax_mem, projection_mem = self.getTotMemReq()
        M = self.mem_size

        #If Kernel parallelism is one for all components then finding how to group
        #layers together is a knap-sack problem with some dependency constraints st.
        #neighbouring layers are bagged together
        if (self.hlp == 1 and self.kp_hidden == 1 and self.kp_embedding == 1 and 
            self.kp_projection == 1 and self.kp_softmax == 1 and tot_mem > M):
            value = [embedding_mem, hidden_mem, projection_mem, softmax_mem]
            num_bags = self.bag(value)
            
        elif (self.hlp > 1 and self.kp_hidden == 1 and self.kp_embedding == 1 and 
              self.kp_projection == 1 and self.kp_softmax == 1 and tot_mem > M):
              value1 = [embedding_mem, hidden_mem / self.hlp] 
              value2 = [hidden_mem / self.hlp, projection_mem, softmax_mem] 
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

    def setTileDim(self):
        self.L2_tile_dim = 0
        self.shared_tile_dim = 0

        if (self.L2_size > 0):
            self.L2_tile_dim = math.ceil(math.pow(2, math.floor(math.log(math.sqrt((self.L2_size / self.precision) / 3), 2))))
        if (self.shared_mem_size > 0):
            self.sm_tile_dim = math.ceil(math.pow(2, math.floor(math.log(math.sqrt((self.shared_mem_size / self.precision) / 3), 2))))
        print("L2Tile_dim: {}, SMTile_dim: {}".format(self.L2_tile_dim, self.sm_tile_dim))
    
    def getTileDim(self):
        #assert(self.L2_tile_dim != 0)
        #assert(self.sm_tile_dim != 0)
        return self.L2_tile_dim, self.sm_tile_dim

    def roofline(self, flop, gmem, l2mem=0):

        self.tot_flop += flop
        self.tot_mem  += gmem

        inflection_point = self.th / self.mem_bw
        comp_int = flop / gmem
        #L2_tile_dim, sm_tile_dim = self.getTileDim()
        #single_block_comp_time = (2 * L2_tile_dim * L2_tile_dim * L2_tile_dim) / self.th
        #single_block_mem_time  = (3 * L2_tile_dim * L2_tile_dim * self.precision) / self.mem_bw
      
        time  = 0
        if comp_int < inflection_point: #mem-bound
            time = (gmem / self.mem_bw) + mem_latency + (0 if self.L2_bw == 0 else (l2mem / self.L2_bw))
        else: #compute-bound
            time = (flop / self.th)
        
        if self.debug:
            print("inflection_point: {:.2f}".format(inflection_point))
            print("comp_int: {:.2f}".format(comp_int))
        #    print("flop: {}".format(flop))
        #    print("l2mem: {}".format(l2mem))
        #    print("time: {}".format(time))
        
        return time

    def getGEMMTime(self, A, B, C, name):
        GEMM_flop, GEMM_gmem, GEMM_l2mem = self.GEMM(A, B, C)
        GEMM_flop_t, GEMM_gmem_t, GEMM_l2mem_t = self.GEMM(C, B, A)
        
        GEMM_time = self.roofline(GEMM_flop, GEMM_gmem, GEMM_l2mem) + self.O
        GEMM_time_t = self.roofline(GEMM_flop_t, GEMM_gmem_t, GEMM_l2mem_t) + self.O
        
        transpose = False
        time = GEMM_time
        if (GEMM_time > GEMM_time_t):
            transpose = True
            time = GEMM_time_t


        if self.debug:
            if transpose:
                print("{} GEMM_flop_t: {:,}, GEMM_gmem_t: {:,}".format(name, int(GEMM_flop_t),int(GEMM_gmem_t)))
                print("{} GEMM_time_t: {:,}\n".format(name, GEMM_time))
            else:
                print("{} GEMM_flop: {:,}, GEMM_gmem: {:,}, GEMM_l2mem: {:,}".format(name, int(GEMM_flop),int(GEMM_gmem), int(GEMM_l2mem)))
                print("{} GEMM_time: {:,}\n".format(name, GEMM_time))

        return transpose, time

    #This is the main function that captures the memory hierarchy impact
    #on the number of accesses to global memory considering not everything fits in 
    #L2 cache and also captures the effect of shared memory
    def GEMM(self, A, B, C):
        GEMM_flop = 2 * A * B * C
        #2 for multiplly and add
        X1, X2 = self.getTileDim()
      
        if (algByte):
            GEMM_gmem = (A * B + B * C + A * C) * self.precision
            GEMM_l2mem = 0

        else:
             #Here we are assuming tiles are in square form
             #Global memory accesses going through L2
             reload_AB = 1
             reload_BC = 1
             reload_AC = 1

             if X1 > 0 :
                 if  B <= X1:
                     reload_AB = 1
                     reload_BC = math.ceil(A / X1)
                     reload_AC = 1
                 else:
                     reload_AB = math.ceil(C / X1)
                     reload_BC = math.ceil(A / X1)
                     reload_AC = 1
             
             GEMM_gmem = (A * B * reload_AB + B * C * reload_BC + A * C * reload_AC) * self.precision
             if self.debug:
                print("gmem: reload_AB: {}, reload_AC: {}, reload_BC: {}\n", reload_AB, reload_AC, reload_BC)

             #L2memory accesses going through Shared memory
             reload_AB = 1
             reload_BC = 1
             reload_AC = 1
            
             if X2 > 0:
                 if  B <= X2:
                     reload_AB = 1
                     reload_BC = math.ceil(A / X2)
                     reload_AC = 1
                 else:
                     reload_AB = math.ceil(C / X2)
                     reload_BC = math.ceil(A / X2)
                     reload_AC = 1
             
             GEMM_l2mem = (A * B * reload_AB + B * C * reload_BC + A * C * reload_AC) * self.precision
             GEMM_flop = GEMM_flop + A * C * (15 + 5 * math.ceil(B / X2))
             if self.debug:
                print("l2mem: reload_AB: {}, reload_AC: {}, reload_BC: {}\n", reload_AB, reload_AC, reload_BC)
    
        return GEMM_flop, GEMM_gmem, GEMM_l2mem

    #Column-Row MM
    def getCf_kp1(self):
        #Multiply
        assert(self.kp_hidden_type == 1)
        assert(self.kp_hidden_dim1 > 1)
        assert(self.kp_hidden_dim1 % 4 == 0 or self.kp_hidden_dim1 == 2) #4 bc it is LSTM cell
        assert((2 * self.D) % self.kp_hidden_dim1 == 0)
        GEMM_time, reduction_time = self.getDistGEMM_f_kp1(self.miniB, 2 * self.D, self.G * self.D, self.kp_hidden_dim1, "Cf_kp1")
        
       
        #Pointwise ops: all the linear/non-linear ops after MM
        point_flop = self.miniB * (self.G * self.D / self.kp_hidden_dim1) * 4
        #4 refers to the number of pointwise ops (mul + add +tanh + mul) on 
        #the critical path 
        point_mem  = (self.precision * self.miniB * (self.G * self.D / self.kp_hidden_dim1) *
                     (3 * 3 + 2 * 1 ))
        # 3(3 memory access per operation with two input and one output)
        # 3(mul +  add + mul) on critical path
        # 2(2 memory access per operation with one input and one output)
        # 1(tanh) on critical path
        point_comm =  self.miniB * (self.G * self.D / self.kp_hidden_dim1) * 3 / self.IBK
        # 3 refers to the number of pointwise ops (mul + add + mul) on the
        # critical path whose inputs are located across different GPUs
        #NOTE:Assuming all communications can happpen in parallel

        point_time = self.roofline(point_flop, point_mem) + self.O + point_comm


        return GEMM_time + reduction_time + point_time
 
    def getCb_kp1(self):
        #TODO:Add local accumulation of weights at every time step
        #Pointwise
        point_flop = (self.miniB) * (self.G * self.D / self.kp_hidden_dim1) * 4
        #4 refers to the number of pointwise ops (mul + add +tanh + mul) on 
        #the critical path 
        point_mem  = (self.precision * self.miniB * 
                                       (self.G * self.D / self.kp_hidden_dim1) *
                                       (3 * 3 + 2 * 1))
        # 3(3 memory access per operation with two input and one output)
        # 3(mul +  add + mul) on critical path
        # 2(2 memory access per operation with one input and one output)
        # 1(tanh) on critical path
   
        point_comm =  self.miniB * (self.G * self.D / self.kp_hidden_dim1) * 3 / self.IBK
        #3 refers to the number of pointwise ops (mul + tanh + mul) on
        # critical path whose inputs are located across different GPUs
        #NOTE:Assuming all communications can happpen in parallel


        point_time = self.roofline(point_flop, point_mem) + self.O + point_comm


        GEMM_time, reduction_time = self.getDistGEMM_b_kp1(self. miniB, 2 * self.D, self.G * self.D, self.kp_hidden_dim1, "Cb_kp1") 

        if self.debug:
            print("(gr) Hidden point_flop: {:,}, point_mem: {:,}\n".format(int(point_flop/G), int(point_mem/G)))

        return GEMM_time + reduction_time + point_time

    #Row-Column MM
    def getCf_kp2(self):
        #Multiply
        assert(self.kp_hidden_type == 2)
        assert(self.kp_hidden_dim1 > 1 or self.kp_hidden_dim2 > 1)
        assert(self.kp_hidden_dim2 % self.G == 0 or self.kp_hidden_dim2 == 2 or self.kp_hidden_dim2 == 1)
        assert(self.miniB % self.kp_hidden_dim1 == 0)
        assert(self.G * self.D % self.kp_hidden_dim2 == 0)

        GEMM_time, reduction_time = self.getDistGEMM_f_kp2(self.miniB, 2 * self.D, self.G * self.D, self.kp_hidden_dim1,self.kp_hidden_dim2, "Cf_kp2")
        
        #Pointwise ops
        point_flop = (self.miniB/self.kp_hidden_dim1) * (self.G * self.D / self.kp_hidden_dim2) * 4
        #4 refers to the number of pointwise ops (mul + add +tanh + mul) on 
        #the critical path 
        point_mem  = (self.precision * (self.miniB / self.kp_hidden_dim1) * 
                     (self.G * self.D / self.kp_hidden_dim2) *
                     (3 * 3 + 2 * 1 ))
        # 3(3 memory access per operation with two input and one output)
        # 3(mul +  add + mul) on critical path
        # 2(2 memory access per operation with one input and one output)
        # 1(tanh) on critical path
        point_comm =  ((self.miniB / self.kp_hidden_dim1) * 
                       (self.G * self.D / self.kp_hidden_dim2) * 3 / self.IBK)
        #3 refers to the number of pointwise ops (mul + add + mul) whose inputs
        #across different GPU

        point_time = self.roofline(point_flop, point_mem) + self.O + point_comm

        
        return GEMM_time + reduction_time + point_time
    
    def getCb_kp2(self):
      
        #Pointwise ops
        point_flop = (self.miniB / self.kp_hidden_dim1) * (self.G * self.D / self.kp_hidden_dim2) * 4
        #4 refers to the number of pointwise ops (mul + add +tanh + mul) on 
        #the critical path 
        # kp_hidden_dim2 is for the reduction sum operation after doing outer product
        # for (B,4D)x(4D,2D).This is outerproduct due to the data distribution.
        point_mem  = (self.precision * (self.miniB / self.kp_hidden_dim1) * 
                                       (self.G * self.D / self.kp_hidden_dim2) *
                                       (3 * 3 + 2 * 1))
        # 3(3 memory access per operation with two input and one output)
        # 3(mul +  add + mul) on critical path
        # 2(2 memory access per operation with one input and one output)
        # 1(tanh) on critical path
   
        point_comm =  self.miniB * (self.G * self.D / self.kp_hidden_dim2) * 3 / self.IBK
        #3 refers to the number of pointwise ops (mul + add +tanh + mul) on 
        #3 refers to the number of hops to gather i,f, o and c in each GPU
        #in order to perform (B,4D)x(4D,2D)

        point_time = self.roofline(point_flop, point_mem) + self.O + point_comm


         
        GEMM_time, reduction_time = self.getDistGEMM_b_kp2(self.miniB, 2 * self.D, self.G * self.D, self.kp_hidden_dim1, self.kp_hidden_dim2, "Cb_kp2")

      
        if self.debug:
            print("(gr) Hidden point_flop: {:,}, point_mem: {:,}\n".format(int(point_flop/G), int(point_mem/G)))

        return GEMM_time + reduction_time + point_time



    def getCf(self):
        #Add Biad adds
        """Get LSTM Cell Time on Forward Path"""
        transpose, GEMM_time = self.getGEMMTime(self.miniB, 2 * self.D, self.G * self.D, "Cf")

        
        point_flop = self.miniB * self.D * self.G * (1 + 5/4 + 1)
        #1: add bias
        #5/4: add nonlinearities, there is one more than the number of gates (self.G)
        #1: pointwise muliply and add
        point_mem  = (self.precision * self.miniB * self.D * self.G *
                     (3 * 2 + 2 * 5/4 ))
        #3: 3 memory accesses for operands with two inputs and one output
        #2: 1 for bias add + 1 for pointwise mul
        #2: 2 memory accesses for operands with one input and one output
        #1: 5/4 non-linearities per gate

        point_time = self.roofline(point_flop, point_mem) + 15 * self.O

  
        if self.debug:
          print("Hidden point_flop: {:,}, point_mem: {:,}\n".format(int(point_flop/G), int(point_mem/G)))
          print("Hidden point_time: {:,}\n".format(point_time))

        return GEMM_time + point_time


   
    def getCb(self):
        """Get LSTM Cell Time on Backward Path"""
        grad_act_transpose, grad_act_time = self.getGEMMTime(self. miniB, self.G * self.D, 2 * self.D, "Cb_act") 
        grad_wt_transpose, grad_wt_time   = self.getGEMMTime(2 * self.D, self.miniB, self.G * self.D, "Cb_wt")

        
        GEMM_time = grad_act_time + grad_wt_time

        point_flop = ((self.miniB * self.D * (1 + 5/4 + 1)) + 
                     (2 * self.D * self.G * self.D)) # local accumulation of wts
        point_mem  = ((self.precision * self.miniB * self.D * (3 * 2 + 2 * 5/4)) + 
                     (2 * self.D * self.G * self.D) * 3) #local accumulation of wts
        #TODO: does the local accumulation needs a factor of two for wts and acc????
        point_time = self.roofline(point_flop, point_mem) + 15 * self.O
   
        if self.debug:
            print("(gr) Hidden point_flop: {:,}, point_mem: {:,}".format(int(point_flop/G), int(point_mem/G)))
            print("Hidden point_time: {:,}\n".format(point_time))

        return GEMM_time + point_time



    def getR(self, Dim0 = None, Dim1 = None, p = None, ib = None, partial = None,
            allReduce = None):
        """Get partail or full reduction or allGather latency"""
        """Partial reduction means each gpu is only collecting a shard of 
        reduced data"""
        """allReduce= False measures allGather latency otherwise allReduce"""
        if (Dim0 == None):
            Dim0 = 2 * self.D
        if (Dim1 == None):
            Dim1 = self.G * self.D
        if (p == None):
            p = self.dp
        if (ib == None): 
            ib = self.IBD
        if (partial == None): 
            partial = False
        if (allReduce == None):
            allReduce = True
        if (p == 1):
            return 0
        #If small data transfers, just broadcast
        threshold = 128
        data_tranfer = 0
        data_prep = 0

        #FIXME: Here I assumed point-2-point links exist across all nodes
        #Implement brodcast timing under ring topology
        if (self.precision * Dim0 * Dim1 < threshold):
            factor = (1/p if partial else 1)
            data_transfer = (self.precision * Dim0 * Dim1 * factor if p > 1 else 0)
            data_prep_comp = Dim0 * Dim1 * (p-1) * factor
            data_prep_mem = (3 * self.precision * Dim0 * Dim1) * (p - 1) * factor
            data_prep = self.roofline(data_prep_comp, data_prep_mem)
        else:
            #Assuming point-2-point link between consecutive data partitions
            #In other words, the network topology assumed is Ring, 
            #therefore all (dp-1) transfers can happen in parallel,
            #To assume different toplogy data_transfer formulation should change
            #e.g. assuming bus, data_transfer formulation would change as follows:
            #data_transfer  = ((self.precision * self.D * self.D) * (self.dp /self.dp)) * 
            #                 (self.G * 2) * (2 * (self.dp - 1))) / self.IBD
            factor = (1 if partial or not allReduce else 2)
            print("!!!!!!!!!!!!!!!!!!!!!!****************factor: {}".format(factor))
            data_transfer  = ((((self.precision * Dim0 * Dim1) / p) / ib) + self.ll) * factor * (p - 1)
            dt=((self.precision * Dim0 * Dim1) / p) * factor * (p - 1)
            
            data_prep_comp = (Dim0 * Dim1) / p
            data_prep_mem  = (3 * self.precision * Dim0 * Dim1 / p) 
            data_prep = ((self.roofline(data_prep_comp, data_prep_mem) + self.O)
                          * (p - 1))

            print("R1: {}, factor: {}\n".format(dt,factor))
        if self.debug:
            print("(gr) allReduce_flop: {:,}, allReduce_mem: {:,}".format(int(data_prep_comp/G), int(data_prep_mem/G)))

        return data_transfer + (data_prep if allReduce else 0)
    
    def gradClipping(self, Dim0 = None, Dim1 = None, name = None):
        if (Dim0 == None):
            Dim0 = 2 * self.D
        if (Dim1 == None):
            Dim1 = self.G * self.D
        if (name == None):
            name = "Hidden"
         #t_list[i] * clip_norm / max(global_norm, clip_norm)
         #where:
         #global_norm = sqrt(sum([l2norm(t)**2 for t in t_list]))
          
        norm_comp = Dim0 * Dim1 * 2
        #1: power 2
        #1: summ
        norm_mem = (Dim0 * Dim1 * 1) * self.precision
        #1: one read per element and power it by 2 in local registers  anfd 
        #summing to local acc

      
        clip_comp = Dim0 * Dim1 * 2
        #1: pointwise mul
        #1: pointwise div

        clip_mem = (Dim0 * Dim1 * 2) * self.precision
        #1: one read for pointwise mul
        #1: one write for pointwise div

        gradclip_mem = norm_mem + clip_mem
        gradclip_comp = norm_comp + clip_comp

        gradclip_time = self.roofline(gradclip_comp, gradclip_mem)

        if self.debug:
            print("({}) gradclip_flop: {:,}, gradclip_mem: {:,}".format(name, gradclip_comp, gradclip_mem))
            print("({}) gradclip_time: {:,}\n".format(name, gradclip_time))

        return gradclip_time


    def applyGrad(self, Dim0 = None, Dim1 = None, name = None):
        if (Dim0 == None):
            Dim0 = 2 * self.D
        if (Dim1 == None):
            Dim1 = self.G * self.D
        if (name == None):
            name = "Hidden"
        
        applyGrad_comp = Dim0 * Dim1 * 3
        #3: one pointwise division by  scalar after reducing all the gradients, 
        #   one final addition of gradients to the weights
        #   one multiply by learning rate
        applyGrad_mem = ((2 * Dim0 * Dim1 * self.precision) +
                         (3 * Dim0 * Dim1 * self.precision) +
                         (2 * Dim0 * Dim1 * self.precision))
        #2:read and write for pointiwse div
        #3: 2 reads and one write for pointwise add
        #2: 1 reads and one write for multiplication by lr
        applyGrad_time = self.roofline(applyGrad_comp, applyGrad_mem)
       
        clip_time = self.gradClipping(Dim0, Dim1, name)
        
        grad_time = applyGrad_time + clip_time

        if self.debug:
            print("({}) applyGrad_flop: {:,}, applyGrad_mem: {:,}".format(name, applyGrad_comp, applyGrad_mem))
            print("({}) applyGrad_time: {:,}\n".format(name, applyGrad_time))
       
        return grad_time

    def getDistGEMM_f_kp1(self, m, k, n, dim1, name):
        transpose, GEMM_time = self.getGEMMTime(m , k/dim1, n , name)
        
        #Sum-Reduce within each row for use in the next time step
        reduction_time = self.getR(Dim0 = m, 
                                   Dim1 = n,
                                   p = dim1,
                                   ib = self.IBK,
                                   partial = True,
                                   allReduce = True)
        return GEMM_time, reduction_time


    def getDistGEMM_b_kp1(self, m, k, n, dim1, name):
        #All-gather gradients wrt activations (BS,V) in each GPU node before MM
        reduction_time = self.getR(Dim0 = m, 
                                   Dim1 = n,
                                   p = dim1,
                                   ib = self.IBK,
                                   partial = False,
                                   allReduce = False)

        #Multiply full grad_activation with shards of weights
        grad_wt_transpose,  grad_wt_time  = self.getGEMMTime(k / dim1, m, n, name + "wt")
        #Multiply full grad-activation with shards of activations 
        grad_act_transpose, grad_act_time = self.getGEMMTime(m, n, k / dim1, name + "act")


        GEMM_time = grad_wt_time + grad_act_time + reduction_time

        return GEMM_time, reduction_time

    def getDistGEMM_f_kp2(self, m, k, n, dim1, dim2, name):
        transpose, GEMM_time = self.getGEMMTime(m / dim1, k, n / dim2, name)
        #All gather activations within each row for use in the next time step
        reduction_time = self.getR(Dim0 = m / dim1, 
                                   Dim1 = n,
                                   p = dim2,
                                   ib = self.IBK,
                                   partial = False,
                                   allReduce = False)
        return GEMM_time, reduction_time

    def getDistGEMM_b_kp2(self, m, k, n, dim1, dim2, name):
        #To calculate grad(wt), partially gather transpose(activations) across rows before MM
        reduction_time_wt1 = self.getR(Dim0 = k, 
                                       Dim1 = m,
                                       p = dim1,
                                       ib = self.IBK,
                                       partial = True,
                                       allReduce = False)
        #To calculate grad(wt), gather output matrix (A') across row within one column before MM
        reduction_time_wt2 = self.getR(Dim0 = m, 
                                       Dim1 = n / dim2,
                                       p = dim1,
                                       ib = self.IBK,
                                       partial = False,
                                       allReduce = False)

        #To calculate grad(wt), gather weght matrix (W) across row within one column after MM
        reduction_time_wt3 = self.getR(Dim0 = m, 
                                       Dim1 = n / dim2,
                                       p = dim1,
                                       ib = self.IBK,
                                       partial = False,
                                       allReduce = False)
        ########################################################################################
         #To calculate grad(act), gather output matrix (A') across columns within one row before MM
        reduction_time_act1 = self.getR(Dim0 = m / dim1, 
                                       Dim1 = n,
                                       p = dim2,
                                       ib = self.IBK,
                                       partial = False,
                                       allReduce = False)

        #To calculate grad(act), partially gather transpose(weight) across rows before MM
        reduction_time_act2 = self.getR(Dim0 = k, 
                                       Dim1 = n,
                                       p = dim2,
                                       ib = self.IBK,
                                       partial = True,
                                       allReduce = False)
               #To calculate grad(wt), partally gather weght matrix (W) across row within one column after MM
        reduction_time_act3 = self.getR(Dim0 = m / dim1, 
                                       Dim1 = k,
                                       p = dim2,
                                       ib = self.IBK,
                                       partial = False,
                                       allReduce = False)

        reduction_time = reduction_time_wt1 + reduction_time_wt2 + reduction_time_wt3 + reduction_time_act1 +reduction_time_act2 +reduction_time_act3

        #Multiply full grad_activation with shards of weights
        grad_wt_transpose,  grad_wt_time  = self.getGEMMTime(k / dim1, m, n / dim2, name + "wt")
        #Multiply full grad-activation with shards of activations 
        grad_act_transpose, grad_act_time = self.getGEMMTime(m / dim1, n, k / dim2, name + "act")

        GEMM_time = grad_wt_time + grad_act_time

        return GEMM_time, reduction_time
   
    def getProjection_f(self):
        transpose, GEMM_time = self.getGEMMTime(self.miniB * self.S, self.D, self.projection, "projection")
        return GEMM_time

    def getProjection_b(self):
        grad_wt_transpose,  grad_wt_time  = self.getGEMMTime(self.projection, self.miniB * self.S, self.D, "projection_b_wt")
        grad_act_transpose, grad_act_time = self.getGEMMTime(self.miniB * self.S, self.projection, self.D, "projection_b_act")

        GEMM_time = grad_wt_time + grad_act_time
        return GEMM_time

    def getProjection_f_kp1(self):
        assert(self.kp_projection_type == 1)
        assert(self.kp_projection_dim1 > 1)
        assert(self.D % self.kp_projection_dim1 == 0)

        GEMM_time, reduction_time = self.getDistGEMM_f_kp1(self.miniB * self.S, self.D, self.projection, self.kp_projection_dim1, "projection_f")
        return GEMM_time + reduction_time

    def getProjection_b_kp1(self):
        assert(self.kp_projection_type == 1)
        assert(self.kp_projection_dim1 > 1)
        assert(self.D % self.kp_projection_dim1 == 0)
        
        GEMM_time, reduction_time = self.getDistGEMM_b_kp1(self.miniB * self.S, self.D, self.projection, self.kp_projection_dim1, "projection_b")
        return GEMM_time + reduction_time

    def getProjection_f_kp2(self):
        assert(self.kp_projection_type == 2)
        assert(self.kp_projection_dim1 > 1 or self.kp_projection_dim2 > 1)
        assert((self.miniB * self.S) % self.kp_projection_dim1 == 0)
        assert(self.projection % self.kp_projection_dim2 == 0)

        GEMM_time, reduction_time = self.getDistGEMM_f_kp2(self.miniB * self.S, self.D, self.projection, self.kp_projection_dim1, self.kp_projection_dim2, "projection_f")
        return GEMM_time + reduction_time

    def getProjection_b_kp2(self):
        assert(self.kp_projection_type == 2)
        assert(self.kp_projection_dim1 > 1 or self.kp_projection_dim2 > 1)
        assert((self.miniB * self.S) % self.kp_projection_dim1 == 0)
        assert(self.projection % self.kp_projection_dim2 == 0)
        
        GEMM_time, reduction_time = self.getDistGEMM_f_kp2(self.miniB * self.S, self.D, self.projection, self.kp_projection_dim1, self.kp_projection_dim2, "projection_b")
        return GEMM_time + reduction_time

    def getSoftmax_f(self):
        transpose, GEMM_time = self.getGEMMTime(self.miniB * self.S, (self.projection if proj else self.D), self.V, "softmax_f")

        #Final matrix after GEMM has (B X S, V) dimensionality
        #We get exponential on each of the elements in a row
        #and then reduce all elements in the row to one
        #Therefore for each row we do V sigmoids and V-1 additions and V divisions
        #For each row, we read V sigmoids and write one
        point_flop = (self.miniB * self.S * (3 * self.V - 1))
        
        point_mem  = self.precision * self.miniB * self.S * (7 * self.V)
        #2: one read and one write for sigmoid
        #1: one read for reduction
        #1: one write for extension
        #2: for pointwise division

        point_time = self.roofline(point_flop, point_mem) + 4 * self.O 
   
        if self.debug:
            print("Softmax point_flop: {:,}, point_mem: {:,}".format(int(point_flop/G), int(point_mem/G)))
            print("point_time: {:,}\n".format( point_time))

        return GEMM_time + point_time

    #FIXME: where is the reduction time?
    def getSoftmax_b(self):

        grad_wt_transpose,  grad_wt_time  = self.getGEMMTime((self.projection if proj else self.D), self.miniB * self.S, self.V, "softmax_b_wt")
        grad_act_transpose, grad_act_time = self.getGEMMTime(self.miniB * self.S, self.V, (self.projection if proj else self.D), "softmax_b_act")

        GEMM_time = grad_wt_time + grad_act_time
        point_flop = (self.miniB * self.S) * (self.V) * 5
        #1: one for one of the divisions, grad(A) (y=A/B)
        #2: one for division and multiplication, grad(B)
        #1: one for addition, copies turn into add
        #1: one for sigmoid

        point_mem  = self.precision * (self.miniB * self.S) * self.V * 11
        #3: grad(A) in pointwise division
        #3: grad(B) in pointwise division
        #3: addition in copy backprop
        #2: sigmoid

        point_time = self.roofline(point_flop, point_mem) + 4 * self.O 
    
        if self.debug:
            print("(gr) Softmax point_flop: {:,}, point_mem: {:,}".format(int(point_flop/G), int(point_mem/G)))
            print("(gr) Softmax point_time: {:,}\n".format(point_time))

        dt=(self.precision * self.miniB * self.D * self.S)*2

        return GEMM_time + point_time

    #Column-Row MM
    def getSoftmax_f_kp1(self):
        #Multiply
        assert(self.kp_softmax_type == 1)
        assert(self.kp_softmax_dim1 > 1)
        assert((self.projection if proj else self.D) % self.kp_softmax_dim1 == 0)

        GEMM_time, reduction_time = self.getDistGEMM_f_kp1(self.miniB * self.S, self.projection, self.V, self.kp_softmax_dim1, "softmax_f_kp1")

        #Final matrix after GEMM has (B X S, V) dimensionality
        #We get exponential on each of the elements in a row
        #and then reduce all elements in the row to one
        #Therefore for each row we do V sigmoids and V-1 additions and V divisions
        #For each row, we read V sigmoids and write one
        
        point_flop = (self.miniB * self.S) * (self.V / self.kp_softmax_dim1) * 3
        point_mem  = self.precision * (self.miniB * self.S) * (self.V / self.kp_softmax_dim1) * 7
        #2: sigmoid
        #1: one read for reduction, the accumulate is a register
        #1: one for write/extend the reduction result into all cells 
        #3: division needs one for read and one for write. 

        point_comm = self.precision * (self.miniB * self.S) * (self.kp_softmax_dim1) / self.IBK
        #communicating partail sum per row from one GPU to all others to perform sum reduce

        point_time = self.roofline(point_flop, point_mem) + self.O + point_comm
   
        if self.debug:
            print("Softmax point_flop: {:,}, point_mem: {:,}".format(int(point_flop/G), int(point_mem/G)))
            print("Softmax GEMM_time: {:,}, point_time: {:,}\n".format(GEMM_time, point_time))

        return GEMM_time + reduction_time + point_time

    def getSoftmax_b_kp1(self):

        point_flop = (self.miniB * self.S) * (self.V / self.kp_softmax_dim1) * 5
        #1: one for one of the divisions, grad(A) (y=A/B)
        #2: one for division and multiplication, grad(B)
        #1: one for addition, copies turn into add
        #1: one for sigmoid

        point_mem  = self.precision * (self.miniB * self.S) * ((11 * self.V )/ self.kp_softmax_dim1)
        #3: grad(A) in pointwise division
        #3: grad(B) in pointwise division
        #3: addition in copy backprop
        #2: sigmoid

        point_comm = 0

        point_time = self.roofline(point_flop, point_mem) + self.O + point_comm
   

        
        GEMM_time, reduction_time = self.getDistGEMM_b_kp1(self.miniB * self.S, self.projection, self.V, self.kp_softmax_dim1, "softmax_b_kp1")
        
    
        if self.debug:
            print("(gr) Softmax point_flop: {:,}, point_mem: {:,}\n".format(int(point_flop/G), int(point_mem/G)))
        
        return reduction_time + GEMM_time + point_time

    #Row-Column MM
    def getSoftmax_f_kp2(self):
        #Multiply
        assert(self.kp_softmax_type == 2)
        assert(self.kp_softmax_dim1 > 1 or self.kp_softmax_dim2 > 1)
        assert((self.miniB * self.S) % self.kp_softmax_dim1 == 0)
        assert((self.projection if proj else self.D) % self.kp_softmax_dim2 == 0)

        GEMM_time, reduction_time = self.getDistGEMM_f_kp2(self.miniB * self.S, self.projection, self.V, self.kp_softmax_dim1, self.kp_softmax_dim2, "softmax_f_kp2")
        

        #Final matrix after GEMM has (B X S, V) dimensionality
        #We get exponential on each of the elements in a row
        #and then reduce all elements in the row to one
        #Therefore for each row we do V sigmoids and V-1 additions and V divisions
        #For each row, we read V sigmoids and write one
        
        point_flop = (self.miniB * self.S / self.kp_softmax_dim1) * (self.V / self.kp_softmax_dim2) * 3
        point_mem  = self.precision * (self.miniB * self.S / self.kp_softmax_dim1) * (self.V / self.kp_softmax_dim2) * 7
        #2: sigmoid
        #1: one read for reduction, the accumulate is a register
        #1: one for write/extend the reduction result into all cells 
        #3: division needs one for read and one for write. 

        point_comm = self.precision * (self.miniB * self.S / self.kp_softmax_dim1) * (self.kp_softmax_dim2) / self.IBK
        #communicating partail sum per row from one GPU to all others to perform sum reduce

        point_time = self.roofline(point_flop, point_mem) + self.O + point_comm
  

        if self.debug:
            print("Softmax point_flop: {:,}, point_mem: {:,}".format(int(point_flop/G), int(point_mem/G)))
            print("Softmax GEMM_time: {:,}, point_time: {:,}\n".format(GEMM_time, point_time))

        return GEMM_time + point_time + reduction_time

    def getSoftmax_b_kp2(self):

        point_flop = (self.miniB * self.S / self.kp_softmax_dim1) * (self.V / self.kp_softmax_dim2) * 5
        #1: one for one of the divisions, grad(A) (y=A/B)
        #2: one for division and multiplication, grad(B)
        #1: one for addition, copies turn into add
        #1: one for sigmoid

        point_mem  = self.precision * (self.miniB * self.S / self.kp_softmax_dim1) * ((11 * self.V )/ self.kp_softmax_dim2)
        #3: grad(A) in pointwise division
        #3: grad(B) in pointwise division
        #3: addition in copy backprop
        #2: sigmoid

        point_comm = 0

        point_time = self.roofline(point_flop, point_mem) + self.O + point_comm
   

        GEMM_time, reduction_time = self.getDistGEMM_b_kp2(self.miniB * self.S, (self.projection if proj else self.D), self.V, self.kp_softmax_dim1, self.kp_softmax_dim2, "softmax_b_kp2")

    
        if self.debug:
            print("(gr) Softmax point_flop: {:,}, point_mem: {:,}\n".format(int(point_flop/G), int(point_mem/G)))
        
        return reduction_time + GEMM_time + point_time


    def getEmbedding_f(self):
        embedding_mem = 2 * (self.miniB * self.D * self.S * self.precision)
        embedding_time = (embedding_mem)/ (self.mem_bw) + mem_latency + self.O
        if self.debug:
            print("Embedding_mem: {:,}".format(int(embedding_mem/G)))
        return embedding_time


    def getEmbedding_b(self):
        data_transfer  = 0 if (self.dp == 1) else (((self.precision * self.miniB * self.D * self.S) / self.IBD + self.ll) * 2 * (self.dp -1 )) 
        
        embedding_mem = 2 * self.miniB * self.D * self.S * self.precision
        embedding_mem_time = (embedding_mem / self.mem_bw) + mem_latency + self.O

        dt=(self.precision * self.miniB * self.D * self.S)*2
        print("R2: {}".format(dt))

        if self.debug:
            print("(gr) Embedding_mem: {:,}".format(int(embedding_mem/G)))
        return data_transfer + embedding_mem_time



    def getHiddenMem(self, L, Dim1, Dim2, Dim3, S):
        #Activations refer to output activations that need to be stored
        hidden_act = Dim1 * Dim2 * S * L
        hidden_wt  = (Dim2 + 1) * Dim3 * L
        hidden_point = (Dim1 * Dim2 / 2) * 9 * L * S
        #3 sigmoids
        #2 tanh
        #3 pointwise multiply
        #1 addition
        hidden_mem = (hidden_act + hidden_wt + hidden_point) * self.precision
   
        return hidden_mem, hidden_act, hidden_wt, hidden_point
 
    def getSoftmaxMem(self, B, S, P, V):
         #activation output from each layer, assuming input ativation are taken 
        #into account in the previous layer
        softmax_act = B * S * V 
        softmax_wt = (P + 1) * V
        softmax_point = 2 * B * S * V + B * S
        #NOTE: sigmoid and exp could have been combined
        #1 sigmoids
        #1 exp
        #1 pointwise div
        softmax_mem = (softmax_act + softmax_wt + softmax_point) * self.precision

        return softmax_mem, softmax_act, softmax_wt, softmax_point

    def getProjectionMem(self, B, S, P, D):
        projection_act = B * S * P
        projection_wt = (D + 1) * P
        projection_point= B * S * P
        projection_mem = (projection_act + projection_wt + projection_point) * self.precision
      
        return projection_mem, projection_act, projection_wt, projection_point

    def getEmbeddingMem(self, B, S, V, D):
        embedding_act = self.miniB * self.S * self.D
        embedding_wt = self.V * self.D
        embedding_point = 0
        embedding_mem = (embedding_wt + embedding_act + embedding_point) * self.precision

        return embedding_mem, embedding_act, embedding_wt, embedding_point

    def getTotMemReq(self):
        #TODO: Add pointwise op memory 
        hidden_mem, hidden_act, hidden_wt, hidden_point =  self.getHiddenMem(L=self.L, Dim1 = self.miniB, Dim2 = 2 * self.D, Dim3 = self.G * self.D, S = self.S)
        softmax_mem, softmax_act, softmax_wt, softmax_point =  self.getSoftmaxMem(B=self.miniB, S=self.S, P=self.projection, V=self.V)
        projection_mem, projection_act, projection_wt, projection_point =  self.getProjectionMem(B=self.miniB, S=self.S, P=self.projection, D=self.D)
        embedding_mem, embedding_act, embedding_wt, embedding_point =  self.getEmbeddingMem(B=self.miniB, S=self.S, V=self.V, D=self.D)
  
        print("*****embedding_mem: {}, embedding_act: {}, embedding_wt: {}\n".format(embedding_mem/1e9, embedding_act * self.precision/1e9, embedding_wt * self.precision/1e9))

        #tot_mem = hidden_mem + softmax_mem + embedding_mem + projection_mem
        wt_mem = (hidden_wt + softmax_wt + projection_wt + embedding_wt) * self.precision
        act_mem = (hidden_act + softmax_act + projection_act + embedding_act) * self.precision
        point_mem = (hidden_point + softmax_point + projection_point + embedding_point) * self.precision
        tot_mem = wt_mem + act_mem + point_mem

        assert(tot_mem == hidden_mem + softmax_mem + projection_mem + embedding_mem)

        print("**Parameter Breakdown:***")
        print("hidden: {}, softmax: {}, projection: {}, embedding: {}\n".
             format(hidden_wt/1e9, softmax_wt/1e9, projection_wt/1e9, 
             embedding_wt/1e9))

        print("Memory Usage Breakdown (wt/act/pointwise)")
        print("wt_mem: {}, act_mem: {}, point_mem: {},"
              "tot_mem: {}\n".format(
              wt_mem/1e9, act_mem/1e9, point_mem/1e9, tot_mem/1e9))
        return tot_mem, embedding_mem, hidden_mem, softmax_mem, projection_mem


    def getMemUsagePerCore(self):

        #TODO: Add pointwise op memory 
        hidden_mem, hidden_act, hidden_wt, point_act =  self.getHiddenMem(L=self.L/self.hlp, Dim1 = self.miniB / (self.kp_hidden_dim1 if self.kp_hidden_type == 2 else  1), Dim2 = 2 * self.D / (1 if self.kp_hidden_type == 2 else self.kp_hidden_dim1),  Dim3 = self.D * self.G / (self.kp_hidden_dim2 if self.kp_hidden_type == 2 else 1), S = self.S)

        #activation output from each layer, assuming input ativation are taken 
        #into account in the previous layer
        softmax_mem, softmax_act, softmax_wt, softmax_point =  self.getSoftmaxMem(B=self.miniB / (self.kp_softmax_dim1 if self.kp_softmax_type == 2 else  1), S=self.S, P=self.projection/ (1 if self.kp_softmax_type == 2 else self.kp_softmax_dim1 ), V=self.V/(self.kp_softmax_dim2 if self.kp_softmax_type == 2 else 1))

        projection_mem, projection_act, projection_wt, projection_point =  self.getProjectionMem(B=self.miniB/(self.kp_projection_dim1 if self.kp_projection_type == 2 else  1), S=self.S, D=self.D/(1 if self.kp_projection_type == 2 else self.kp_projection_dim1), P=self.projection/(self.kp_projection_dim2 if self.kp_projection_type == 2 else 1))
        
        #embedding_mem = self.miniB * self.S * self.D * self.precision + self.V * self.D / self.kp_embedding_dim1
        embedding_mem, embedding_act, embedding_wt, embedding_point =  self.getEmbeddingMem(B=self.miniB/(self.kp_embedding_dim1 if self.kp_embedding_type==1 else self.kp_embedding_dim1 * self.kp_embedding_dim2), S=self.S, V=self.V, D=self.D)
        
          
        return embedding_mem, hidden_mem, softmax_mem, projection_mem



    def getInterModelCommLatency(self):
        if self.lp == 1:
            return 0

        w = self.precision * self.miniB * self.D / self.IBM
        return w


    def dprint(self, string):
        if self.debug:
            print(string)
    
    def calcTime(self):
        W = self.getInterModelCommLatency() 
        R = self.getR()
        applyGrad_hidden = self.applyGrad(Dim0 = 2 * self.D, Dim1 = self.G * self.D, name = "Hidden")
        applyGrad_projection = self.applyGrad(Dim0 = self.D, Dim1 = self.projection, name = "Projection")
        applyGrad_softmax = self.applyGrad(Dim0 = (self.projection if proj else self.D), Dim1 = self.V, name = "Softmax")
        R += applyGrad_hidden
        L = self.L
        S = self.S

        if self.kp_hidden_type == -1:
            Cf = self.getCf()
            Cb = self.getCb()
        elif self.kp_hidden_type == 1: #CR
            Cf = self.getCf_kp1()
            Cb = self.getCb_kp1()
        elif self.kp_hidden_type== 2: #RC
            Cf = self.getCf_kp2()
            Cb = self.getCb_kp2()
        else:
            print("Incorrect distributed GEMM type, 1: Column-Row, 2: Row-Column")
            sys.exit()
       
        if self.kp_softmax_type == -1:
            softmax_f = self.getSoftmax_f()
            softmax_b = self.getSoftmax_b()
        elif self.kp_softmax_type == 1: #CR
            softmax_f = self.getSoftmax_f_kp1()
            softmax_b = self.getSoftmax_b_kp1()
        elif self.kp_softmax_type== 2: #RC
            softmax_f = self.getSoftmax_f_kp2()
            softmax_b = self.getSoftmax_b_kp2()
        else:
            print("Incorrect distributed GEMM type, 1: Column-Row, 2: Row-Column")
            sys.exit()
        
        if self.kp_projection_type == -1:
            projection_f = self.getProjection_f()
            projection_b = self.getProjection_b()
        elif self.kp_projection_type == 1: #CR
            projection_f = self.getProjection_f_kp1()
            projection_b = self.getProjection_b_kp1()
        elif self.kp_projection_type== 2: #RC
            projection_f = self.getProjection_f_kp2()
            projection_b = self.getProjection_b_kp2()
        else:
            print("Incorrect distributed GEMM type, 1: Column-Row, 2: Row-Column")
            sys.exit()
        

        embedding_f = self.getEmbedding_f()
        embedding_b = self.getEmbedding_b()

        lp = self.lp
        hlp = self.hlp
        kp_hidden_dim1 = self.kp_hidden_dim1
        kp_hidden_dim2 = self.kp_hidden_dim2
        kp_softmax_dim1 = self.kp_softmax_dim1
        kp_softmax_dim2 = self.kp_softmax_dim2
        kp_embedding_dim1 = self.kp_embedding_dim1
        kp_embedding_dim2 = self.kp_embedding_dim2
        kp_projection_dim1 = self.kp_projection_dim1
        kp_projection_dim2 = self.kp_projection_dim2
        kp_hidden_type = self.kp_hidden_type
        kp_softmax_type = self.kp_softmax_type
        kp_embedding_type = self.kp_embedding_type
        dp = self.dp

        #FIXME: THE TOTAL NUMBER OF GPUS IS A FUNCTION OF MAPPING.
        #as of now I return the minimum required assuming other layers
        #can fit in hidden layer memory.This might or might not be true
        embedding_gpu = kp_embedding_dim1 * kp_embedding_dim2 * dp
        hidden_gpu = hlp * kp_hidden_dim1 * kp_hidden_dim2 * dp
        projection_gpu = kp_projection_dim1 * kp_projection_dim2 * dp
        softmax_gpu = kp_softmax_dim1 * kp_softmax_dim2 * dp

        tot_gpu = max(embedding_gpu, hidden_gpu, projection_gpu, softmax_gpu) 
        
        print("th: {}Tflops, mem_bw: {}GB/s, mem_size: {}GB, L2_bw: {}GB/s, L2_size: {}MB, IBK: {}GB/s"
              .format(self.th/1e12/th_scale, self.mem_bw/(1024*1024*1024)/mem_scale, self.mem_size/(1024*1024*1024), self.L2_bw/(1024*1024*1024), self.L2_size/(1024*1024), self.IBK/(1024*1024*1024)))
        #print("th: {}Tflops, mem_bw: {}GB/s, mem_size: {}GB, L2_bw: {}GB/s, L2_size: {}MB"
        #      .format(self.th/1e12 * tot_gpu, self.mem_bw/1e9 * tot_gpu, self.mem_size/1e9 * tot_gpu, self.L2_bw/1e9 * tot_gpu, self.L2_size/1e6 * tot_gpu))
        if self.debug:
          print("dp: {}, lp: {}, hlp: {}, kp_hidden_dim1: {}, kp_hidden_dim2: {}, kp_softmax_dim1: {}, kp_softmax_dim2: {}, kp_embedding_dim1: {}, kp_embedding_dim2: {},  kp_hidden_type: {}, kp_softmax_type: {}, kp_embedding_type: {}\n".
                format(dp, 
                       lp, 
                       hlp, 
                       kp_hidden_dim1, 
                       kp_hidden_dim2, 
                       kp_softmax_dim1, 
                       kp_softmax_dim2, 
                       kp_embedding_dim1, 
                       kp_embedding_dim2, 
                       kp_hidden_type, 
                       kp_softmax_type, 
                       kp_embedding_type))

          
          print("Cf: {}, Cb:; {}, R: {}, softmax_f: {}, softmax_b:{}, embedding_f: {}, embedding_b: {}, projection_f: {}, projection_b: {}\n".
                 format(Cf, 
                        Cb,
                        R,
                        softmax_f, 
                        softmax_b, 
                        embedding_f, 
                        embedding_b,
                        projection_f,
                        projection_b))

        time = 0
        if (self.attached):
            #Assuming softmax is blocking on all the time sequences to arrive 
            #Same for embedding
            time = (softmax_f + softmax_b + embedding_f + embedding_b + 
                    ((projection_f + projection_b + applyGrad_projection) if proj else 0) + 
                    applyGrad_softmax)

        #TODO: This part makes it specific to stack of RNNs
        #Replce it with critical path analysis in a given graph
        if (W <= (Cf * L / hlp)):
          if (Cb >= R):
            time += ((L + L/hlp * (S - 1)) * Cf + W * (hlp - 1) +
                     (L + L/hlp * (S - 1)) * Cb + W * (hlp - 1) + R)
          else:
            time += ((L + L/hlp * (S - 1)) * Cf + W * (hlp - 1) + 
                    (L + L/hlp * (S - 2) + 1) * Cb + R * (L/hlp))
        
        elif (W >= (Cb * L / hlp)):
          if (Cb >= R):
            time += (L * Cf + (S + hlp - 2) * W + 
                    L * Cb + (S + hlp - 2) * W + R)
          else:
            time += (L * Cf + (S + hlp - 2) * W + 
                    (L - L/hlp + 1) * Cb + (S + hlp - 2) * W + (L/hlp) * R)
        else:
          if (Cb >= R):
            time += (L * Cf + (S + hlp - 2) * W +
                    (L + L/hlp * (S - 1)) * Cb + W * (hlp - 1) + R)
          else:
            time += (L * Cf + (S + hlp - 2) * W +
                    (L + L/hlp * (S - 2) + 1) * Cb + R * (L/hlp))
        
        
        self.tot_time = time
        
        tot_param = self.tot_par()
        print("tot_par: {}\n".format(tot_param/1e9))
    

        return time

    def getTime(self):
        return self.tot_time

@click.command()
@click.option("--exp_config", help="path to experiment config", required=True)
@click.option("--debug", help="debug", default=False)
def main(exp_config, debug):
    exp_path = os.path.expandvars(os.path.expanduser(exp_config))
    exp_config = config.parse_config(exp_path)

    TC = TimeCalculation(exp_config)
    TC.debug = debug
    tot_time = TC.calcTime()


    print("Time: {0:.8f}".format(tot_time))


   
if __name__ == "__main__":
    main()
