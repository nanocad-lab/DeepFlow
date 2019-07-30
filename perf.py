#!/tools/lm-venv/py3.6-tf-1.3.0-svail/bin/python

import click
import math
import os
import sys
import config

from parallelism import Parallelism
from topology import Topology
import util
from hw_component import Core, DRAM, L2, SharedMem, RegMem, Network
from model import Model

algByte=False #algorithmic ops false
proj=False #consider projection layer, turn off for validation
tensorflow_overhead=750e-6 #0.75 u-seconds

G=1 #1024.0*1024.0*1024.0


class TimeCalculation:
    def __init__(self, exp_config):
        #Model Parameters
        self.model              = Model(exp_config)
        self.B                  = self.model.batch_size
        self.V                  = self.model.vocab_size
        self.L                  = self.model.num_layers
        self.D                  = self.model.hidden_dim
        self.projection         = self.model.projection
        self.S                  = self.model.seq_len
        self.G                  = self.model.num_gates
        self.NL                 = self.model.num_non_linear
        self.A                  = self.model.num_add
        self.P                  = self.model.num_pointwise
        
        #Software Parameters
        self.O                  = exp_config.sw_config.kernel_launch_overhead
        self.precision          = exp_config.sw_config.precision
        self.attached           = True
        

        #Hardware Parameters
        self.core               = Core(exp_config)
        self.th                 = self.core.getThroughput()

        self.mm                 = DRAM(exp_config) 
        self.mem_bw             = self.mm.getThroughput()
        self.mem_size           = self.mm.getSize()
        self.mem_latency        = self.mm.getLatency()
        
        self.L2Mem              = L2(exp_config) 
        self.L2_bw              = self.L2Mem.getThroughput()
        self.L2_size            = self.L2Mem.getSize()
        self.L2_tile_dim        = self.L2Mem.getTileDim()
        self.L2_latency         = self.L2Mem.getLatency()

        self.sharedMem          = SharedMem(exp_config) 
        self.shared_mem_bw      = self.sharedMem.getThroughput()
        self.shared_mem_size    = self.sharedMem.getSize()
        self.shared_mem_tile_dim= self.sharedMem.getTileDim()
        self.shared_mem_latency = self.sharedMem.getLatency()

        self.regMem             = RegMem(exp_config) 
        self.reg_bw             = self.regMem.getThroughput()
        self.reg_size           = self.regMem.getSize()
        self.reg_tile_dim       = self.regMem.getTileDim()
        self.reg_latency        = self.regMem.getLatency()

        self.network            = Network(exp_config)
        self.IBK                = self.network.kernel_throughput
        self.IBD                = self.network.data_throughput
        self.IBM                = self.network.layer_throughput
        self.LLK                = self.network.kernel_latency
        self.LLD                = self.network.data_latency
        self.LLM                = self.network.layer_latency
        
        #Scheduling Parameters
        par                     = Parallelism(exp_config)
        par.findParallelStrategy()
        self.autoPar            = par.autoPar
        self.lp                 = par.lp
        self.kp_hidden_dim1     = par.kp_hidden_dim1
        self.kp_softmax_dim1    = par.kp_softmax_dim1
        self.kp_embedding_dim1  = par.kp_embedding_dim1
        self.kp_projection_dim1 = par.kp_projection_dim1
        self.kp_hidden_dim2     = par.kp_hidden_dim2
        self.kp_softmax_dim2    = par.kp_softmax_dim2
        self.kp_embedding_dim2  = par.kp_embedding_dim2
        self.kp_projection_dim2 = par.kp_projection_dim2
        self.dp                 = par.dp
        self.kp_hidden_type     = par.kp_hidden_type #1: CR, 2: RC
        self.kp_softmax_type    = par.kp_softmax_type #1: CR, 2: RC
        self.kp_embedding_type  = par.kp_embedding_type #1: CR, 2: RC
        self.kp_projection_type = par.kp_projection_type #1: CR, 2: RC

        #Define miniBatch size
        self.miniB              = math.ceil(self.B / self.dp)

        #Statistics Param
        self.tot_flop           = 0
        self.tot_mem            = 0
        self.tot_time           = 0
        self.debug              = False

    #Number of parameters
    def tot_param(self):
        embedding = self.V * self.D
        hidden = (2 * self.D + 1) * (self.G * self.D) * self.L
        projection = self.D * self.projection
        softmax = ((self.projection if proj else self.D) + 1) * self.V

        tot_param = embedding + hidden + projection + softmax
        return tot_param

    def printSysConfig(self, exp_config):

      kiloByte = 1024
      megaByte = kiloByte * 1024
      gigaByte = megaByte * 1024
      teraByte = gigaByte * 1024

      print("======================")
      print("Hardware Configuration")
      print("======================")
      print("Throughput: {:.1f} Tflops\n"
            "Memory Bandwidth: {:.1f} GB/s\n"
            "Memory Size: {:.1f} GB\n"
            "L2 Bandwidth: {:.1f} TB/s\n"
            "L2 Size: {:.1f} MB\n"
            "Shared Memory Bandwidth: {:.1f} TB/s\n"
            "Shared Memory Size: {:.1f} MB\n"
            "Register Memory Bandwidth: {:.1f} TB/s\n"
            "Register Size: {:.1f} MB\n"
            "Interconnection Bandwidth (Data Dimension): {:.1f} GB/s"
            .format(self.core.nominal_throughput/1e12, 
                    self.mm.dynamic_throughput/(gigaByte), 
                    self.mm.size/(gigaByte), 
                    self.L2Mem.dynamic_throughput/(teraByte), 
                    self.L2Mem.size/(megaByte), 
                    self.sharedMem.dynamic_throughput/(teraByte),
                    self.sharedMem.size/(megaByte),
                    self.regMem.dynamic_throughput/(teraByte),
                    self.regMem.size/(megaByte),
                    self.IBD/(gigaByte)))
       

      M = self.mem_size
      tot_mem, embedding_mem, hidden_mem, softmax_mem, projection_mem, wt_mem, act_mem, point_mem = util.getTotMemReq(exp_config)
      print("\n\n===========================================")
      print("Memory Requirement Breakdown per Data Shard")
      print("===========================================")
      print("Total Memory: {:.1f} GB\n"
            "Embedding Memory: {:.1f} GB\n"
            "Hidden Memory: {:.1f} GB\n"
            "Softmax Memory: {:.1f} GB\n"
            "Projection Memory: {:.1f} GB"
            .format(tot_mem/gigaByte, 
                    embedding_mem/gigaByte, 
                    hidden_mem/gigaByte, 
                    softmax_mem/gigaByte, 
                    projection_mem/gigaByte))
      
      print("\nTotal Memory: {:.1f} GB\n"
            "Weight Memory: {:.1f} GB\n"
            "Activation Memory: {:.1f} GB\n"
            "Pointwise Memory: {:.1f} GB\n"
            .format(tot_mem/gigaByte, 
                    wt_mem/gigaByte, 
                    act_mem/gigaByte, 
                    point_mem/gigaByte))


      print("\nMemory Overflow Rate (Total Memory Required per Data Shard / Memory capacity per node): {:.1f}\n"
            .format(tot_mem/M))

      
      tot_mem, embedding_mem, hidden_mem, softmax_mem, projection_mem = util.getMemUsagePerCore(exp_config)
      print("\n\n=========================================================")
      print("Memory Requirement Breakdown per Data Shard Per Model Shard")
      print("===========================================================")
      print("Total Memory: {:.1f} GB\n"
          "Embedding Memory: {:.1f} GB\n"
          "Hidden Memory: {:.1f} GB\n"
          "Softmax Memory: {:.1f} GB\n"
          "Projection Memory: {:.1f} GB"
             .format(tot_mem/gigaByte, 
                     embedding_mem/gigaByte, 
                     hidden_mem/gigaByte, 
                     softmax_mem/gigaByte, 
                     projection_mem/gigaByte))

      print("\nMemory Overflow Rate (Total Memory Required per Data Shard / Memory capacity per node): {:.1f}\n"
            .format(tot_mem/M))

      print("\nTotal Memory: {:.1f} GB\n"
            "Weight Memory: {:.1f} GB\n"
            "Activation Memory: {:.1f} GB\n"
            "Pointwise Memory: {:.1f} GB\n"
            .format(tot_mem/gigaByte, 
                    wt_mem/gigaByte, 
                    act_mem/gigaByte, 
                    point_mem/gigaByte))
      
      print("\n\n====================")
      print("Parallelism Strategy")
      print("====================")
      print("dp: {}, lp: {}, lp: {}, kp_hidden_dim1: {}, kp_hidden_dim2: {}," 
             "kp_softmax_dim1: {}, kp_softmax_dim2: {}, kp_embedding: {}," 
             "kp_projection_dim1: {}, kp_proejction_dim2: {}\n"
             .format(self.dp,
              self.lp, self.lp, self.kp_hidden_dim1, self.kp_hidden_dim2, 
              self.kp_softmax_dim1, self.kp_softmax_dim2, self.kp_embedding_dim1, 
              self.kp_projection_dim1, self.kp_projection_dim2))   



    def roofline(self, flop, gmem, l2mem=0, smem=0, rmem=0):

        self.tot_flop += flop
        self.tot_mem  += gmem

        inflection_point = self.th / self.mem_bw
        comp_int = flop / gmem
        #L2_tile_dim, sm_tile_dim = self.getTileDim()
        #single_block_comp_time = (2 * L2_tile_dim * L2_tile_dim * L2_tile_dim) / self.th
        #single_block_mem_time  = (3 * L2_tile_dim * L2_tile_dim * self.precision) / self.mem_bw
      
        time  = 0
        if comp_int < inflection_point: #mem-bound
            time = ((gmem / self.mem_bw) + self.mem_latency + 
                    (0 if self.L2_bw == 0 else (l2mem / self.L2_bw)) + 
                    (0 if self.shared_mem_bw == 0 else (smem / self.shared_mem_bw)) +
                    (0 if self.reg_bw == 0 else (smem / self.reg_bw)))
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
        GEMM_flop, GEMM_gmem, GEMM_l2mem, GEMM_smem, GEMM_rmem = self.GEMM(A, B, C)
        GEMM_flop_t, GEMM_gmem_t, GEMM_l2mem_t, GEMM_smem_t, GEMM_rmem_t = self.GEMM(C, B, A)
        
        GEMM_time = self.roofline(GEMM_flop, GEMM_gmem, GEMM_l2mem, GEMM_smem, GEMM_rmem) + self.O
        GEMM_time_t = self.roofline(GEMM_flop_t, GEMM_gmem_t, GEMM_l2mem_t, GEMM_smem_t, GEMM_rmem_t) + self.O
        
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
    def GEMM(self, A_, B_, C_):
        A = util.power2RoundUp(A_)
        B = util.power2RoundUp(B_)
        C = util.power2RoundUp(C_)
        #A = A_
        #B = B_
        #C = C_


        if self.debug:
          print("Matrix dimension at Global Memory: {:,} x {:,} x {:,}".format(A, B, C))

        GEMM_flop = 2 * A * B * C
        #2 for multiplly and add
        X1 = self.L2_tile_dim
        X2 = self.shared_mem_tile_dim
        X3 = self.reg_tile_dim
      
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
             if self.debug:
                print("Matrix dimension at L2 Memory: {:,} x {:,} x {:,}".format(X1, X1, X1))

             #Modeling Shared_memory
             #number of L2memory accesses going through Shared memory
             reload_AB = 1
             reload_BC = 1
             reload_AC = 1
            
             if X2 > 0:
                As = X1
                Bs = X1
                Cs = X1
                
                if X2 > X1:
                    X2 = X1

                if  Bs <= X2:
                    reload_AB = 1
                    reload_BC = math.ceil(As / X2)
                    reload_AC = 1
                else:
                    reload_AB = math.ceil(Cs / X2)
                    reload_BC = math.ceil(As / X2)
                    reload_AC = 1

             num_repeat = A/X1 * C/X1
             
             GEMM_l2mem = (num_repeat *
                           self.core.num_bundle * 
                           (As * Bs * reload_AB + 
                            Bs * Cs * reload_BC + 
                            As * Cs * reload_AC) *
                            self.precision)
             if self.debug:
                print("Matrix dimension at Shared Memory: {:,} x {:,} x {:,}".format(X2, X2, X2))

             #Modeling Register file
             #number of shared memory accesses through registers
             #NOTE:For GPU hardware where the register file per SM is larger than shared memory,
             #restreaming factor should be one.
             reload_AB = 1
             reload_BC = 1
             reload_AC = 1
             
            
             if X3 > 0:
                Ar = X2
                Br = X2 
                Cr = X2

                if X3 > X2:
                    X3 = X2

                if  Bs <= X2:
                    reload_AB = 1
                    reload_BC = math.ceil(Ar / X3)
                    reload_AC = 0 # Results are usually directly written to higher-level 
                else:
                    reload_AB = math.ceil(Cr / X3)
                    reload_BC = math.ceil(As / X3)
                    reload_AC = 0
             
             num_repeat  *= As/X2 * Cs/X2 
             GEMM_smem    = (num_repeat *
                             (Ar * Br * reload_AB + 
                              Br * Cr * reload_BC + 
                              Ar * Cr * reload_AC) * 
                              self.precision)

             if self.debug:
                print("Matrix dimension at Register Memory: {:,} x {:,} x {:,}".format(X3, X3, X3))

             #number of accesses to register file
             #TODO: Verify this is true...
             GEMM_rmem    = GEMM_flop * (1 + 3 + 1)
             #1: memory from read
             #3: multiply add
             #1: write to memory


             #Heuristics for other stuff besides multiply and add (like address calculation)
             GEMM_flop = GEMM_flop + As * Cs * (15 + 5 * math.ceil(Bs / X2))


             if self.debug:
                print("l2mem: reload_AB: {}, reload_AC: {}, reload_BC: {}\n", reload_AB, reload_AC, reload_BC)
    
        return GEMM_flop, GEMM_gmem, GEMM_l2mem, GEMM_smem, GEMM_rmem

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


    #Reduction and all-gather time estimation
    def getR(self, Dim0 = None, Dim1 = None, p = None, ib = None, ll = None, partial = None,
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
        if (ll == None): 
            ll = self.LLD
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
            data_transfer  = ((((self.precision * Dim0 * Dim1) / p) / ib) + ll) * factor * (p - 1)
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
                                   ll = self.LLK,
                                   partial = True,
                                   allReduce = True)
        return GEMM_time, reduction_time


    def getDistGEMM_b_kp1(self, m, k, n, dim1, name):
        #All-gather gradients wrt activations (BS,V) in each GPU node before MM
        reduction_time = self.getR(Dim0 = m, 
                                   Dim1 = n,
                                   p = dim1,
                                   ib = self.IBK,
                                   ll = self.LLK,
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
                                   ll = self.LLK,
                                   partial = False,
                                   allReduce = False)
        return GEMM_time, reduction_time

    def getDistGEMM_b_kp2(self, m, k, n, dim1, dim2, name):
        #To calculate grad(wt), partially gather transpose(activations) across rows before MM
        reduction_time_wt1 = self.getR(Dim0 = k, 
                                       Dim1 = m,
                                       p = dim1,
                                       ib = self.IBK,
                                       ll = self.LLK,
                                       partial = True,
                                       allReduce = False)
        #To calculate grad(wt), gather output matrix (A') across row within one column before MM
        reduction_time_wt2 = self.getR(Dim0 = m, 
                                       Dim1 = n / dim2,
                                       p = dim1,
                                       ib = self.IBK,
                                       ll = self.LLK,
                                       partial = False,
                                       allReduce = False)

        #To calculate grad(wt), gather weght matrix (W) across row within one column after MM
        reduction_time_wt3 = self.getR(Dim0 = m, 
                                       Dim1 = n / dim2,
                                       p = dim1,
                                       ib = self.IBK,
                                       ll = self.LLK,
                                       partial = False,
                                       allReduce = False)
        ########################################################################################
         #To calculate grad(act), gather output matrix (A') across columns within one row before MM
        reduction_time_act1 = self.getR(Dim0 = m / dim1, 
                                       Dim1 = n,
                                       p = dim2,
                                       ib = self.IBK,
                                       ll = self.LLK,
                                       partial = False,
                                       allReduce = False)

        #To calculate grad(act), partially gather transpose(weight) across rows before MM
        reduction_time_act2 = self.getR(Dim0 = k, 
                                       Dim1 = n,
                                       p = dim2,
                                       ib = self.IBK,
                                       ll = self.LLK,
                                       partial = True,
                                       allReduce = False)
               #To calculate grad(wt), partally gather weght matrix (W) across row within one column after MM
        reduction_time_act3 = self.getR(Dim0 = m / dim1, 
                                       Dim1 = k,
                                       p = dim2,
                                       ib = self.IBK,
                                       ll = self.LLK,
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
        embedding_time = (embedding_mem)/ (self.mem_bw) + self.mem_latency + self.O
        if self.debug:
            print("Embedding_mem: {:,}".format(int(embedding_mem/G)))
        return embedding_time


    def getEmbedding_b(self):
        p2p_data_transfer = (self.precision * self.miniB * self.D * self.S)
        data_transfer_time  = 0 if (self.dp == 1) else (((p2p_data_transfer) / self.IBD + self.LLD) * 2 * (self.dp -1 )) 
        
        embedding_mem = 2 * self.miniB * self.D * self.S * self.precision
        embedding_mem_time = (embedding_mem / self.mem_bw) + self.mem_latency + self.O


        if self.debug:
            print("(gr) Embedding_mem: {:,}".format(int(embedding_mem/G)))
        return data_transfer_time + embedding_mem_time




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
        hidden_gpu = lp * kp_hidden_dim1 * kp_hidden_dim2 * dp
        projection_gpu = kp_projection_dim1 * kp_projection_dim2 * dp
        softmax_gpu = kp_softmax_dim1 * kp_softmax_dim2 * dp

        tot_gpu = max(embedding_gpu, hidden_gpu, projection_gpu, softmax_gpu) 
        
        #print("th: {}Tflops, mem_bw: {}GB/s, mem_size: {}GB, L2_bw: {}GB/s, L2_size: {}MB"
        #      .format(self.th/1e12 * tot_gpu, self.mem_bw/1e9 * tot_gpu, self.mem_size/1e9 * tot_gpu, self.L2_bw/1e9 * tot_gpu, self.L2_size/1e6 * tot_gpu))
        if self.debug:
          print("dp: {}, lp: {}, kp_hidden_dim1: {}, kp_hidden_dim2: {}, kp_softmax_dim1: {}, kp_softmax_dim2: {}, kp_embedding_dim1: {}, kp_embedding_dim2: {},  kp_hidden_type: {}, kp_softmax_type: {}, kp_embedding_type: {}\n".
                format(dp, 
                       lp, 
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
        if (W <= (Cf * L / lp)):
          if (Cb >= R):
            time += ((L + L/lp * (S - 1)) * Cf + W * (lp - 1) +
                     (L + L/lp * (S - 1)) * Cb + W * (lp - 1) + R)
          else:
            time += ((L + L/lp * (S - 1)) * Cf + W * (lp - 1) + 
                    (L + L/lp * (S - 2) + 1) * Cb + R * (L/lp))
        
        elif (W >= (Cb * L / lp)):
          if (Cb >= R):
            time += (L * Cf + (S + lp - 2) * W + 
                    L * Cb + (S + lp - 2) * W + R)
          else:
            time += (L * Cf + (S + lp - 2) * W + 
                    (L - L/lp + 1) * Cb + (S + lp - 2) * W + (L/lp) * R)
        else:
          if (Cb >= R):
            time += (L * Cf + (S + lp - 2) * W +
                    (L + L/lp * (S - 1)) * Cb + W * (lp - 1) + R)
          else:
            time += (L * Cf + (S + lp - 2) * W +
                    (L + L/lp * (S - 2) + 1) * Cb + R * (L/lp))
        
        
        self.tot_time = time
        
        tot_param = self.tot_param()
        print("#Parameters: {:.2f} Billion\n".format(tot_param/1e9))
    

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

    TC.printSysConfig(exp_config)

    print("Time: {0:.8f}".format(tot_time))

   
if __name__ == "__main__":
    main()
