#!/tools/lm-venv/py3.6-tf-1.3.0-svail/bin/python

import click
import math
import os
import sys
import config
import shutil

from parallelism import Parallelism
from topology import Topology
from simulate import Graph
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

        #try:
        #    assert(self.mem_bw > 0), "mem_bw = 0"
        #except Exception as e:
        #    print("The LLM bandwidth cannot be zero\n"
        #    "{}".format(e), flush=True)
        #    sys.exit(0)


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
        
        #System Parameters
        self.num_wafer          = exp_config.system_config.num_wafers
        self.num_workers        = exp_config.system_config.num_workers

        self.network            = Network(exp_config)

        intra_throughput, inter_throughput = self.network.calcThroughput()
        intra_latency, inter_latency       = self.network.calcLatency()
                
        inter_derate            = exp_config.system_config.inter_derate
        intra_derate            = exp_config.system_config.intra_derate
        par2cross               = exp_config.system_config.par2cross

        #cross communication will pass through intra links too
        if self.num_wafer > 1 and self.num_workers > 1:
            derated_inter_throughput = min(intra_throughput/intra_derate, 
                                         inter_throughput/inter_derate)
        else:
            derated_inter_throughput = 0

        if self.num_workers > 1:
            derated_intra_throughput = intra_throughput/intra_derate
        else:
            derated_intra_throughput = 0

        self.IBK1, self.LLK1    = ((derated_inter_throughput, inter_latency) if par2cross["kp1"] else 
                                   (derated_intra_throughput, intra_latency)) 
        self.IBK2, self.LLK2    = ((derated_inter_throughput, inter_latency) if par2cross["kp2"] else 
                                   (derated_intra_throughput, intra_latency)) 
        self.IBD, self.LLD      = ((derated_inter_throughput, inter_latency) if par2cross["dp"] else 
                                   (derated_intra_throughput, intra_latency)) 
        self.IBL, self.LLL      = ((derated_inter_throughput, inter_latency) if par2cross["lp"] else 
                                   (derated_intra_throughput, intra_latency)) 
        
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

    def printSysConfig(self, exp_config, output_file):

      kiloByte = 1024
      megaByte = kiloByte * 1024
      gigaByte = megaByte * 1024
      teraByte = gigaByte * 1024

      with open(output_file, "w") as f:
            f.write("======================\n")
            f.write("Hardware Configuration\n")
            f.write("======================\n")
            f.write("Throughput: {:.1f} Tflops\n"
                    "Memory Bandwidth: {:.1f} GB/s\n"
                    "Memory Size: {:.1f} GB\n"
                    "L2 Bandwidth: {:.1f} TB/s\n"
                    "L2 Size: {:.1f} MB\n"
                    "Shared Memory Bandwidth: {:.1f} TB/s\n"
                    "Shared Memory Size: {:.1f} MB\n"
                    "Register Memory Bandwidth: {:.1f} TB/s\n"
                    "Register Size: {:.1f} MB\n"
                    "Intra-node Bandwidth: {:.1f} GB/s\n"
                    "Inter-node Bandwidth: {:.1f} GB/s\n"
                    .format(self.core.operating_throughput/1e12, 
                            self.mm.dynamic_throughput/(gigaByte), 
                            self.mm.size/(gigaByte), 
                            self.L2Mem.dynamic_throughput/(teraByte), 
                            self.L2Mem.size/(megaByte), 
                            self.sharedMem.dynamic_throughput/(teraByte),
                            self.sharedMem.size/(megaByte),
                            self.regMem.dynamic_throughput/(teraByte),
                            self.regMem.size/(megaByte),
                            self.network.intra_network.throughput/(gigaByte),
                            self.network.inter_network.throughput/(gigaByte)))

            M = self.mem_size
            tot_mem, embedding_mem, hidden_mem, softmax_mem, projection_mem, wt_mem, act_mem, point_mem = util.getTotMemReq(exp_config)
            f.write("\n\n===========================================\n")
            f.write("Memory Requirement Breakdown per Data Shard\n")
            f.write("===========================================\n")
            f.write("Total Memory: {:.1f} GB\n"
                    "Embedding Memory: {:.1f} GB\n"
                    "Hidden Memory: {:.1f} GB\n"
                    "Softmax Memory: {:.1f} GB\n"
                    "Projection Memory: {:.1f} GB\n"
                    .format(tot_mem/gigaByte, 
                            embedding_mem/gigaByte, 
                            hidden_mem/gigaByte, 
                            softmax_mem/gigaByte, 
                            projection_mem/gigaByte))
            
            f.write("\nTotal Memory: {:.1f} GB\n"
                    "Weight Memory: {:.1f} GB\n"
                    "Activation Memory: {:.1f} GB\n"
                    "Pointwise Memory: {:.1f} GB\n"
                    .format(tot_mem/gigaByte, 
                            wt_mem/gigaByte, 
                            act_mem/gigaByte, 
                            point_mem/gigaByte))


            f.write("\nMemory Overflow Rate (Total Memory Required per Data Shard / Memory capacity per node): {:.1f}\n".format(float("inf") if M==0 else tot_mem/M))

            
            tot_mem, embedding_mem, hidden_mem, softmax_mem, projection_mem = util.getMemUsagePerCore(exp_config)
            f.write("\n\n=========================================================\n")
            f.write("Memory Requirement Breakdown per Data Shard Per Model Shard\n")
            f.write("===========================================================\n")
            f.write("Total Memory: {:.1f} GB\n"
                    "Embedding Memory: {:.1f} GB\n"
                    "Hidden Memory: {:.1f} GB\n"
                    "Softmax Memory: {:.1f} GB\n"
                    "Projection Memory: {:.1f} GB"
                       .format(tot_mem/gigaByte, 
                               embedding_mem/gigaByte, 
                               hidden_mem/gigaByte, 
                               softmax_mem/gigaByte, 
                               projection_mem/gigaByte))

            f.write("\nMemory Overflow Rate (Total Memory Required per Data Shard Per Model Shard/ Memory capacity per node): {:.1f}\n"
                  .format(float("inf") if M == 0 else tot_mem/M))

            f.write("\nTotal Memory: {:.1f} GB\n"
                    "Weight Memory: {:.1f} GB\n"
                    "Activation Memory: {:.1f} GB\n"
                    "Pointwise Memory: {:.1f} GB\n"
                    .format(tot_mem/gigaByte, 
                            wt_mem/gigaByte, 
                            act_mem/gigaByte, 
                            point_mem/gigaByte))
            
            f.write("\n\n====================\n")
            f.write("Parallelism Strategy\n")
            f.write("====================\n")
            f.write("dp: {}, lp: {}, lp: {}, kp_hidden_dim1: {}, kp_hidden_dim2: {}," 
                    "kp_softmax_dim1: {}, kp_softmax_dim2: {}, kp_embedding: {}," 
                    "kp_projection_dim1: {}, kp_proejction_dim2: {}\n"
                    .format(self.dp,
                     self.lp, self.lp, self.kp_hidden_dim1, self.kp_hidden_dim2, 
                     self.kp_softmax_dim1, self.kp_softmax_dim2, self.kp_embedding_dim1, 
                     self.kp_projection_dim1, self.kp_projection_dim2))   



    def roofline(self, flop, gmem, l2mem=0, smem=0, rmem=0, name=''):

        try:
            assert(gmem > 0) , "gmem = 0"
        except Exception as e:
            print("Number of accesses to the last level of memory hierarchy cannot be zero:\n"
            "{}".format(e), flush=True)
            sys.exit(0)

        self.tot_flop += flop
        self.tot_mem  += gmem

        inflection_point_gmem   = float("inf") if self.mem_bw == 0 else self.th / self.mem_bw
        inflection_point_l2mem  = float("inf") if self.L2_bw == 0 else self.th / self.L2_bw
        inflection_point_smem   = float("inf") if self.shared_mem_bw == 0 else self.th / self.shared_mem_bw
        inflection_point_rmem   = float("inf") if self.reg_bw == 0 else self.th / self.reg_bw

        comp_int_gmem  = 0 if gmem == 0 else flop / gmem
        comp_int_l2mem = 0 if l2mem == 0 else flop / l2mem
        comp_int_smem  = 0 if smem == 0 else flop / smem
        comp_int_rmem  = 0 if rmem == 0 else flop / rmem
     
        #All memory levels except for the last level are optional
        #That's why if they do not exist their time contribution is zero
        #for the last level (gmem), if it does not exist the time should be infinit
        time_gmem  = 0
        if comp_int_gmem < inflection_point_gmem: #mem-bound
            time_gmem = (float("inf") if (self.mem_bw == 0 or gmem == 0) else (gmem / self.mem_bw)) + self.mem_latency
        else: #compute-bound
            time_gmem = float("inf") if (self.th == 0) else (flop / self.th)
        
        time_l2mem  = 0
        if comp_int_l2mem < inflection_point_l2mem: #mem-bound
            time_l2mem = (0 if (self.L2_bw == 0 or l2mem == 0) else (l2mem / self.L2_bw))
            #print("mem_bound")
        else: #compute-bound
            #print("compute")
            time_l2mem = float("inf") if (self.th == 0) else (flop / self.th)
        
        time_smem  = 0
        if comp_int_smem < inflection_point_smem: #mem-bound
            time_smem = (0 if (self.shared_mem_bw == 0 or smem == 0) else (smem / self.shared_mem_bw))
        else: #compute-bound
            time_smem = float("inf") if (self.th == 0) else (flop / self.th)
        
        time_rmem  = 0
        if comp_int_rmem < inflection_point_rmem: #mem-bound
            time_rmem = (0 if (self.reg_bw == 0 or rmem == 0) else (rmem / self.reg_bw))
        else: #compute-bound
            time_rmem = float("inf") if (self.th == 0) else (flop / self.th)
        
        time = max(time_gmem, time_l2mem, time_smem, time_rmem)
        
        
        #print("Name: {}, Time: {}, gmem: {}, l2mem: {}, smem: {}, rmem: {}, MemBW: {}".format(name, time, time_gmem, time_l2mem, time_smem, time_rmem, self.mem_bw/1e12))
        if self.debug:
            print("inflection_point_gmem: {:.2f}".format(inflection_point_gmem))
            print("comp_int_gmem: {:.2f}".format(comp_int_gmem))
            print("inflection_point_L2mem: {:.2f}".format(inflection_point_l2mem))
            print("comp_int_L2mem: {:.2f}".format(comp_int_l2mem))
        
        return time

    
    def getGEMMTime(self, A, B, C, name):
        GEMM_flop, GEMM_gmem, GEMM_l2mem, GEMM_smem, GEMM_rmem = self.GEMM(A, B, C)
        GEMM_flop_t, GEMM_gmem_t, GEMM_l2mem_t, GEMM_smem_t, GEMM_rmem_t = self.GEMM(C, B, A)
        
        GEMM_time = self.roofline(GEMM_flop, GEMM_gmem, GEMM_l2mem, GEMM_smem, GEMM_rmem, name) + self.O
        GEMM_time_t = self.roofline(GEMM_flop_t, GEMM_gmem_t, GEMM_l2mem_t, GEMM_smem_t, GEMM_rmem_t, name) + self.O
        
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

        GEMM_flop = 0
        GEMM_gmem = 0
        GEMM_l2mem = 0
        GEMM_smem = 0
        GEMM_rmem = 0

        num_repeat = 1

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
        
             if X1 != 0:
                As = X1
                Bs = X1
                Cs = X1
             else:
                As = A
                Bs = B
                Cs = C
            
             if X2 > X1 and X1 != 0:
                 X2 = X1

             if X2 > 0:
                if  Bs <= X2:
                    reload_AB = 1
                    reload_BC = math.ceil(As / X2)
                    reload_AC = 1
                else:
                    reload_AB = math.ceil(Cs / X2)
                    reload_BC = math.ceil(As / X2)
                    reload_AC = 1
                
                try:
                    num_repeat = A/X1 * C/X1
                except:
                    num_repeat = 1
             
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
             
             if X2 != 0:
                Ar = X2
                Br = X2 
                Cr = X2
             else:
                Ar = As
                Br = Bs
                Cr = Cs
            
             if X3 > X2 and X2 != 0:
                X3 = X2

             if X3 > 0:
                if  Bs <= X2:
                    reload_AB = 1
                    reload_BC = math.ceil(Ar / X3)
                    reload_AC = 0 # Results are usually directly written to higher-level 
                else:
                    reload_AB = math.ceil(Cr / X3)
                    reload_BC = math.ceil(Ar / X3)
                    reload_AC = 0
            
                try:
                    num_repeat  *= As/X2 * Cs/X2 
                except:
                    num_repeat  *= 1

                GEMM_smem    = (num_repeat *
                                self.core.num_bundle *
                                (Ar * Br * reload_AB + 
                                 Br * Cr * reload_BC + 
                                 Ar * Cr * reload_AC) * 
                                 self.precision)

             if self.debug:
                print("Matrix dimension at Register Memory: {:,} x {:,} x {:,}".format(X3, X3, X3))

             #number of accesses to register file
             #TODO: Verify this is true...
             GEMM_rmem    = GEMM_flop * 2
          

             if X3 == 0:
               GEMM_smem  = GEMM_rmem
               GEMM_rmem  = 0
             if X2 == 0:
               GEMM_l2mem = GEMM_smem
               GEMM_smem = 0
             if X1 == 0:
               GEMM_gmem  = GEMM_l2mem
               GEMM_l2mem = 0

             #Heuristics for other stuff besides multiply and add (like address calculation)
             #try:
             #   GEMM_flop = GEMM_flop + As * Cs * (15 + 5 * math.ceil(Bs / X2))
             #except:
             #   GEMM_flop = GEMM_flop + As * Cs * 15


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
        point_comm =  self.miniB * (self.G * self.D / self.kp_hidden_dim1) * 3 / self.IBK1
        # 3 refers to the number of pointwise ops (mul + add + mul) on the
        # critical path whose inputs are located across different GPUs
        #NOTE:Assuming all communications can happpen in parallel

        point_time = self.roofline(point_flop, point_mem, name='pointwise_cf_kp1') + self.O + point_comm


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
   
        point_comm =  self.miniB * (self.G * self.D / self.kp_hidden_dim1) * 3 / self.IBK1
        #3 refers to the number of pointwise ops (mul + tanh + mul) on
        # critical path whose inputs are located across different GPUs
        #NOTE:Assuming all communications can happpen in parallel


        point_time = self.roofline(point_flop, point_mem, name='pointwise_Cb_kp1') + self.O + point_comm


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
                       (self.G * self.D / self.kp_hidden_dim2) * 3 / self.IBK2)
        #3 refers to the number of pointwise ops (mul + add + mul) whose inputs
        #across different GPU

        point_time = self.roofline(point_flop, point_mem, name='pointwise_Cf_kp2') + self.O + point_comm

        
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
   
        point_comm =  self.miniB * (self.G * self.D / self.kp_hidden_dim2) * 3 / self.IBK2
        #3 refers to the number of pointwise ops (mul + add +tanh + mul) on 
        #3 refers to the number of hops to gather i,f, o and c in each GPU
        #in order to perform (B,4D)x(4D,2D)

        point_time = self.roofline(point_flop, point_mem, name='pointwise_Cb_kp2') + self.O + point_comm


         
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

        point_time = self.roofline(point_flop, point_mem, name='pointwise_Cf') + 15 * self.O

  
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
        point_time = self.roofline(point_flop, point_mem, name='pointwise_Cb') + 15 * self.O
   
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
        """Partial: True, All-reduce:True, half All-reduce"""
        """Partial: True, All-reduce:False, All-gather"""
        """Partial: False, All-reduce:True, All-reduce"""
        """Partial: False, All-reduce:False, All-gather"""
        if (Dim0 == None):
            #for data parallel reduction, 
            Dim0 = (2 * self.D // self.kp_hidden_dim) if (self.kp_hidden_type == 1) else (2 * self.D // self.kp_hidden_dim2 if (self.kp_hidden_type == 2) else (2 * self.D))
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
            data_prep = self.roofline(data_prep_comp, data_prep_mem, name='R-prepTime')
        else:
            #Assuming point-2-point link between consecutive data partitions
            #In other words, the network topology assumed is Ring, 
            #therefore all (dp-1) transfers can happen in parallel,
            #To assume different toplogy data_transfer formulation should change
            #e.g. assuming bus, data_transfer formulation would change as follows:
            #data_transfer  = ((self.precision * self.D * self.D) * (self.dp /self.dp)) * 
            #                 (self.G * 2) * (2 * (self.dp - 1))) / self.IBD
            factor = (1 if partial or not allReduce else 2)
            data_transfer  = float("inf") if (ib == 0) else ((((self.precision * Dim0 * Dim1) / p) / ib) + ll) * factor * (p - 1)
            #dt = ((self.precision * Dim0 * Dim1) / p) * factor * (p - 1)
            
            data_prep_comp = (Dim0 * Dim1) / p
            data_prep_mem  = (3 * self.precision * Dim0 * Dim1 / p) 
            data_prep = ((self.roofline(data_prep_comp, data_prep_mem, name='R-prepTime') + self.O)
                          * (p - 1))

            #print("R1: {}, factor: {}\n".format(dt,factor))
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

        gradclip_time = self.roofline(gradclip_comp, gradclip_mem, name='pointwise-grad-clipping')

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
        applyGrad_time = self.roofline(applyGrad_comp, applyGrad_mem, name='pointwise-applyGrad')
       
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
                                   ib = self.IBK1,
                                   ll = self.LLK1,
                                   partial = True,
                                   allReduce = True)
        return GEMM_time, reduction_time


    def getDistGEMM_b_kp1(self, m, k, n, dim1, name):
        #calculate grad wrt. act (A'. W^T)
        #gather whole(A') before MM
        #A' is distibuted as columns across different nodes
        reduction_time = self.getR(Dim0 = m, 
                                   Dim1 = n,
                                   p = dim1,
                                   ib = self.IBK1,
                                   ll = self.LLK1,
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
        reduction_time = 0
        return GEMM_time, reduction_time

    def getDistGEMM_b_kp2(self, m, k, n, dim1, dim2, name):
        ######################################################################################
        #calculate grad wrt. weights (A^T. grad(A')) 
        #gather row(A^T)
        reduction_time_wt1 = self.getR(Dim0 = k, 
                                       Dim1 = m,
                                       p = dim1,
                                       ib = self.IBK1,
                                       ll = self.LLK1,
                                       partial = False,
                                       allReduce = False)/2
        #To calculate grad wrt weights (A^T, grad(A')), 
        #gather column grad(A')
        reduction_time_wt2 = self.getR(Dim0 = m, 
                                       Dim1 = n / dim2,
                                       p = dim1,
                                       ib = self.IBK1,
                                       ll = self.LLK1,
                                       partial = False,
                                       allReduce = False)

        ########################################################################################
        #calculate grad wrt. act (grad(A'). w^T)
        #gather row grad(A')
        reduction_time_act1 = self.getR(Dim0 = m / dim1, 
                                        Dim1 = n,
                                        p = dim2,
                                        ib = self.IBK2,
                                        ll = self.LLK2,
                                        partial = False,
                                        allReduce = False)
        #calculate grad wrt. act (grad(A'). w^T)
        #gather col(w^T)
        reduction_time_act2 = self.getR(Dim0 = k, 
                                       Dim1 = n,
                                       p = dim2,
                                       ib = self.IBK2,
                                       ll = self.LLK2,
                                       partial = False,
                                       allReduce = False)/2
        
        reduction_time = reduction_time_wt1 + reduction_time_wt2 + reduction_time_act1 +reduction_time_act2

        #Multiply full grad_activation with shards of weights
        grad_wt_transpose,  grad_wt_time  = self.getGEMMTime(k / dim1, m, n / dim2, name + "wt")
        #Multiply full grad-activation with shards of activations 
        grad_act_transpose, grad_act_time = self.getGEMMTime(m / dim1, n, k / dim2, name + "act")

        GEMM_time = grad_wt_time + grad_act_time

        return GEMM_time, reduction_time

    def getDataParallelReduction(self, k, n, dim1, dim2, name):
        #k = 2 * self.D
        #n = 4 * self.D
        #dim1 = self.kp_hidden_dim1
        #dim2 = self.kp_hidden_dim2

        reduction_time_wt_kp = 0
        reduction_time_wt_dp = 0
        apply_grad_time = 0

        if self.kp_hidden_type == 1: #CR
            reduction_time_wt_kp = 0
            reduction_time_wt_dp = self.getR(Dim0 = k/dim1, 
                                             Dim1 = n,
                                             p = self.dp,
                                             ib = self.IBD,
                                             ll = self.LLD,
                                             partial = False,
                                             allReduce = True)
            apply_grad_time = self.applyGrad(Dim0 = k/dim1, Dim1 = n, name = name)
        
        elif self.kp_hidden_type == 2: #RC
            reduction_time_wt_dp = self.getR(Dim0 = k/dim1, 
                                             Dim1 = n/dim2,
                                             p = self.dp,
                                             ib = self.IBD,
                                             ll = self.LLD,
                                             partial = False,
                                             allReduce = True)

            #gather col(w)
            reduction_time_wt_kp = self.getR(Dim0 = k, 
                                             Dim1 = n/dim2,
                                             p = dim1,
                                             ib = self.IBK1,
                                             ll = self.LLK1,
                                             partial = False,
                                             allReduce = False)
            apply_grad_time = self.applyGrad(Dim0 = k, Dim1 = n/dim2, name = name)
        else:
            reduction_time_wt_kp = 0
            reduction_time_wt_dp = self.getR(Dim0 = k, 
                                             Dim1 = n,
                                             p = self.dp,
                                             ib = self.IBD,
                                             ll = self.LLD,
                                             partial = False,
                                             allReduce = True)
            apply_grad_time = self.applyGrad(Dim0 = k, Dim1 = n, name = name)

        reduction_time =  reduction_time_wt_kp + reduction_time_wt_dp + apply_grad_time 
        return reduction_time
    
    
    def getProjection_f(self):
        transpose, GEMM_time = self.getGEMMTime(self.miniB, self.D, self.projection, "projection")
        return GEMM_time

    def getProjection_b(self):
        grad_wt_transpose,  grad_wt_time  = self.getGEMMTime(self.projection, self.miniB, self.D, "projection_b_wt")
        grad_act_transpose, grad_act_time = self.getGEMMTime(self.miniB, self.projection, self.D, "projection_b_act")

        GEMM_time = grad_wt_time + grad_act_time
        return GEMM_time

    def getProjection_f_kp1(self):
        assert(self.kp_projection_type == 1)
        assert(self.kp_projection_dim1 > 1)
        assert(self.D % self.kp_projection_dim1 == 0)

        GEMM_time, reduction_time = self.getDistGEMM_f_kp1(self.miniB, self.D, self.projection, self.kp_projection_dim1, "projection_f")
        return GEMM_time + reduction_time

    def getProjection_b_kp1(self):
        assert(self.kp_projection_type == 1)
        assert(self.kp_projection_dim1 > 1)
        assert(self.D % self.kp_projection_dim1 == 0)
        
        GEMM_time, reduction_time = self.getDistGEMM_b_kp1(self.miniB, self.D, self.projection, self.kp_projection_dim1, "projection_b")
        return GEMM_time + reduction_time

    def getProjection_f_kp2(self):
        assert(self.kp_projection_type == 2)
        assert(self.kp_projection_dim1 > 1 or self.kp_projection_dim2 > 1)
        assert((self.miniB) % self.kp_projection_dim1 == 0)
        assert(self.projection % self.kp_projection_dim2 == 0)

        GEMM_time, reduction_time = self.getDistGEMM_f_kp2(self.miniB, self.D, self.projection, self.kp_projection_dim1, self.kp_projection_dim2, "projection_f")
        return GEMM_time + reduction_time

    def getProjection_b_kp2(self):
        assert(self.kp_projection_type == 2)
        assert(self.kp_projection_dim1 > 1 or self.kp_projection_dim2 > 1)
        assert((self.miniB) % self.kp_projection_dim1 == 0)
        assert(self.projection % self.kp_projection_dim2 == 0)
        
        GEMM_time, reduction_time = self.getDistGEMM_f_kp2(self.miniB, self.D, self.projection, self.kp_projection_dim1, self.kp_projection_dim2, "projection_b")
        return GEMM_time + reduction_time

    def getSoftmax_f(self):
        transpose, GEMM_time = self.getGEMMTime(self.miniB, (self.projection if proj else self.D), self.V, "softmax_f")

        #Final matrix after GEMM has (B, V) dimensionality
        #We get exponential on each of the elements in a row
        #and then normalize them across the row
        #Therefore for each row we do V sigmoids and V-1 additions and V divisions
        #For each row, we read V sigmoids and write one
        #Up to here is 3 operations
        point_flop = (self.miniB * (3 * self.V - 1))
        
        point_mem  = self.precision * self.miniB * (7 * self.V)
        #2: one read and one write for sigmoid
        #1: one read for reduction
        #1: one write for extension
        #2: for pointwise division

        point_time = self.roofline(point_flop, point_mem, name='pointwise-softmax-f') + 4 * self.O 
   
        if self.debug:
            print("Softmax point_flop: {:,}, point_mem: {:,}".format(int(point_flop/G), int(point_mem/G)))
            print("point_time: {:,}\n".format( point_time))

        return GEMM_time + point_time

    #FIXME: where is the reduction time?
    def getSoftmax_b(self):
        grad_wt_transpose,  grad_wt_time  = self.getGEMMTime((self.projection if proj else self.D), self.miniB, self.V, "softmax_b_wt")
        grad_act_transpose, grad_act_time = self.getGEMMTime(self.miniB, self.V, (self.projection if proj else self.D), "softmax_b_act")

        GEMM_time = grad_wt_time + grad_act_time
        point_flop = self.miniB * self.V * 5
        #1: one for one of the divisions, grad(A) (y=A/B)
        #2: one for division and multiplication, grad(B)
        #1: one for addition, copies turn into add
        #1: one for sigmoid

        point_mem  = self.precision * self.miniB * self.V * 11
        #3: grad(A) in pointwise division
        #3: grad(B) in pointwise division
        #3: addition in copy backprop
        #2: sigmoid

        point_time = self.roofline(point_flop, point_mem, name='pointwise-softmax-b') + 4 * self.O 
    
        if self.debug:
            print("(gr) Softmax point_flop: {:,}, point_mem: {:,}".format(int(point_flop/G), int(point_mem/G)))
            print("(gr) Softmax point_time: {:,}\n".format(point_time))


        return GEMM_time + point_time

    #Column-Row MM
    def getSoftmax_f_kp1(self):
        #Multiply
        assert(self.kp_softmax_type == 1)
        assert(self.kp_softmax_dim1 > 1)
        assert((self.projection if proj else self.D) % self.kp_softmax_dim1 == 0)

        GEMM_time, reduction_time = self.getDistGEMM_f_kp1(self.miniB, self.projection if proj else self.D, self.V, self.kp_softmax_dim1, "softmax_f_kp1")

        #Final matrix after GEMM has (B, V) dimensionality
        #We get exponential on each of the elements in a row
        #and then reduce all elements in the row to one
        #Therefore for each row we do V sigmoids and V-1 additions and V divisions
        #For each row, we read V sigmoids and write one
        #After GEMM reduction, each matrix has the full (B,V) 
        #but each needs to only operate on 1/dim1 rows to get the reduction
        point_flop = (self.miniB / self.kp_softmax_dim1) * self.V * 3
        point_mem  = self.precision * (self.miniB / self.kp_softmax_dim1) * self.V * 7
        #2: sigmoid
        #1: one read for reduction, the accumulate is a register
        #1: one for write/extend the reduction result into all cells 
        #3: division needs one for read and one for write. 

        point_comm = self.getR(Dim0 = self.miniB, 
                               Dim1 = 1,
                               p = self.kp_softmax_dim1,
                               ib = self.IBK1,
                               ll = self.LLK1,
                               partial = False,
                               allReduce = False)
        #communicating partail sum per row from one GPU to all others to perform sum reduce

        point_time = self.roofline(point_flop, point_mem, name='pointwise-softmax-f-kp1') + self.O + point_comm
   
        if self.debug:
            print("Softmax point_flop: {:,}, point_mem: {:,}".format(int(point_flop/G), int(point_mem/G)))
            print("Softmax GEMM_time: {:,}, point_time: {:,}\n".format(GEMM_time, point_time))

        return GEMM_time + reduction_time + point_time

    def getSoftmax_b_kp1(self):

        point_flop = (self.miniB) * (self.V / self.kp_softmax_dim1) * 5
        #1: one for one of the divisions, grad(A) (y=A/B)
        #2: one for division and multiplication, grad(B)
        #1: one for addition, copies turn into add
        #1: one for sigmoid

        point_mem  = self.precision * (self.miniB) * ((11 * self.V )/ self.kp_softmax_dim1)
        #3: grad(A) in pointwise division
        #3: grad(B) in pointwise division
        #3: addition in copy backprop
        #2: sigmoid

        point_comm = 0

        point_time = self.roofline(point_flop, point_mem, name='pointwise-softmax-b-kp1') + self.O + point_comm
   

        
        GEMM_time, reduction_time = self.getDistGEMM_b_kp1(self.miniB, self.projection if proj else self.D, self.V, self.kp_softmax_dim1, "softmax_b_kp1")
        
    
        if self.debug:
            print("(gr) Softmax point_flop: {:,}, point_mem: {:,}\n".format(int(point_flop/G), int(point_mem/G)))
        
        return reduction_time + GEMM_time + point_time

    #Row-Column MM
    def getSoftmax_f_kp2(self):
        #Multiply
        assert(self.kp_softmax_type == 2)
        assert(self.kp_softmax_dim1 > 1 or self.kp_softmax_dim2 > 1)
        assert((self.miniB) % self.kp_softmax_dim1 == 0)
        assert((self.projection if proj else self.D) % self.kp_softmax_dim2 == 0)

        GEMM_time, reduction_time = self.getDistGEMM_f_kp2(self.miniB, self.projection if proj else self.D, self.V, self.kp_softmax_dim1, self.kp_softmax_dim2, "softmax_f_kp2")
        

        #Final matrix after GEMM has (B X S, V) dimensionality
        #We get exponential on each of the elements in a row
        #and then reduce all elements in the row to one
        #Therefore for each row we do V sigmoids and V-1 additions and V divisions
        #For each row, we read V sigmoids and write one
        
        point_flop = (self.miniB / self.kp_softmax_dim1) * (self.V / self.kp_softmax_dim2) * 3
        point_mem  = self.precision * (self.miniB / self.kp_softmax_dim1) * (self.V / self.kp_softmax_dim2) * 7
        #2: sigmoid
        #1: one read for reduction, the accumulate is a register
        #1: one for write/broadcast the reduction result into all cells 
        #3: division needs one for read and one for write. 

        point_comm = self.precision * (self.miniB / self.kp_softmax_dim1) * (self.kp_softmax_dim2) / self.IBK2
        
        point_time = self.roofline(point_flop, point_mem, name='pointwise-Softmax_f_kp2') + self.O + point_comm
  

        if self.debug:
            print("Softmax point_flop: {:,}, point_mem: {:,}".format(int(point_flop/G), int(point_mem/G)))
            print("Softmax GEMM_time: {:,}, point_time: {:,}\n".format(GEMM_time, point_time))

        return GEMM_time + point_time + reduction_time

    def getSoftmax_b_kp2(self):

        point_flop = (self.miniB / self.kp_softmax_dim1) * (self.V / self.kp_softmax_dim2) * 5
        #1: one for one of the divisions, grad(A) (y=A/B)
        #2: one for division and multiplication, grad(B)
        #1: one for addition, copies turn into add
        #1: one for sigmoid

        point_mem  = self.precision * (self.miniB / self.kp_softmax_dim1) * ((11 * self.V )/ self.kp_softmax_dim2)
        #3: grad(A) in pointwise division
        #3: grad(B) in pointwise division
        #3: addition in copy backprop
        #2: sigmoid

        point_comm = 0

        point_time = self.roofline(point_flop, point_mem, name='pointwise-Softmax_b_kp2') + self.O + point_comm
   

        GEMM_time, reduction_time = self.getDistGEMM_b_kp2(self.miniB * self.S, (self.projection if proj else self.D), self.V, self.kp_softmax_dim1, self.kp_softmax_dim2, "softmax_b_kp2")

    
        if self.debug:
            print("(gr) Softmax point_flop: {:,}, point_mem: {:,}\n".format(int(point_flop/G), int(point_mem/G)))
        
        return reduction_time + GEMM_time + point_time


    def getEmbedding_f(self):
        embedding_mem = 2 * (self.miniB * self.D * self.precision)
        #embedding_time = (embedding_mem)/ (self.mem_bw) + self.mem_latency + self.O
        embedding_time = self.roofline(0, embedding_mem, name='embedding_f') + self.O
        if self.debug:
            print("Embedding_mem: {:,}".format(int(embedding_mem/G)))
        return embedding_time


    def getEmbedding_b(self):
        #p2p_data_transfer = (self.precision * self.miniB * self.D)
        #data_transfer_time  = 0 if (self.dp == 1) else (float("inf") if (self.IBD == 0) else (((p2p_data_transfer) / self.IBD + self.LLD) * 2 * (self.dp -1 ))) 
        
        embedding_mem = 2 * self.miniB * self.D * self.precision
        #embedding_mem_time = (embedding_mem / self.mem_bw) + self.mem_latency + self.O
        embedding_mem_time = self.roofline(0, embedding_mem, name='embedding_b') + self.O

        if self.debug:
            print("(gr) Embedding_mem: {:,}".format(int(embedding_mem/G)))
        #return data_transfer_time + embedding_mem_time
        return embedding_mem_time

    def getEmbedding_f_kp1(self):
        #Each GPU has only a portion of the activations since each GPU had only a row of the weights
        reduction_time_act= self.getR(Dim0 = self.miniB, 
                                      Dim1 = self.D,
                                      p = self.kp_embedding_dim1,
                                      ib = self.IBK1,
                                      ll = self.LLK1,
                                      partial = False,
                                      allReduce = True)
        embedding_mem = 2 * (self.miniB * self.D * self.precision)
        #embedding_time = (embedding_mem)/ (self.mem_bw) + self.mem_latency + self.O
        embedding_time = self.roofline(0, embedding_mem, name='embedding_f') + self.O
        if self.debug:
            print("Embedding_mem: {:,}".format(int(embedding_mem/G)))
        return embedding_time + reduction_time_act


    def getEmbedding_b_kp1(self):
        #Activations from previous row arrive in column fasion, they need to be gathered
        #before applying them to the local portion of the embeddings
        reduction_time_act= self.getR(Dim0 = self.miniB, 
                                      Dim1 = self.D,
                                      p = self.kp_embedding_dim1,
                                      ib = self.IBK1,
                                      ll = self.LLK1,
                                      partial = False,
                                      allReduce = False)
        #Each GPU would read through the entire actication and write as many at most as many of B rows
        embedding_mem = 2 * self.miniB * self.D * self.precision
        embedding_mem_time = self.roofline(0, embedding_mem, name='embedding_b') + self.O

        if self.debug:
            print("(gr) Embedding_mem: {:,}".format(int(embedding_mem/G)))
        return embedding_mem_time + reduction_time_act

    def getEmbedding_f_kp2(self):
        embedding_mem = 2 * ((self.miniB / self.kp_embedding_dim1) * (self.D / self.kp_embedding_dim2) * self.precision)
        embedding_time = self.roofline(0, embedding_mem, name='embedding_f') + self.O
        if self.debug:
            print("Embedding_mem: {:,}".format(int(embedding_mem/G)))
        return embedding_time


    def getEmbedding_b_kp2(self):
        #Every GPU will update a little tile of the embedding
        #need to be gathered after the update across the rows of each column
        reduction_time_act= self.getR(Dim0 = self.miniB, 
                                      Dim1 = self.D / self.kp_embedding_dim2,
                                      p = self.kp_embedding_dim1,
                                      ib = self.IBK1,
                                      ll = self.LLK1,
                                      partial = False,
                                      allReduce = False)
        
        embedding_mem = 2 * (self.miniB / self.kp_embedding_dim1) * (self.D / self.kp_embedding_dim2) * self.precision
        embedding_mem_time = self.roofline(0, embedding_mem, name='embedding_b') + self.O

        if self.debug:
            print("(gr) Embedding_mem: {:,}".format(int(embedding_mem/G)))
        return embedding_mem_time + reduction_time_act

    def getInterLayerCommLatency(self, dim1, dim2):
        w = 0
        if self.lp > 1:
          w = self.precision * dim1 * dim2 / self.IBL + self.LLL
        return w


    def dprint(self, string):
        if self.debug:
            print(string)
    
    def readjust_type(self):
      if self.kp_hidden_dim1 == 1 and self.kp_hidden_dim2 == 1:
        self.kp_hidden_type = -1
      
      if self.kp_softmax_dim1 == 1 and self.kp_softmax_dim2 == 1:
        self.kp_softmax_type = -1

      if self.kp_embedding_dim1 == 1 and self.kp_embedding_dim2 == 1:
        self.kp_embedding_type = -1

    def calcTime(self):
        B = self.miniB
        D = self.D
        G = self.G
        L = self.L
        S = self.S
        V = self.V
        lp = self.lp
        dp = self.dp
        
        applyGrad_hidden = self.applyGrad(Dim0 = 2 * D, Dim1 = G * D, name = "Hidden")
        applyGrad_softmax = self.applyGrad(Dim0 = (self.projection if proj else self.D), Dim1 = V, name = "Softmax")
        

        self.readjust_type()

        if self.kp_hidden_type == -1:
            Cf = self.getCf()
            Cb = self.getCb()
            Tf = self.getInterLayerCommLatency(B, D) 
        elif self.kp_hidden_type == 1: #CR
            Cf = self.getCf_kp1()
            Cb = self.getCb_kp1()
            Tf = self.getInterLayerCommLatency(B, D / self.kp_hidden_dim1) 
        elif self.kp_hidden_type == 2: #RC
            Cf = self.getCf_kp2()
            Cb = self.getCb_kp2()
            Tf = self.getInterLayerCommLatency(B / self.kp_hidden_dim1, D / self.kp_hidden_dim2) 
        else:
            print("Incorrect distributed GEMM type, 1: Column-Row, 2: Row-Column")
            sys.exit()
        
        if self.lp == 1:
            Tf = 0
            
        Tb = Tf
       
        if self.kp_softmax_type == -1:
            Sf = self.getSoftmax_f()
            Sb = self.getSoftmax_b()
        elif self.kp_softmax_type == 1: #CR
            Sf = self.getSoftmax_f_kp1()
            Sb = self.getSoftmax_b_kp1()
        elif self.kp_softmax_type== 2: #RC
            Sf = self.getSoftmax_f_kp2()
            Sb = self.getSoftmax_b_kp2()
        else:
            print("Incorrect distributed GEMM type, 1: Column-Row, 2: Row-Column")
            sys.exit()
        
        if self.kp_embedding_type == -1:
            Ef = self.getEmbedding_f()
            Eb = self.getEmbedding_b()
        elif self.kp_embedding_type == 1: #CR
            Ef = self.getEmbedding_f_kp1()
            Eb = self.getEmbedding_b_kp1()
        elif self.kp_embedding_type== 2: #RC
            Ef = self.getEmbedding_f_kp2()
            Eb = self.getEmbedding_b_kp2()
        else:
            print("Incorrect distributed GEMM type, 1: Column-Row, 2: Row-Column")
            sys.exit()
       

        Rc = self.getDataParallelReduction(k = 2 * D, n = G * D, dim1 = self.kp_hidden_dim1, dim2 = self.kp_hidden_dim2, name = "Hidden Reduction")
        Rs = self.getDataParallelReduction(k = D, n = V, dim1 = self.kp_softmax_dim1, dim2 = self.kp_softmax_dim2, name = "Softmax Reduction")
        Re = self.getDataParallelReduction(k = V, n = D, dim1 = self.kp_embedding_dim1, dim2 = self.kp_embedding_dim2, name = "Embedding Reduction")

                        
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

          
            print("Cf: {}, Cb:; {}, softmax_f: {}, softmax_b:{}, embedding_f: {}, embedding_b: {}," 
                   "Rs: {}, Rc:{}, Re: {}\n".format(Cf, 
                                                   Cb,
                                                   Sf, 
                                                   Sb, 
                                                   Ef, 
                                                   Eb,
                                                   Rs,
                                                   Rc,
                                                   Re))


        g = Graph(num_seq = S, num_layer = L, lp = lp, Ef = Ef, Cf = Cf, Sf = Sf, Tf = Tf, Eb = Eb, Cb = Cb, Sb = Sb, Tb = Tb, Re = Re, Rc = Rc, Rs = Rs)
            
        fw_roots = g.construct_fwd_graph()
        bw_roots = g.construct_bwd_graph()

        time_fw = g.simulate(fw_roots[0], 0)
        time_bw = g.simulate(bw_roots[g.num_seq - 1], g.lp + 1)
    
        self.tot_time = time_fw + time_bw
        
        tot_param = self.tot_param()
        #print("#Parameters: {:.2f} Billion\n".format(tot_param/1e9))
    

        return self.tot_time, tot_param

    def getTime(self):
        return self.tot_time

@click.command("standalone")        
@click.option("--exp_config", help="Path to experiment config", required=True)
@click.option("--exp_dir", help="Checkpoint/log directory", required=True)
@click.option("--debug", help="debug", default=False, type=bool)
def main(exp_config, exp_dir, debug):
    debug=False
    exp_path = os.path.expandvars(os.path.expanduser(exp_config))
    exp_config = config.parse_config(exp_path)

    #try:
    #    #print("Removing directory:" + exp_dir)
    #    shutil.rmtree(exp_dir)
    #except:
    #    pass
    #os.makedirs(exp_dir)

    TC = TimeCalculation(exp_config)
    TC.debug = debug
    tot_time, tot_param = TC.calcTime()

    output_file = exp_dir + "/summary.txt"
    
    TC.printSysConfig(exp_config, output_file)

    with open(output_file, "a+") as f:
        f.write("Time: {0:.8f}\n".format(tot_time))
        f.write("Params (Billion): {0:.8f}\n".format(tot_param/1e9))

   
if __name__ == "__main__":
    main()
