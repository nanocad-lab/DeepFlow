#!/tools/lm-venv/py3.6-tf-1.3.0-svail/bin/python

import click
import math
import os
import sys
import config
import shutil
import itertools
import numpy as np

from parallelism import Parallelism
from topology import Topology
from simulate import Graph
import util
from hw_component import Core, MemoryHierarchy, Network
from model import Model

algByte=False #algorithmic ops false
proj=False #consider projection layer, turn off for end-2-end validation, as baeline model does not have projection layer
validating_v100=True

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
        self.FMA_width          = self.core.FMA_width
        self.dataflow          = self.core.dataflow

        self.memoryHierarchy     = MemoryHierarchy(exp_config)
        self.num_levels          = self.memoryHierarchy.num_levels
        self.memLayer            = self.memoryHierarchy.memLayer
        self.tileSpace           = self.generateTileSpace()
      
        #TODO: move this to config file
        self.H2Dbw               = 12.4*1024*1024*1024 

        #System Parameters
        self.num_wafer          = exp_config.system_config.num_wafers
        self.num_workers        = exp_config.system_config.num_workers

        self.network            = Network(exp_config)

        intra_throughput, inter_throughput = self.network.calcThroughput()
        intra_latency, inter_latency       = self.network.calcLatency()
                
        inter_derate            = exp_config.system_config.inter_derate
        intra_derate            = exp_config.system_config.intra_derate
        par2cross               = exp_config.system_config.par2cross
        
        derated_inter_throughput = -1
        derated_intra_throughput = -1
        
        #inter-wafercommunications will pass through intra links too
        if self.num_wafer > 1 and self.num_workers > 1:
          if intra_derate != 0:
            derated_inter_throughput = min(intra_throughput/intra_derate, 
                                           inter_throughput/inter_derate)
          else:
            derated_inter_throughput = inter_throughput/inter_derate
        else:
            derated_inter_throughput = 0

        if self.num_workers > 1 and intra_derate != 0:
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
        self.validating_GEMM    = False
    
    def updateParams(self, debug, m, n, k, t, kp1, kp2, dp, lp, gemm,
                      batch_size, hidden_dim, seq_len, vocab_size, num_layer):

        self.B = batch_size
        self.D = hidden_dim
        self.S = seq_len
        self.V = vocab_size
        self.L = num_layer
        
        #Define miniBatch size
        self.dp = dp if dp != None else self.dp
        self.miniB              = math.ceil(self.B / self.dp)
        
        self.debug = debug
        self.validating_GEMM = gemm
        self.lp = lp if lp != None else self.lp
        self.kp_hidden_dim1 = kp1 if kp1 != None else self.kp_hidden_dim1
        self.kp_hidden_dim1 = kp1 if kp1 != None else self.kp_hidden_dim1
        self.kp_hidden_dim2 = kp2 if kp2 != None else self.kp_hidden_dim2
        self.kp_hidden_type = (2 if t == 'RC' else (1 if t == 'CR' else self.kp_hidden_type))
        
        #TODO: decide if we want kp1, kp2 to control other layers besides hidden layer
        self.kp_softmax_dim1 = kp1 if kp1 != None else self.kp_softmax_dim1
        self.kp_softmax_dim2 = kp2 if kp2 != None else self.kp_softmax_dim2
        self.kp_softmax_type = (2 if t == 'RC' else (1 if t == 'CR' else self.kp_softmax_type))
    
        self.kp_embedding_dim1 = kp1 if kp1 != None else self.kp_embedding_dim1
        self.kp_embedding_dim2 = kp2 if kp2 != None else self.kp_embedding_dim2
        self.kp_embedding_type = (2 if t == 'RC' else (1 if t == 'CR' else self.kp_embedding_type))
        
        self.kp_projection_dim1 = kp1 if kp1 != None else self.kp_projection_dim1
        self.kp_projection_dim2 = kp2 if kp2 != None else self.kp_projection_dim2
        self.kp_projection_type = (2 if t == 'RC' else (1 if t == 'CR' else self.kp_projection_type))
        
        #TODO: need to change all equations to be a function of m,n and k
        #self.D              = n//4
        
        print("kp1: {}".format(self.kp_hidden_dim1))
        print("kp2: {}".format(self.kp_hidden_dim2))
        #TODO: It is a hacky way of capturing assymetry across links within V100
        #move this to network topology and distinguish between inter and intra network
        if validating_v100:
          self.IBK1 = util.scale_down(self.IBK1, self.kp_hidden_dim1, "kp1")
          self.IBK2 = util.scale_down(self.IBK2, self.kp_hidden_dim2, "kp2")
          self.IBD  = util.scale_down(self.IBD, self.dp, "dp")
          self.IBL  = util.scale_down(self.IBL, self.lp, "lp")
             
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
          f.write("==========================\n")
          f.write("Hardware Configuration\n")
          f.write("==========================\n")

          f.write("Throughput: {:.5f} Tflops\n".format(self.core.operating_throughput/1e12))
          for i in range(self.num_levels-1, -1, -1):
              mem_bw    = self.memLayer[i].dynamic_throughput
              mem_size  = self.memLayer[i].size

              if mem_bw < 1e3 * gigaByte:
                  f.write("L{:} Bandwidth: {:.1f} GB/s\n".format(i, mem_bw/(gigaByte)))
              else:
                  f.write("L{:} Bandwidth: {:.1f} TB/s\n".format(i, mem_bw/(teraByte)))

              if mem_size < 1e3 * megaByte:
                  f.write("L{:} Size: {:.10f} MB\n".format(i, mem_size/(megaByte)))
              elif mem_size < 1e3 * gigaByte:
                  f.write("L{:} Size: {:.1f} GB\n".format(i, mem_size/(gigaByte)))
              else:
                  f.write("L{:} Size: {:.1f} TB\n".format(i, mem_size/(teraByte)))
          
          f.write("Intra-node Bandwidth: {:.1f} GB/s\n".format(self.network.intra_network.throughput/(gigaByte)))
          f.write("Inter-node Bandwidth: {:.1f} GB/s\n".format(self.network.inter_network.throughput/(gigaByte)))
          
          M = self.memLayer[self.num_levels - 1].size
          tot_mem, embedding_mem, hidden_mem, softmax_mem, projection_mem, wt_mem, act_mem, point_mem = util.getTotMemReq(exp_config, 
                                                                                                                          batch_size = self.B,
                                                                                                                          hidden_dim = self.D,
                                                                                                                          vocab_size = self.V,
                                                                                                                          seq_len = self.S,
                                                                                                                          num_layer = self.L,
                                                                                                                          dp = self.dp,
                                                                                                                          lp = self.lp,
                                                                                                                          kp1 = self.kp_hidden_dim1,
                                                                                                                          kp2 = self.kp_hidden_dim2,
                                                                                                                          kp_type = self.kp_hidden_type)
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

          
          tot_mem, embedding_mem, hidden_mem, softmax_mem, projection_mem, wt_mem, act_mem, point_mem = util.getMemUsagePerCore(exp_config,
                                                                                                                                batch_size = self.B,
                                                                                                                                hidden_dim = self.D,
                                                                                                                                vocab_size = self.V,
                                                                                                                                seq_len = self.S,
                                                                                                                                num_layer = self.L,
                                                                                                                                dp = self.dp,
                                                                                                                                lp = self.lp,
                                                                                                                                kp1 = self.kp_hidden_dim1,
                                                                                                                                kp2 = self.kp_hidden_dim2,
                                                                                                                                kp_type = self.kp_hidden_type)
          f.write("\n\n===========================================================\n")
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

          f.write("\nMemory Overflow Rate (Total Memory Required per Data Shard Per Model Shard/ Memory capacity per node): {:.5f}\n"
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
          f.write("dp: {}, lp: {}, kp_hidden_dim1: {}, kp_hidden_dim2: {}," 
                  "kp_softmax_dim1: {}, kp_softmax_dim2: {}, kp_embedding1: {}, kp_embedding2: {}," 
                  "kp_projection_dim1: {}, kp_proejction_dim2: {}\n"
                  .format(self.dp, self.lp, self.kp_hidden_dim1, self.kp_hidden_dim2, 
                   self.kp_softmax_dim1, self.kp_softmax_dim2, self.kp_embedding_dim1, self.kp_embedding_dim2,
                   self.kp_projection_dim1, self.kp_projection_dim2))   


          f.write("\n\n==============================================================================\n")
          f.write("Hardware Component Stats\n")
          f.write("==============================================================================\n")
          self.core.printStats(f)
          for i in range(0, self.num_levels):
              self.memLayer[i].printStats(f)

          self.network.printStats(f) 

    def roofline(self, flop, mem_access_, name=''):

        #print("Roofline: entered {}".format(name))
        mem_access = []
        if isinstance(mem_access_, int):
            mem_access.append(mem_access_)
        elif isinstance(mem_access_, float):
            mem_access.append(int(mem_access_))
        elif isinstance(mem_access_, list):
            mem_access = mem_access_
        else:
            print(mem_access_)
            print("mem_access_ should be inetger or list, wrong input", flush=True)
            sys.exit(0)
             

          
        num_level        = len(mem_access)
        time             = [0] * num_level
        comp_int         = [0] * num_level
        inflection_point = [0] * num_level

        try:
            assert(mem_access[num_level - 1] > 0) , "last_level_mem = 0"
        except Exception as e:
            print("{}: Number of accesses to the last level of memory hierarchy cannot be zero:\n {}".format(name, e), flush=True)
            sys.exit(0)

        for i in range(0, num_level):
            time[i]      = 0
            mem_bw       = self.memLayer[i].getThroughput()
            mem_latency  = self.memLayer[i].getLatency()
            num_mem      = mem_access[i] 
            inflection_point[i]   = float("inf") if mem_bw == 0 else self.th / mem_bw
            comp_int[i]  = 0 if num_mem == 0 else flop / num_mem
     
            if comp_int[i] < inflection_point[i]: #mem-bound
                time[i] = (float("inf") if (mem_bw == 0 or num_mem == 0) else (num_mem / mem_bw)) + mem_latency
            else: #compute-bound
                time[i] = float("inf") if (self.th == 0) else (flop / self.th)
        
       
        max_time = max(time)
        
        
        if self.debug:
            print('{}: {}'.format(name, max_time))
            print('GEMM flops: {:,}'.format(flop))
            for i in range(0, num_level):
                print("L{}".format(i))
                print("inflection_point: {:.2f}".format(inflection_point[i]))
                print("comp_int: {:.2f}".format(comp_int[i]))
                print("time: {}".format(time[i]))
                print()
        
        #print("Roofline: exited {}".format(name))
        return max_time

    

    #Convert GEMM into sqaure tiles
   # def getGEMMTime(self, A_, B_, C_, name):
   #     
   #     #A = util.power2RoundUp(A_)
   #     #B = util.power2RoundUp(B_)
   #     #C = util.power2RoundUp(C_)
   #     A = A_
   #     B = B_
   #     C = C_

   #     #return False, self.GEMM_wrapper(A, B, C, name)
   #     dim        = min(min(A, B), C)
   #     Af         = math.ceil(A / dim)
   #     Bf         = math.ceil(B / dim)
   #     Cf         = math.ceil(C / dim)

   #     time       = (Af * Bf * Cf) * self.GEMM_Strassen(dim, name) + (Af * Cf * (Bf-1)) * self.getAddTime(dim, dim, name)
   #     return False, time

   # def GEMM_Strassen(self, dim, name):
   #     if dim <= 512:
   #         time = self.GEMM_wrapper(dim, dim, dim, name)
   #         return time
   #     else:
   #         time = 7 * self.GEMM_Strassen(dim // 2, name) #+ 18 * self.getAddTime(dim // 2, dim // 2, name)
   #         return time
   #  
   # def getAddTime(self, A, B, name):
   #     ADD_flop  = A * B
   #     ADD_gmem  = 3 * A * B * self.precision
   #     ADD_time  = self.roofline(ADD_flop, ADD_gmem, name='FMA addition') + self.O
   #     return ADD_time

    def getGEMMTime(self, dim1, dim2, dim3, name):
       tile2time = {}
       orderSpace = self.generateOrder(dim1, dim2, dim3, name)
       for order_dims in orderSpace:
          if self.debug:
            print("===============================================================")
            print("order: {}".format(order_dims))
            print("===============================================================")
          for tile_dims in self.tileSpace:
            if self.debug:
              print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
              print("tile: {}".format(tile_dims))
              print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            GEMM_flop, mem_access = self.GEMM(order_dims, tile_dims, name)
            GEMM_time = self.roofline(GEMM_flop,mem_access, name) + self.O
            tile2time[(order_dims, tile_dims)] = GEMM_time

       
       best_tile = min(tile2time, key=tile2time.get)
       best_time = tile2time[best_tile]

       if self.debug:
          print("{}: Best Time: {:,}, Best Order: {}, Best Tile: {}\n".format(name, best_time, best_tile[0], best_tile[1]))

       return best_time, best_tile[0], best_tile[1]
    
    def generateOrder(self, dim1, dim2, dim3, name):

        if self.dataflow =="best": # best stationary
           if dim1 >= max(dim2, dim3):
              self.dataflow = "wst"
           elif dim2 >= max(dim1, dim3):
              self.dataflow = "ost"
           elif dim3 >= max(dim1, dim2):
              self.dataflow = "ast"

        order=[]
        if self.dataflow == "wst": #weight stationary
            order.append((dim2, dim3, dim1))
            if dim2 != dim3:
                order.append((dim3, dim2, dim1))
        elif self.dataflow == "ast": #activation stationary
            order.append((dim1, dim2, dim3))
            if dim2 != dim1:
                order.append((dim2, dim1, dim3))
        elif self.dataflow == "ost": #output stationary
            order.append((dim1, dim3, dim2))
            if dim1 != dim3:
                order.append((dim3, dim1, dim2))
        elif self.dataflow == "none": # not stationary
            if dim1 != dim2 and dim2 != dim3 and dim1 != dim3:
                order=list(itertools.permutations([dim1, dim2, dim3]))
            elif dim1 == dim2 and dim2 != dim3:
                order = [(dim1, dim2, dim3), (dim1, dim3, dim2), (dim3, dim1, dim2)]
            elif dim1 == dim3 and dim2 != dim1:
                order = [(dim1, dim2, dim3), (dim1, dim3, dim2), (dim2, dim1, dim3)]
            elif dim2 == dim3 and dim1 != dim2:
                order = [(dim1, dim2, dim3), (dim2, dim1, dim3), (dim2, dim3, dim1)]

        return order

    def generateTileSpace(self):
        tile_space = []
        tiles = [None] * self.num_levels 
        
        for level in range(0, self.num_levels-1):
            memory = self.memLayer[level]
            #tiles[level] = self.getTileDims(memory)
            tiles[level] = memory.getTileDims()
        
        if self.num_levels == 1:
            tile_space = []
        elif self.num_levels == 2:
            tile_space = tiles[0]
        elif self.num_levels == 3:
            tile_space = [(x,y) for x in tiles[0] for y in tiles[1]]
        elif self.num_levels == 4:
            tile_space = [(x,y,z) for x in tiles[0] for y in tiles[1] for z in tiles[2]]
        else: 
          raise NotImplementedError()

        return tile_space

    def getTileSize(self, lid):
        memory = self.memLayer[lid]
        memory.calcTileDim()
        tile_dim  = memory.getTileDim()
        return tile_dim, tile_dim, tile_dim

    #Count the number of accesses from level-1 to level
    # input matrix A(dim1, dim2) and B(dim2, dim3)
    # output matrix C(dim1, dim3)
    def getNumAccesses(self, level, dim1, dim2, dim3, tile_dim, num_repeat, name):
        #tile1,tile2,tile3 = self.getTileSize(level-1)
        tile1, tile2, tile3 = tile_dim

        orig_size = tile1*tile2 + tile1*tile3 + tile2*tile3
        short_tile_cond = [0,0,0]

        if tile1 > dim1:
            tile1 = dim1
            short_tile_cond[0] = 1  
        if tile2 > dim2:
            tile2 = dim2
            short_tile_cond[1] = 1
        if tile3 > dim3:
            tile3 = dim3
            short_tile_cond[2] = 1

        if short_tile_cond[2] == 0 and (short_tile_cond[0] | short_tile_cond[1]) == 1:
            if level <= 1:
              tile3 = math.floor((orig_size - tile1 * tile2) / (tile1 + tile2))
            else:
            #store bypasses cache, directly goes to memory 
              tile3 = math.floor((orig_size - tile1 * tile2) / tile2)
            if tile3 > dim3:
              tile3 = dim3
            #Uncomment if tile3 needs to be pow of 2
            #tile3 = int(math.pow(2, math.floor(math.log2(tile3))))
        elif short_tile_cond[0] == 0 and (short_tile_cond[1] | short_tile_cond[2]) == 1:
            if level <= 1:
              tile1 = math.floor((orig_size - tile3 * tile2) / (tile3 + tile2))
            else:
            #store bypasses cache, directly goes to memory 
              tile1 = math.floor((orig_size - tile3 * tile2) / tile2)
            if tile1 > dim1:
              tile1 = dim1
        elif short_tile_cond[1] == 0 and (short_tile_cond[0] & short_tile_cond[2]) == 1:
            if level <= 1:
              tile2 = math.floor((orig_size - tile3 * tile1) / (tile3 + tile1))
            else:
              tile2 = math.floor((orig_size) / (tile1 + tile3))
            if tile2 > dim2:
              tile2 = dim2

        reload_A = 1
        reload_B = 1
        reload_C = 1

        if tile1 > 0 and tile2 > 0 and tile3 > 0:
           reload_A = math.ceil(dim3 / tile3)
           reload_B = math.ceil(dim1 / tile1)
           #do not access the slow memory on every write,acculmuate in fast memory 
           reload_C = (1 if level > 1 else math.ceil(dim2 / tile2))
           
         
        num_mem = num_repeat * (dim1 * dim2 * reload_A + dim2 * dim3 * reload_B + dim1 * dim3 * reload_C) * self.precision
        if self.debug:
           print(name)
           print("Matrix dimension at Level {}: {:,} x {:,} x {:,}".format(level, dim1, dim2, dim3))
           print("Tile dimension at Level {}: {:,} x {:,} x {:,}".format(level-1, tile1, tile2, tile3))
           print("reload_A: {}, reload_B: {}, reload_C: {}".format(reload_A, reload_B, reload_C))
           print("num_repeat: {}".format(num_repeat))
           print("Bytes Accessed: {:,}".format(num_mem))
           print("")

        return num_mem, tile1, tile2, tile3
        

    #This is the main function that captures the memory hierarchy impact
    #on the number of accesses to global memory considering not everything fits in 
    #L2 cache and also captures the effect of shared memory
    def GEMM(self, order_dims, tile_dims, name):
        dim1_ = order_dims[0]
        dim2_ = order_dims[1]
        dim3_ = order_dims[2]
        #dim1 = util.power2RoundUp(dim1_)
        #dim2 = util.power2RoundUp(dim2_)
        #dim3 = util.power2RoundUp(dim3_)
        dim1 = dim1_
        dim2 = dim2_
        dim3 = dim3_

        GEMM_flop = dim1 * dim3 * (dim2 + dim2 - 1)
        #dim2 multiply
        #dim2-1 add

        #X1 = self.L2_tile_dim
        #X2 = self.shared_mem_tile_dim
        #X3 = self.reg_tile_dim

        num_accesses = [0] * self.num_levels

        if (algByte):
            num_accesses[self.num_levels - 1] = (dim1 * dim2 + dim2 * dim3 + dim1 * dim3) * self.precision
        else:
            num_repeat = 1
            for level in range(self.num_levels - 1, 0, -1):
                num_accesses[level], tile1, tile2, tile3  = self.getNumAccesses(level, dim1, dim2, dim3, tile_dims[level-1], num_repeat, name)
                try:
                  num_repeat           *= math.ceil(dim1/tile1) * math.ceil(dim2/tile2) * math.ceil(dim3/tile3)
                except:
                  num_repeat           *= 1

                dim1                   = tile1 if tile1 != 0 else dim1
                dim2                   = tile2 if tile2 != 0 else dim2
                dim3                   = tile3 if tile3 != 0 else dim3


            #Number of accesses to level0 (for every 2N^3 computation, 3N^2 memory accesses happen, where N is the width of the systolic engine)
            reuse = 1
            dim1 = dim1_
            dim2 = dim2_
            dim3 = dim3_
            
            if self.dataflow == "none":
                reuse = 1
            elif self.dataflow == "best":
                reuse = max(math.ceil(dim1/self.FMA_width), math.ceil(dim3/self.FMA_width), math.ceil(dim2/self.FMA_width))
            elif self.dataflow == "wst": #wt stationary
                reuse = math.ceil(dim1/self.FMA_width)
            elif self.dataflow == "ast": #act statinary
                reuse = math.ceil(dim3/self.FMA_width)
            elif self.dataflow == "ost": #output stationary
                reuse = math.ceil(dim2/self.FMA_width)
            else:
                raise NotImplementedError()
               
            #TODO: make sure to model underutilized systolic array
            #TODO: support FMA_width_x and FMA_width_y
            num_accesses[0]    = GEMM_flop * ((2 * reuse + 1) / (2 * reuse)) * 1/self.FMA_width * self.precision
            #num_accesses[0]    = GEMM_flop * ((2 * reuse + self.FMA_width) / (2 * reuse)) * 1/self.FMA_width * self.precision
             
            #TODO: do we still need these in new hierarchical version?
            #  if X3 == 0:
            #    GEMM_smem  = GEMM_rmem
            #    GEMM_rmem  = 0
            #  if X2 == 0:
            #    GEMM_l2mem = GEMM_smem
            #    GEMM_smem = 0
            #  if X1 == 0:
            #    GEMM_gmem  = GEMM_l2mem
            #    GEMM_l2mem = 0

            #  try:
            #    GEMM_l2mem = GEMM_smem
            #    GEMM_smem = 0
            #  if X1 == 0:
            #    GEMM_gmem  = GEMM_l2mem
            #    GEMM_l2mem = 0

        return GEMM_flop, num_accesses


    #Column-Row MM
    def getCf_kp1(self):
        #Multiply
        assert(self.kp_hidden_type == 1)
        assert(self.kp_hidden_dim1 > 1)
        assert(self.kp_hidden_dim1 % 4 == 0 or self.kp_hidden_dim1 == 2) #4 bc it is LSTM cell
        assert((2 * self.D) % self.kp_hidden_dim1 == 0)
        GEMM_time, reduction_time = self.getDistGEMM_f_kp1(self.miniB, 2 * self.D, self.G * self.D, self.kp_hidden_dim1, "Cf_kp1")
        
       
        #Pointwise ops: all the linear/non-linear ops after MM
        point_flop = self.miniB * (self.G * self.D / self.kp_hidden_dim1) * 5
        #4 refers to the number of pointwise ops (mul + add +tanh + mul + tanh) on 
        #the critical path 
        point_mem  = (self.precision * self.miniB * (self.G * self.D / self.kp_hidden_dim1) *
                     (3 * 3 + 2 * 2 ))
        # 3(3 memory access per operation with two input and one output)
        # 3(mul +  add + mul) on critical path
        # 2(2 memory access per operation with one input and one output)
        # 1(tanh) on critical path
        
        data_size = 4 * self.miniB * (self.G * self.D / self.kp_hidden_dim1) * self.precision
        # 4 refers to the number of pointwise ops (mul + add + mul + tanh) on the
        # critical path whose inputs are located across different GPUs
        #NOTE:Assuming all communications can happpen in parallel
        mem_transfer = self.roofline(0,  2 * data_size, name="Cf_kp1: memory accesses before and after data transfer over network")
        #2:  one read from the source and one write to the destination memory
        data_transfer =  data_size / self.IBK1 
        point_comm = mem_transfer + data_transfer

        point_time = self.roofline(point_flop, point_mem, name='pointwise_cf_kp1') + 5 * self.O + point_comm


        return GEMM_time + reduction_time + point_time
 
    def getCb_kp1(self):
        #TODO:Add local accumulation of weights at every time step
        #Pointwise
        point_flop = ((self.miniB) * (self.G * self.D / self.kp_hidden_dim1) * 5
                     + (2 * self.D * self.G * self.D / self.kp_hidden_dim1)) # local accumulation of wts
        #4 refers to the number of pointwise ops (mul + add +tanh + mul) on 
        #the critical path 
        point_mem  = (self.precision * self.miniB * 
                      (self.G * self.D / self.kp_hidden_dim1) * (3 * 3 + 2 * 2)
                     + (2 * self.precision * self.D * self.G * self.D / self.kp_hidden_dim1) * 3) # local accumulation of wts
        # 3(3 memory access per operation with two input and one output)
        # 3(mul +  add + mul) on critical path
        return GEMM_time + reduction_time + point_time
 
    def getCb_kp1(self):
        #TODO:Add local accumulation of weights at every time step
        #Pointwise
        point_flop = ((self.miniB) * (self.G * self.D / self.kp_hidden_dim1) * 5
                     + (2 * self.D * self.G * self.D / self.kp_hidden_dim1)) # local accumulation of wts
        #4 refers to the number of pointwise ops (mul + add +tanh + mul) on 
        #the critical path 
        point_mem  = (self.precision * self.miniB * 
                      (self.G * self.D / self.kp_hidden_dim1) * (3 * 3 + 2 * 2)
                     + (2 * self.precision * self.D * self.G * self.D / self.kp_hidden_dim1) * 3) # local accumulation of wts
        # 3(3 memory access per operation with two input and one output)
        # 3(mul +  add + mul) on critical path
        # 2(2 memory access per operation with one input and one output)
        # 1(tanh) on critical path
  
        data_size =  4 * self.miniB * (self.G * self.D / self.kp_hidden_dim1) * self.precision
        mem_transfer = self.roofline(0,  2 * data_size, name='Cb_kp1: memory accesses before and after data transfer over network') 
        data_transfer =  data_size / self.IBK1
        point_comm = mem_transfer + data_transfer
        #3 refers to the number of pointwise ops (mul + tanh + mul) on
        # critical path whose inputs are located across different GPUs
        #NOTE:Assuming all communications can happpen in parallel


        point_time = self.roofline(point_flop, point_mem, name='pointwise_Cb_kp1') + 5 * self.O + point_comm

        #GEMM_wrt_act and wt is calculated under getDistGEMM_b_kp1
        GEMM_time, reduction_time = self.getDistGEMM_b_kp1(self.miniB, 2 * self.D, self.G * self.D, self.kp_hidden_dim1, "Cb_kp1")
        
        if self.debug:
            print("(gr) Hidden point_flop: {:,}, point_mem: {:,}\n".format(int(point_flop/1e9), int(point_mem/1e9)))

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
        point_flop = (self.miniB/self.kp_hidden_dim1) * (self.G * self.D / self.kp_hidden_dim2) * 5
        #4 refers to the number of pointwise ops (mul + add +tanh + mul) on 
        #the critical path 
        point_mem  = int(self.precision * (self.miniB / self.kp_hidden_dim1) * 
                        (self.G * self.D / self.kp_hidden_dim2) *
                        (3 * 3 + 2 * 2 ))
        # 3(3 memory access per operation with two input and one output)
        # 3(mul +  add + mul) on critical path
        # 2(2 memory access per operation with one input and one output)
        # 1(tanh) on critical path
        data_size =  ((self.miniB / self.kp_hidden_dim1) * 
                       (self.G * self.D / self.kp_hidden_dim2) * 4 * self.precision)
        #4 refers to the number of pointwise ops (mul + add + tanh + mul) whose inputs
        #across different GPU
        
        point_comm = 0
        if (self.kp_softmax_dim2 > 1):
          mem_transfer = self.roofline(0,  2 * data_size, name='Cf_kp2: memory accesses before and after data transfer over network') 
          data_transfer =  data_size / self.IBK2
          point_comm = mem_transfer + data_transfer

        point_time = self.roofline(point_flop, point_mem, name='pointwise_Cf_kp2') + 5 * self.O + point_comm

        
        return GEMM_time + reduction_time + point_time
    
    def getCb_kp2(self):
      
        #Pointwise ops
        point_flop = ((self.miniB / self.kp_hidden_dim1) * (self.G * self.D / self.kp_hidden_dim2) * 5
                     + (2 * self.D * self.G * self.D / self.kp_hidden_dim2)) # local accumulation of wts
        #4 refers to the number of pointwise ops (mul + add +tanh + mul) on 
        #the critical path 
        # kp_hidden_dim2 is for the reduction sum operation after doing outer product
        # for (B,4D)x(4D,2D).This is outerproduct due to the data distribution.
        point_mem  = int((self.precision * (self.miniB / self.kp_hidden_dim1) * 
                                       (self.G * self.D / self.kp_hidden_dim2) *
                                       (3 * 3 + 2 * 2))
                     + (2 * self.precision * self.D * self.G * self.D / self.kp_hidden_dim2) * 3) # local accumulation of wts
        # 3(3 memory access per operation with two input and one output)
        # 3(mul +  add + mul) on critical path
        # 2(2 memory access per operation with one input and one output)
        # 1(tanh) on critical path
   
        data_size =  int(self.miniB * (self.G * self.D / self.kp_hidden_dim2) * 4 * self.precision)
        #3 refers to the number of pointwise ops (mul + add +tanh + mul) on 
        #3 refers to the number of hops to gather i,f, o and c in each GPU
        #in order to perform (B,4D)x(4D,2D)

        point_comm = 0
        if (self.kp_softmax_dim2 > 1):
          mem_transfer = self.roofline(0,  2 * data_size, name='Cb_kp2:memory accesses before and after data transfer over network') 
          data_transfer =  data_size / self.IBK2
          point_comm = mem_transfer + data_transfer
        
        point_time = self.roofline(point_flop, point_mem, name='pointwise_Cb_kp2') + 5 * self.O + point_comm



        GEMM_time, reduction_time = self.getDistGEMM_b_kp2(self.miniB, 2 * self.D, self.G * self.D, self.kp_hidden_dim1,self.kp_hidden_dim2, "Cb_kp2")
      
        if self.debug:
            print("(gr) Hidden point_flop: {:,}, point_mem: {:,}\n".format(int(point_flop/1e9), int(point_mem/1e9)))

        return GEMM_time + reduction_time + point_time



    def getCf(self, m, n, k):
        #Add Biad adds
        """Get LSTM Cell Time on Forward Path"""
        GEMM_time = self.getGEMMTime(m, k, n, "Cf")

        
        point_flop = m * n * 5
        #1: add bias
        #5: add nonlinearities, there is one more than the number of gates (self.G)
        #1: pointwise muliply and add
        point_mem  = (self.precision * m * n *
                     (3 * 3 + 2 * 2 ))
        #3: 3 memory accesses for operands with two inputs and one output
        #2: 1 for bias add + 1 for pointwise mul
        #2: 2 memory accesses for operands with one input and one output
        #1: 5/4 non-linearities per gate

        point_time = self.roofline(point_flop, point_mem, name='pointwise_Cf') + 5 * self.O

  
        if self.debug:
          gigaByte = 1024 * 1024 * 1024
          print("Hidden point_flop: {:,}, point_mem: {:,}\n".format(int(point_flop/1e9), int(point_mem/gigaByte)))
          print("Hidden point_time: {:,}\n".format(point_time))

        if self.validating_GEMM:
          return GEMM_time
        else:
          return GEMM_time[0] + point_time


   
    def getCb(self):
        """Get LSTM Cell Time on Backward Path"""
        grad_act_time,_,_ = self.getGEMMTime(self. miniB, self.G * self.D, 2 * self.D, "Cb_act") 
        grad_wt_time,_,_   = self.getGEMMTime(2 * self.D, self.miniB, self.G * self.D, "Cb_wt")

        
        GEMM_time = grad_act_time + grad_wt_time

        point_flop = ((self.miniB * self.D * 5) + 
                     (2 * self.D * self.G * self.D)) # local accumulation of wts
        point_mem  = ((self.precision * self.miniB * self.D * (3 * 3 + 2 * 2)) + 
                     (2 * self.precision * self.D * self.G * self.D) * 3) #local accumulation of wts
        point_time = self.roofline(point_flop, point_mem, name='pointwise_Cb') + 5 * self.O
   
        if self.debug:
            print("(gr) Hidden/ point_flop: {:,}, point_mem: {:,} ".format(int(point_flop/1e9), int(point_mem/1e9)))
            print("Hidden point_time: {:,}\n".format(point_time))

        return GEMM_time + point_time


    #Reduction and all-gather time estimation
    def getR(self, Dim0 = None, Dim1 = None, p = None, ib = None, ll = None, partial = None,
            allReduce = None, name = None):
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
        #NOTE: Keep threshold zero to avoid if loop
        threshold = 0
        data_tranfer = 0
        data_prep = 0

        #FIXME: Here I assumed point-2-point links exist across all nodes
        #Implement brodcast timing under ring topology
        if (self.precision * Dim0 * Dim1 < threshold):
            factor = (1/p if partial else 1)
            data_transfer = (((self.precision * Dim0 * Dim1)/ib + ll) * factor if p > 1 else 0)
            data_prep_comp = Dim0 * Dim1 * (p-1) * factor
            data_prep_mem = int((3 * self.precision * Dim0 * Dim1) * (p - 1) * factor)
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
            mem_access  = self.roofline(0, int(2 * self.precision * Dim0 * Dim1 / p), name='Reduction: memory accesses before and after data transfer over network')
            data_transfer  = float("inf") if (ib == 0) else ((((self.precision * Dim0 * Dim1) / p) / ib) + mem_access + ll) * factor * (p - 1)
            #dt = ((self.precision * Dim0 * Dim1) / p) * factor * (p - 1)
            
            #First round accumlates the updates as going around the ring
            data_prep_comp = (Dim0 * Dim1) / p
            data_prep_mem  = int(3 * self.precision * Dim0 * Dim1 / p)
            data_prep = ((self.roofline(data_prep_comp, data_prep_mem, name='R-prepTime') + self.O) * (p - 1))
            
            #all-gather-concat

            data_concat_mem = 3 * Dim0 * Dim1 * self.precision
            concat_time = ((self.roofline(0, data_concat_mem, name='all-gather-concat') + self.O))

            
            #print("R1: {}, factor: {}\n".format(dt,factor))
        if self.debug:
            print("Bandwidth: {:,} GB/s".format(ib/(1024*1024*1024)))
            print("data_transfer_time: {:,}, data_prep_time: {:,}, concat_time: {:,}".format(data_transfer, (data_prep if allReduce else 0), (concat_time if not allReduce else 0)))
            print("(data_prep) allReduce_flop: {:,}, allReduce_mem: {:,}".format(int(data_prep_comp), int(data_prep_mem)))
            print("(data_transfer) {:,}".format(int(self.precision * Dim0 * Dim1 / (p))))

        return data_transfer + (data_prep if allReduce else 0) + (concat_time if not allReduce else 0)
    
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
        applyGrad_mem = ((1 * Dim0 * Dim1 * self.precision) +
                         (2 * Dim0 * Dim1 * self.precision) +
                         (1 * Dim0 * Dim1 * self.precision))
        #1: read for pointiwse div
        #2: 1 reads and one write for pointwise add
        #1: one write for multiplication by lr
        applyGrad_time = self.roofline(applyGrad_comp, applyGrad_mem, name='pointwise-applyGrad')
       
        clip_time = self.gradClipping(Dim0, Dim1, name)
        
        grad_time = applyGrad_time + clip_time

        if self.debug:
            print("({}) applyGrad_flop: {:,}, applyGrad_mem: {:,}".format(name, applyGrad_comp, applyGrad_mem))
            print("({}) applyGrad_time: {:,}\n".format(name, applyGrad_time))
       
        return grad_time

    def getDistGEMM_f_kp1(self, m, k, n, dim1, name):
        GEMM_time = self.getGEMMTime(m , k//dim1, n , name)
        
        #Sum-Reduce within each row for use in the next time step
        reduction_time = self.getR(Dim0 = m, 
                                   Dim1 = n,
                                   p = dim1,
                                   ib = self.IBK1,
                                   ll = self.LLK1,
                                   partial = True,
                                   allReduce = True,
                                   name = name)
        if self.validating_GEMM:
          print("GEMM_time: {}, Reduction_time:{}".format(GEMM_time[0], reduction_time))
          return GEMM_time[0] + reduction_time, GEMM_time[1], GEMM_time[2]
        else:
          return GEMM_time[0], reduction_time


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
                                   allReduce = False,
                                   name = name)

        #Multiply full grad_activation with shards of weights
        grad_wt_time,_,_  = self.getGEMMTime(k, (m // dim1), n, name + "wt")
        #Multiply full grad-activation with shards of activations 
        grad_act_time,_,_ = self.getGEMMTime(m, (n // dim1), k, name + "act")


        GEMM_time = grad_wt_time + grad_act_time

        return GEMM_time, reduction_time

    def getDistGEMM_f_kp2(self, m, k, n, dim1, dim2, name):
        GEMM_time = self.getGEMMTime(m // dim1, k, n // dim2, name)
        reduction_time = self.getR(Dim0 = m // dim1, 
                                   Dim1 = n,
                                   p = dim2,
                                   ib = self.IBK2,
                                   ll = self.LLK2,
                                   partial = False,
                                   allReduce = False,
                                   name = name)
        if self.validating_GEMM:
          print("GEMM_time: {}, Reduction_time:{}".format(GEMM_time[0], reduction_time))
          return GEMM_time[0] + reduction_time, GEMM_time[1], GEMM_time[2]
        else:
          return GEMM_time[0], reduction_time

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
                                       allReduce = False,
                                       name = name)/2
        #To calculate grad wrt weights (A^T, grad(A')), 
        #gather column grad(A')
        reduction_time_wt2 = self.getR(Dim0 = m, 
                                       Dim1 = n / dim2,
                                       p = dim1,
                                       ib = self.IBK1,
                                       ll = self.LLK1,
                                       partial = False,
                                       allReduce = False,
                                       name = name)

        ########################################################################################
        #calculate grad wrt. act (grad(A'). w^T)
        #gather row grad(A')
        reduction_time_act1 = self.getR(Dim0 = m / dim1, 
                                        Dim1 = n,
                                        p = dim2,
                                        ib = self.IBK2,
                                        ll = self.LLK2,
                                        partial = False,
                                        allReduce = False,
                                        name = name)
        #calculate grad wrt. act (grad(A'). w^T)
        #gather col(w^T)
        reduction_time_act2 = self.getR(Dim0 = k, 
                                       Dim1 = n,
                                       p = dim2,
                                       ib = self.IBK2,
                                       ll = self.LLK2,
                                       partial = False,
                                       allReduce = False,
                                       name = name)/2
        
        reduction_time = reduction_time_wt1 + reduction_time_wt2 + reduction_time_act1 +reduction_time_act2

        #Multiply full grad_activation with shards of weights
        grad_wt_time,_,_  = self.getGEMMTime(k / dim1, m, n / dim2, name + "wt")
        #Multiply full grad-activation with shards of activations 
        grad_act_time,_,_ = self.getGEMMTime(m / dim1, n, k / dim2, name + "act")

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
                                             allReduce = True,
                                             name = name)
            apply_grad_time = self.applyGrad(Dim0 = k/dim1, Dim1 = n, name = name)
        
        elif self.kp_hidden_type == 2: #RC
            reduction_time_wt_dp = self.getR(Dim0 = k/dim1, 
                                             Dim1 = n/dim2,
                                             p = self.dp,
                                             ib = self.IBD,
                                             ll = self.LLD,
                                             partial = False,
                                             allReduce = True,
                                             name = name)

            #gather col(w)
            reduction_time_wt_kp = self.getR(Dim0 = k, 
                                             Dim1 = n/dim2,
                                             p = dim1,
                                             ib = self.IBK1,
                                             ll = self.LLK1,
                                             partial = False,
                                             allReduce = False,
                                             name = name)
            apply_grad_time = self.applyGrad(Dim0 = k, Dim1 = n/dim2, name = name)
        else:
            reduction_time_wt_kp = 0
            reduction_time_wt_dp = self.getR(Dim0 = k, 
                                             Dim1 = n,
                                             p = self.dp,
                                             ib = self.IBD,
                                             ll = self.LLD,
                                             partial = False,
                                             allReduce = True,
                                             name = name)
            apply_grad_time = self.applyGrad(Dim0 = k, Dim1 = n, name = name)

        reduction_time =  reduction_time_wt_kp + reduction_time_wt_dp + apply_grad_time 
        return reduction_time
    
    
    def getProjection_f(self):
        GEMM_time,_,_ = self.getGEMMTime(self.miniB, self.D, self.projection, "projection")
        return GEMM_time

    def getProjection_b(self):
        grad_wt_time,_,_  = self.getGEMMTime(self.projection, self.miniB, self.D, "projection_b_wt")
        grad_act_time,_,_ = self.getGEMMTime(self.miniB, self.projection, self.D, "projection_b_act")

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
        GEMM_time,_,_ = self.getGEMMTime(self.miniB, (self.projection if proj else self.D), self.V, "softmax_f")

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
            print("Softmax point_flop: {:,}, point_mem: {:,}".format(int(point_flop/1e9), int(point_mem/1e9)))
            print("point_time: {:,}\n".format( point_time))

        return GEMM_time + point_time

    #FIXME: where is the reduction time?
    def getSoftmax_b(self):
        grad_wt_time,_,_  = self.getGEMMTime((self.projection if proj else self.D), self.miniB, self.V, "softmax_b_wt")
        grad_act_time,_,_ = self.getGEMMTime(self.miniB, self.V, (self.projection if proj else self.D), "softmax_b_act")

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
            print("(gr) Softmax point_flop: {:,}, point_mem: {:,}".format(int(point_flop/1e9), int(point_mem/1e9)))
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
                               allReduce = False,
                               name="getSoftmax_f_kp1")
        #communicating partail sum per row from one GPU to all others to perform sum reduce

        point_time = self.roofline(point_flop, point_mem, name='pointwise-softmax-f-kp1') + self.O + point_comm
   
        if self.debug:
            print("Softmax point_flop: {:,}, point_mem: {:,}".format(int(point_flop/1e9), int(point_mem/1e9)))
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
            print("(gr) Softmax point_flop: {:,}, point_mem: {:,}\n".format(int(point_flop/1e9), int(point_mem/1e9)))
        
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

        data_size = self.precision * (self.miniB / self.kp_softmax_dim1) * (self.kp_softmax_dim2)
       
        point_comm = 0
        if (self.kp_softmax_dim2 > 1):
          mem_transfer = self.roofline(0,  2 * data_size, name='memory accesses before and after data transfer over network') 
          data_transfer =  data_size / self.IBK2
          point_comm = mem_transfer + data_transfer

        point_time = self.roofline(point_flop, point_mem, name='pointwise-Softmax_f_kp2') + self.O + point_comm
  

        if self.debug:
            print("Softmax point_flop: {:,}, point_mem: {:,}".format(int(point_flop/1e9), int(point_mem/1e9)))
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
   

        GEMM_time, reduction_time = self.getDistGEMM_b_kp2(self.miniB, self.projection if proj else self.D, self.V, self.kp_softmax_dim1, self.kp_softmax_dim2, "softmax_b_kp2")
    
        if self.debug:
            print("(gr) Softmax point_flop: {:,}, point_mem: {:,}\n".format(int(point_flop/1e9), int(point_mem/1e9)))
        
        return reduction_time + GEMM_time + point_time


    def getEmbedding_f(self):
        embedding_mem = 2 * (self.miniB * self.D * self.precision)
        #embedding_time = (embedding_mem)/ (self.mem_bw) + self.mem_latency + self.O
        embedding_time = self.roofline(0, embedding_mem, name='embedding_f') + self.O
        embedding_transfer_time = 2 * self.miniB * self.D * self.precision / self.H2Dbw
        if self.debug:
            print("Embedding_mem: {:,}".format(int(embedding_mem/1e9)))
        return embedding_time + embedding_transfer_time


    def getEmbedding_b(self):
        #p2p_data_transfer = (self.precision * self.miniB * self.D)
        #data_transfer_time  = 0 if (self.dp == 1) else (float("inf") if (self.IBD == 0) else (((p2p_data_transfer) / self.IBD + self.LLD) * 2 * (self.dp -1 ))) 
        
        embedding_mem = 2 * self.miniB * self.D * self.precision
        #embedding_mem_time = (embedding_mem / self.mem_bw) + self.mem_latency + self.O
        embedding_mem_time = self.roofline(0, embedding_mem, name='embedding_b') + self.O

        if self.debug:
            print("(gr) Embedding_mem: {:,}".format(int(embedding_mem/1e9)))
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
                                      allReduce = False,
                                      name="getEmbedding_f_kp1")
        embedding_mem = 2 * (self.miniB * self.D * self.precision)
        #embedding_time = (embedding_mem)/ (self.mem_bw) + self.mem_latency + self.O
        embedding_time = self.roofline(0, embedding_mem, name='embedding_f') + self.O
        if self.debug:
            print("Embedding_mem: {:,}".format(int(embedding_mem/1e9)))
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
                                      allReduce = False,
                                      name="getEmbedding_b_kp1")
        #Each GPU would read through the entire actication and write as many at most as many of B rows
        embedding_mem = 2 * self.miniB * self.D * self.precision
        embedding_mem_time = self.roofline(0, embedding_mem, name='embedding_b') + self.O

        if self.debug:
            print("(gr) Embedding_mem: {:,}".format(int(embedding_mem/1e9)))
        return embedding_mem_time + reduction_time_act

    def getEmbedding_f_kp2(self):
        embedding_mem = 2 * ((self.miniB / self.kp_embedding_dim1) * (self.D / self.kp_embedding_dim2) * self.precision)
        embedding_time = self.roofline(0, embedding_mem, name='embedding_f') + self.O
        if self.debug:
            print("Embedding_mem: {:,}".format(int(embedding_mem/1e9)))
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
                                      allReduce = False,
                                      name="getEmbedding_b_kp2")
        
        embedding_mem = 2 * (self.miniB / self.kp_embedding_dim1) * (self.D / self.kp_embedding_dim2) * self.precision
        embedding_mem_time = self.roofline(0, embedding_mem, name='embedding_b') + self.O

        if self.debug:
            print("(gr) Embedding_mem: {:,}".format(int(embedding_mem/1e9)))
        return embedding_mem_time + reduction_time_act

    def getInterLayerCommLatency(self, dim1, dim2):
        w = 0
        if self.lp > 1:
          w_size         = self.precision * dim1 * dim2
          transfer_time  = w_size / self.IBL + self.LLL
          mem_time       = self.roofline(0, 2 * w_size, name='inter_layer')
          #2: read from memory of previous layer and write to the memory of the next layer
          w              = mem_time + transfer_time
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

        self.readjust_type()

        if self.kp_hidden_type == -1:
            Cf = self.getCf(m = B, k = 2*D,  n = G*D)
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
                          self.kp_hidden_dim1, 
                          self.kp_hidden_dim2, 
                          self.kp_softmax_dim1, 
                          self.kp_softmax_dim2, 
                          self.kp_embedding_dim1, 
                          self.kp_embedding_dim2, 
                          self.kp_hidden_type, 
                          self.kp_softmax_type, 
                          self.kp_embedding_type))

          
            print("Cf: {} Cb: {} softmax_f: {} softmax_b: {} embedding_f: {} embedding_b: {} " 
                   "Rs: {} Rc: {} Re: {}\n".format(Cf, 
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
        time_bw = g.simulate(bw_roots[g.num_seq - 1], g.lp - 1)
       
        self.tot_time  = time_fw + time_bw
        tot_param = self.tot_param()
        
        return self.tot_time, tot_param

    def getTime(self):
        return self.tot_time
def callPerf(exp_config, exp_dir, debug):
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

@click.command("standalone")
@click.option("--args_input", help="Shall it read the args from the input command (True) or from exp_config (False)", default=False, type=bool, required=False)
@click.option("--exp_config", help="Path to experiment config", required=True)
@click.option("--exp_dir", help="Checkpoint/log directory", required=True)
@click.option("--debug", help="debug", default=False, type=bool)
@click.option("--m", help="input dimension", default=32768, type=int, required=False) #only use for GEMM validation. This allows arbitrary choice of dimension. For LSTM, dimensions are fixed at m=mini_batch, k=2*D and n=4*D.
@click.option("--n", help="output dimension", default=32768, type=int, required=False) #only use for GEMM validation
@click.option("--k", help="input dimension", default=32768, type=int, required=False) #only use for GEMM validation
@click.option("--t", help="parallelism strategy (RC or CR)", default='None', type=str, required=False) #only use for GEMM validation
@click.option("--kp1", help="RC:parallelism along input dimension, CR: parallelism along inner dimension", default=None, type=int, required=False) #only use for GEMM validation
@click.option("--kp2", help="RC:parallelism along output dimension", default=None, type=int, required=False) #only use for GEMM validation
@click.option("--gemm", help="report ONLY GEMM time", default=False, type=bool, required=False) #only use for GEMM validation
@click.option("--batch_size", help="Total Batch Size", default=2048, type=int, required=False)
@click.option("--hidden_dim", help="Hidden Dimension per LSTM layer", default=19968, type=int, required=False)
@click.option("--seq_len", help="Number of times to unroll LSTM", default=20, type=int, required=False)
@click.option("--vocab_size", help="Vocabulary Size", default=800000, type=int, required=False)
@click.option("--num_layer", help="number of lstm layers", default=2, type=int, required=False)
@click.option("--dp", help="data parallelism", default=None, type=int, required=False) #only use for GEMM validation
@click.option("--lp", help="layer parallelism", default=None, type=int, required=False) #only use for GEMM validation

def main(exp_config, exp_dir, debug, m, n, k, t, kp1, kp2, gemm, batch_size, hidden_dim, seq_len, vocab_size, num_layer, dp, lp, args_input=False):
    exp_path = os.path.expandvars(os.path.expanduser(exp_config))
    exp_config = config.parse_config(exp_path)
    output_file = exp_dir + "/summary.txt" ##Output dir should be created manually


    TC = TimeCalculation(exp_config)
    if args_input:
        TC.updateParams(debug, m, n, k, t, kp1, kp2, dp, lp, gemm, 
                    batch_size, hidden_dim, seq_len, vocab_size, num_layer)

    #Report GEMM time on fw path
    if TC.validating_GEMM:
        
        if kp1 == 1 and kp2 ==1: #no parallelism
          gemm_time = TC.getCf(m, k, n)
        elif t == 'CR':
          gemm_time = TC.getDistGEMM_f_kp1(m, k, n, kp1, "Cf_CR")
        elif t == 'RC':
          gemm_time = TC.getDistGEMM_f_kp2(m, k, n, kp1, kp2, "Cf_RC")
        else:
          print("Incorrect parallelism type, CR: Column-Row, RC: Row-Column")
          sys.exit()
          
        with open(output_file, "w") as f:
          f.write("Best Order: {}\n".format(gemm_time[1]))
          f.write("Best Tile: {}\n".format(gemm_time[2]))
          f.write("Time: {}\n".format(gemm_time[0]))
        return

    tot_time, tot_param = TC.calcTime()
    TC.printSysConfig(exp_config, output_file)
    
    with open(output_file, "a+") as f:
        f.write("\n\n==============================================\n")
        f.write("Performance Results\n")
        f.write("==============================================\n")
        f.write("Time: {0:.8f}\n".format(tot_time))
        f.write("Params (Billion): {0:.8f}\n".format(tot_param/1e9))

   
if __name__ == "__main__":
    main()
