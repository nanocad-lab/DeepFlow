#!/tools/lm-venv/py3.6-tf-1.3.0-svail/bin/python

import click
import math
import os
import sys
import config
import shutil
import itertools
# import numpy as np

from parallelism import Parallelism
from topology import Topology
from simulate_LLM import Graph
import util
from hw_component import Core, MemoryHierarchy, Network
from model import Model_LSTM, Model_GEMM, Model_LLM 
from tile import TiledGEMM, formatBytes
from LLM_util import process_gemm_shapes
from simulate_LLM import visualize_graph
algByte = False  # algorithmic ops false
proj = False  # consider projection layer, turn off for end-2-end validation, as baeline model does not have projection layer
validating_v100 = True


class TimeCalculationLLM:
    def __init__(self, hw_config, model_config, mode):
# Mode parameter
        

        # Software Parameters
        self.O = hw_config.sw_config.kernel_launch_overhead
        self.precision = hw_config.sw_config.precision
        self.attached = True

        # Hardware Parameters
        self.core = Core(hw_config)
        self.th = self.core.getThroughput()
        self.FMA_dims = self.core.FMA_dims  # (FMA_x, FMA_y)
        self.dataflow = self.core.dataflow

        self.memoryHierarchy = MemoryHierarchy(hw_config)
        self.num_levels = self.memoryHierarchy.num_levels
        self.memLayer = self.memoryHierarchy.memLayer
        self.tileSpace = self.generateTileSpace()

        # TODO: move this to config file
        self.H2Dbw = 12.4 * 1024 * 1024 * 1024


        # System Parameters
        self.num_wafer = hw_config.system_config.num_wafers
        self.num_workers = hw_config.system_config.num_workers

        self.network = Network(hw_config)

        intra_throughput, inter_throughput = self.network.calcThroughput()
        intra_latency, inter_latency = self.network.calcLatency()

        inter_derate = hw_config.system_config.inter_derate
        intra_derate = hw_config.system_config.intra_derate
        par2cross = hw_config.system_config.par2cross

        derated_inter_throughput = -1
        derated_intra_throughput = -1

        # inter-wafer communications will pass through intra links too
        if self.num_wafer > 1 and self.num_workers > 1:
            if intra_derate != 0:
                derated_inter_throughput = min(
                    intra_throughput / intra_derate, inter_throughput / inter_derate
                )
            else:
                derated_inter_throughput = inter_throughput / inter_derate
        else:
            derated_inter_throughput = 0

        if self.num_workers > 1 and intra_derate != 0:
            derated_intra_throughput = intra_throughput / intra_derate
        else:
            derated_intra_throughput = 0

        self.IBK1, self.LLK1 = (
            (derated_inter_throughput, inter_latency)
            if par2cross["kp1"]
            else (derated_intra_throughput, intra_latency)
        )
        self.IBK2, self.LLK2 = (
            (derated_inter_throughput, inter_latency)
            if par2cross["kp2"]
            else (derated_intra_throughput, intra_latency)
        )
        self.IBD, self.LLD = (
            (derated_inter_throughput, inter_latency)
            if par2cross["dp"]
            else (derated_intra_throughput, intra_latency)
        )
        self.IBL, self.LLL = (
            (derated_inter_throughput, inter_latency)
            if par2cross["lp"]
            else (derated_intra_throughput, intra_latency)
        )

        # Scheduling Parameters
        par = Parallelism(hw_config)
        par.findParallelStrategy()
        self.autoPar = par.autoPar
        self.lp = par.lp
        self.kp_hidden_dim1 = par.kp_hidden_dim1
        self.kp_softmax_dim1 = par.kp_softmax_dim1
        self.kp_embedding_dim1 = par.kp_embedding_dim1
        self.kp_projection_dim1 = par.kp_projection_dim1
        self.kp_hidden_dim2 = par.kp_hidden_dim2
        self.kp_softmax_dim2 = par.kp_softmax_dim2
        self.kp_embedding_dim2 = par.kp_embedding_dim2
        self.kp_projection_dim2 = par.kp_projection_dim2
        self.dp = par.dp
        self.kp_hidden_type = par.kp_hidden_type  # 1: CR, 2: RC
        self.kp_softmax_type = par.kp_softmax_type  # 1: CR, 2: RC
        self.kp_embedding_type = par.kp_embedding_type  # 1: CR, 2: RC
        self.kp_projection_type = par.kp_projection_type  # 1: CR, 2: RC
        self.t = par.t  # type of parallelism, e.g., "CR", "RC", "none"
        self.kp1= par.kp1  # first parallelism parameter
        self.kp2 = par.kp2  # second parallelism parameter
        
        self.updateParParams(self.t, self.kp1, self.kp2)
        # # Define miniBatch size
        # self.miniB = math.ceil(self.B / self.dp)

        # Statistics Param
        self.tot_flop = 0
        self.tot_mem = 0
        self.tot_time = 0
        self.debug = False
        self.validating_GEMM = False
        
        self.mode = mode
        
        # Dynamically select and instantiate the model class
        model_class = self.get_model_class(mode)
        self.model = model_class(model_config)  # Instantiate the model class


        # Model Parameters
        # self.model = self.get_model_class(mode)
        if mode == "LSTM":
            self.B = self.model.batch_size
            self.V = self.model.vocab_size
            self.L = self.model.num_layers
            self.D = self.model.hidden_dim
            self.projection = self.model.projection
            self.S = self.model.seq_len
            self.G = self.model.num_gates
            self.NL = self.model.num_non_linear
            self.A = self.model.num_add
            self.P = self.model.num_pointwise
            # Define miniBatch size
            self.miniB = math.ceil(self.B / self.dp)

        if mode == "GEMM":
            
            self.M = self.model.M
            self.K = self.model.K
            self.N = self.model.N
            
        if mode == "LLM":
            self.batch_size = self.model.batch_size
            self.vocab_size = self.model.vocab_size
            self.num_layers = self.model.num_layers
            self.hidden_dim = self.model.hidden_dim
            self.seq_len = self.model.seq_len
            self.num_heads = self.model.num_heads
            self.ffn_mult = self.model.ffn_mult
            if self.ffn_mult is not None:
                self.ffn_dim = self.model.hidden_dim * self.ffn_mult
            else:
                self.ffn_dim = self.model.ffn_dim
            self.n_tokens = self.model.n_tokens
            self.communication_time = self.model.communication_time
            self.N_PP = self.model.N_PP
            self.miniB = math.ceil(self.batch_size / self.dp)
            
    
    def get_model_class(self, model_type):
        """Return the appropriate model class based on the model type."""
        model_classes = {
            "LSTM": Model_LSTM,
            "GEMM": Model_GEMM,
            "LLM": Model_LLM, 
            # Add other model types here as needed
        }
        if model_type not in model_classes:
            raise ValueError(f"Unsupported model type: {model_type}")
        return model_classes[model_type]
        
    def updateParParams(
        self,
        t,
        kp1,
        kp2,
    ):

        self.kp_hidden_dim1 = kp1 if kp1 != None else self.kp_hidden_dim1
        # self.kp_hidden_dim1 = kp1 if kp1 != None else self.kp_hidden_dim1
        self.kp_hidden_dim2 = kp2 if kp2 != None else self.kp_hidden_dim2
        self.kp_hidden_type = (
            2 if t == "RC" else (1 if t == "CR" else self.kp_hidden_type)
        )

        # TODO: decide if we want kp1, kp2 to control other layers besides hidden layer
        self.kp_softmax_dim1 = kp1 if kp1 != None else self.kp_softmax_dim1
        self.kp_softmax_dim2 = kp2 if kp2 != None else self.kp_softmax_dim2
        self.kp_softmax_type = (
            2 if t == "RC" else (1 if t == "CR" else self.kp_softmax_type)
        )

        self.kp_embedding_dim1 = kp1 if kp1 != None else self.kp_embedding_dim1
        self.kp_embedding_dim2 = kp2 if kp2 != None else self.kp_embedding_dim2
        self.kp_embedding_type = (
            2 if t == "RC" else (1 if t == "CR" else self.kp_embedding_type)
        )

        self.kp_projection_dim1 = kp1 if kp1 != None else self.kp_projection_dim1
        self.kp_projection_dim2 = kp2 if kp2 != None else self.kp_projection_dim2
        self.kp_projection_type = (
            2 if t == "RC" else (1 if t == "CR" else self.kp_projection_type)
        )

        # TODO: need to change all equations to be a function of m,n and k
        # self.D              = n//4

        print("kp1: {}".format(self.kp_hidden_dim1))
        print("kp2: {}".format(self.kp_hidden_dim2))
        # TODO: It is a hacky way of capturing assymetry across links within V100
        # move this to network topology and distinguish between inter and intra network
        if validating_v100:
            self.IBK1 = util.scale_down(self.IBK1, self.kp_hidden_dim1, "kp1")
            self.IBK2 = util.scale_down(self.IBK2, self.kp_hidden_dim2, "kp2")
            self.IBD = util.scale_down(self.IBD, self.dp, "dp")
            self.IBL = util.scale_down(self.IBL, self.lp, "lp")
            
    def updateParams(
        self,
        debug,
        m,
        n,
        k,
        t,
        kp1,
        kp2,
        dp,
        lp,
        gemm,
        batch_size,
        hidden_dim,
        seq_len,
        vocab_size,
        num_layer,
    ):
        self.B = batch_size
        self.D = hidden_dim
        self.S = seq_len
        self.V = vocab_size
        self.L = num_layer

        # Define miniBatch size
        self.dp = dp if dp != None else self.dp
        self.miniB = math.ceil(self.B / self.dp)

        self.debug = debug
        self.validating_GEMM = gemm
        self.lp = lp if lp != None else self.lp
        self.kp_hidden_dim1 = kp1 if kp1 != None else self.kp_hidden_dim1
        # self.kp_hidden_dim1 = kp1 if kp1 != None else self.kp_hidden_dim1
        self.kp_hidden_dim2 = kp2 if kp2 != None else self.kp_hidden_dim2
        self.kp_hidden_type = (
            2 if t == "RC" else (1 if t == "CR" else self.kp_hidden_type)
        )

        # TODO: decide if we want kp1, kp2 to control other layers besides hidden layer
        self.kp_softmax_dim1 = kp1 if kp1 != None else self.kp_softmax_dim1
        self.kp_softmax_dim2 = kp2 if kp2 != None else self.kp_softmax_dim2
        self.kp_softmax_type = (
            2 if t == "RC" else (1 if t == "CR" else self.kp_softmax_type)
        )

        self.kp_embedding_dim1 = kp1 if kp1 != None else self.kp_embedding_dim1
        self.kp_embedding_dim2 = kp2 if kp2 != None else self.kp_embedding_dim2
        self.kp_embedding_type = (
            2 if t == "RC" else (1 if t == "CR" else self.kp_embedding_type)
        )

        self.kp_projection_dim1 = kp1 if kp1 != None else self.kp_projection_dim1
        self.kp_projection_dim2 = kp2 if kp2 != None else self.kp_projection_dim2
        self.kp_projection_type = (
            2 if t == "RC" else (1 if t == "CR" else self.kp_projection_type)
        )

        # TODO: need to change all equations to be a function of m,n and k
        # self.D              = n//4

        print("kp1: {}".format(self.kp_hidden_dim1))
        print("kp2: {}".format(self.kp_hidden_dim2))
        # TODO: It is a hacky way of capturing assymetry across links within V100
        # move this to network topology and distinguish between inter and intra network
        if validating_v100:
            self.IBK1 = util.scale_down(self.IBK1, self.kp_hidden_dim1, "kp1")
            self.IBK2 = util.scale_down(self.IBK2, self.kp_hidden_dim2, "kp2")
            self.IBD = util.scale_down(self.IBD, self.dp, "dp")
            self.IBL = util.scale_down(self.IBL, self.lp, "lp")

    def readjust_type(self):
        if self.kp_hidden_dim1 == 1 and self.kp_hidden_dim2 == 1:
            self.kp_hidden_type = -1

        if self.kp_softmax_dim1 == 1 and self.kp_softmax_dim2 == 1:
            self.kp_softmax_type = -1

        if self.kp_embedding_dim1 == 1 and self.kp_embedding_dim2 == 1:
            self.kp_embedding_type = -1


    def printSysConfig(self, exp_hw_config, exp_model_config, output_file):
        kiloByte = 1024
        megaByte = kiloByte * 1024
        gigaByte = megaByte * 1024
        teraByte = gigaByte * 1024

        with open(output_file, "w") as f:
            f.write("==========================\n")
            f.write("Hardware Configuration\n")
            f.write("==========================\n")

            f.write(
                "Throughput: {:.5f} Tflops\n".format(
                    self.core.operating_throughput / 1e12
                )
            )
            for i in range(self.num_levels - 1, -1, -1):
                mem_bw = self.memLayer[i].dynamic_throughput
                mem_size = self.memLayer[i].size

                if mem_bw < 1e3 * gigaByte:
                    f.write(
                        "L{:} Bandwidth: {:.1f} GB/s\n".format(i, mem_bw / (gigaByte))
                    )
                else:
                    f.write(
                        "L{:} Bandwidth: {:.1f} TB/s\n".format(i, mem_bw / (teraByte))
                    )

                if mem_size < 1e3 * megaByte:
                    f.write("L{:} Size: {:.10f} MB\n".format(i, mem_size / (megaByte)))
                elif mem_size < 1e3 * gigaByte:
                    f.write("L{:} Size: {:.1f} GB\n".format(i, mem_size / (gigaByte)))
                else:
                    f.write("L{:} Size: {:.1f} TB\n".format(i, mem_size / (teraByte)))

            f.write(
                "Intra-node Bandwidth: {:.1f} GB/s\n".format(
                    self.network.intra_network.throughput / (gigaByte)
                )
            )
            f.write(
                "Inter-node Bandwidth: {:.1f} GB/s\n".format(
                    self.network.inter_network.throughput / (gigaByte)
                )
            )

            M = self.memLayer[self.num_levels - 1].size
            (
                tot_mem,
                embedding_mem,
                hidden_mem,
                softmax_mem,
                projection_mem,
                wt_mem,
                act_mem,
                point_mem,
            ) = util.getTotMemReq(
                exp_hw_config,
                exp_model_config,
                batch_size=self.batch_size,
                hidden_dim=self.hidden_dim,
                vocab_size=self.vocab_size,
                seq_len=self.seq_len,
                num_layer=self.num_layers,
                dp=self.dp,
                lp=self.lp,
                kp1=self.kp_hidden_dim1,
                kp2=self.kp_hidden_dim2,
                kp_type=self.kp_hidden_type,
            )
            f.write("\n\n===========================================\n")
            f.write("Memory Requirement Breakdown per Data Shard\n")
            f.write("===========================================\n")
            f.write(
                "Total Memory: {:.1f} GB\n"
                "Embedding Memory: {:.1f} GB\n"
                "Hidden Memory: {:.1f} GB\n"
                "Softmax Memory: {:.1f} GB\n"
                "Projection Memory: {:.1f} GB\n".format(
                    tot_mem / gigaByte,
                    embedding_mem / gigaByte,
                    hidden_mem / gigaByte,
                    softmax_mem / gigaByte,
                    projection_mem / gigaByte,
                )
            )

            f.write(
                "\nTotal Memory: {:.1f} GB\n"
                "Weight Memory: {:.1f} GB\n"
                "Activation Memory: {:.1f} GB\n"
                "Pointwise Memory: {:.1f} GB\n".format(
                    tot_mem / gigaByte,
                    wt_mem / gigaByte,
                    act_mem / gigaByte,
                    point_mem / gigaByte,
                )
            )

            f.write(
                "\nMemory Overflow Rate (Total Memory Required per Data Shard / Memory capacity per node): {:.1f}\n".format(
                    float("inf") if M == 0 else tot_mem / M
                )
            )

            (
                tot_mem,
                embedding_mem,
                hidden_mem,
                softmax_mem,
                projection_mem,
                wt_mem,
                act_mem,
                point_mem,
            ) = util.getMemUsagePerCore(
                exp_hw_config,
                exp_model_config,
                batch_size=self.batch_size,
                hidden_dim=self.hidden_dim,
                vocab_size=self.vocab_size,
                seq_len=self.seq_len,
                num_layer=self.num_layers,
                dp=self.dp,
                lp=self.lp,
                kp1=self.kp_hidden_dim1,
                kp2=self.kp_hidden_dim2,
                kp_type=self.kp_hidden_type,
            )
            f.write("\n\n===========================================================\n")
            f.write("Memory Requirement Breakdown per Data Shard Per Model Shard\n")
            f.write("===========================================================\n")
            f.write(
                "Total Memory: {:.1f} GB\n"
                "Embedding Memory: {:.1f} GB\n"
                "Hidden Memory: {:.1f} GB\n"
                "Softmax Memory: {:.1f} GB\n"
                "Projection Memory: {:.1f} GB".format(
                    tot_mem / gigaByte,
                    embedding_mem / gigaByte,
                    hidden_mem / gigaByte,
                    softmax_mem / gigaByte,
                    projection_mem / gigaByte,
                )
            )

            f.write(
                "\nMemory Overflow Rate (Total Memory Required per Data Shard Per Model Shard/ Memory capacity per node): {:.5f}\n".format(
                    float("inf") if M == 0 else tot_mem / M
                )
            )

            f.write(
                "\nTotal Memory: {:.1f} GB\n"
                "Weight Memory: {:.1f} GB\n"
                "Activation Memory: {:.1f} GB\n"
                "Pointwise Memory: {:.1f} GB\n".format(
                    tot_mem / gigaByte,
                    wt_mem / gigaByte,
                    act_mem / gigaByte,
                    point_mem / gigaByte,
                )
            )

            f.write("\n\n====================\n")
            f.write("Parallelism Strategy\n")
            f.write("====================\n")
            f.write(
                "dp: {}, lp: {}, kp_hidden_dim1: {}, kp_hidden_dim2: {},"
                "kp_softmax_dim1: {}, kp_softmax_dim2: {}, kp_embedding1: {}, kp_embedding2: {},"
                "kp_projection_dim1: {}, kp_proejction_dim2: {}\n".format(
                    self.dp,
                    self.lp,
                    self.kp_hidden_dim1,
                    self.kp_hidden_dim2,
                    self.kp_softmax_dim1,
                    self.kp_softmax_dim2,
                    self.kp_embedding_dim1,
                    self.kp_embedding_dim2,
                    self.kp_projection_dim1,
                    self.kp_projection_dim2,
                )
            )

            f.write(
                "\n\n==============================================================================\n"
            )
            f.write("Hardware Component Stats\n")
            f.write(
                "==============================================================================\n"
            )
            self.core.printStats(f)
            for i in range(0, self.num_levels):
                self.memLayer[i].printStats(f)

            self.network.printStats(f)


    def generateTileSpace(self):
        tile_space = []
        tiles = [None] * self.num_levels

        for level in range(0, self.num_levels - 1):
            memory = self.memLayer[level]
            # tiles[level] = self.getTileDims(memory)
            tiles[level] = memory.getTileDims()

        if self.num_levels == 1:
            tile_space = []
        elif self.num_levels == 2:
            tile_space = tiles[0]
        elif self.num_levels == 3:
            tile_space = [(x, y) for x in tiles[0] for y in tiles[1]]
        elif self.num_levels == 4:
            tile_space = [
                (x, y, z) for x in tiles[0] for y in tiles[1] for z in tiles[2]
            ]
        else:
            raise NotImplementedError()

        # inject tile size
        # tile_space = [
        #     ((8, 4, 8), (16, 32, 16), (128,32,128)),
        # ]

        return tile_space

    def getTileSize(self, lid):
        memory = self.memLayer[lid]
        memory.calcTileDim()
        tile_dim = memory.getTileDim()
        return tile_dim, tile_dim, tile_dim

    # Count the number of accesses from level-1 to level
    # input matrix A(dim1, dim2) and B(dim2, dim3)
    # output matrix C(dim1, dim3)
    def getNumAccesses(self, level, dim1, dim2, dim3, tile_dim, num_repeat, name, r):
        # tile1,tile2,tile3 = self.getTileSize(level-1)
        # print("dim1= ", dim1, "dim2= ", dim2, "dim3 = ", dim3)

        tile1, tile2, tile3 = tile_dim
        # print("BEFORE: level = ", level, "|tile1= ", tile1, "tile2= ", tile2, "tile3 = ", tile3) ###############

        orig_size = tile1 * tile2 + tile1 * tile3 + tile2 * tile3
        short_tile_cond = [0, 0, 0]

        if tile1 > dim1:
            tile1 = dim1
            short_tile_cond[0] = 1
        if tile2 > dim2:
            tile2 = dim2
            short_tile_cond[1] = 1
        if tile3 > dim3:
            tile3 = dim3
            short_tile_cond[2] = 1

        # print("AFTER: level= ", level ,"|tile1= ", tile1, "tile2= ", tile2, "tile3 = ", tile3) ###############

        if short_tile_cond[2] == 0 and (short_tile_cond[0] | short_tile_cond[1]) == 1:
            if level <= 1:
                tile3 = math.floor((orig_size - tile1 * tile2) / (tile1 + tile2))
            else:
                # store bypasses cache, directly goes to memory
                tile3 = math.floor((orig_size - tile1 * tile2) / tile2)
            if tile3 > dim3:
                tile3 = dim3
            # Uncomment if tile3 needs to be pow of 2
            # tile3 = int(math.pow(2, math.floor(math.log2(tile3))))
        elif short_tile_cond[0] == 0 and (short_tile_cond[1] | short_tile_cond[2]) == 1:
            if level <= 1:
                tile1 = math.floor((orig_size - tile3 * tile2) / (tile3 + tile2))
            else:
                # store bypasses cache, directly goes to memory
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

        # print("FINAL: level= ", level ,"|tile1= ", tile1, "tile2= ", tile2, "tile3 = ", tile3) ###############

        reload_A = 1
        reload_B = 1
        reload_C = 1

        if tile1 > 0 and tile2 > 0 and tile3 > 0:
            reload_A = math.ceil(dim3 / tile3)
            reload_B = math.ceil(dim1 / tile1)
            # do not access the slow memory on every write,acculmuate in fast memory
            reload_C = 1 if level > 1 else math.ceil(dim2 / tile2)

        num_repeat = r[0] * r[1] * r[2]

        if level == 2:  # access to L1 scratchpad
            num_mem = (
                num_repeat * (dim1 * dim2 * reload_A + dim2 * dim3 * reload_B)
                + r[0] * dim1 * r[2] * dim3 * reload_C
            ) * self.precision
        else:
            num_mem = (
                num_repeat
                * (
                    dim1 * dim2 * reload_A
                    + dim2 * dim3 * reload_B
                    + dim1 * dim3 * reload_C
                )
                * self.precision
            )

        # num_mem = num_repeat * (dim1 * dim2 * reload_A + dim2 * dim3 * reload_B + dim1 * dim3 * reload_C) * self.precision

        if self.debug:
            print(name)
            print(
                "Matrix dimension at Level {}: {:,} x {:,} x {:,}".format(
                    level, dim1, dim2, dim3
                )
            )
            print(
                "Tile dimension at Level {}: {:,} x {:,} x {:,}".format(
                    level - 1, tile1, tile2, tile3
                )
            )
            print(
                "reload_A: {}, reload_B: {}, reload_C: {}".format(
                    reload_A, reload_B, reload_C
                )
            )
            print("num_repeat: {}".format(num_repeat))
            print("Bytes Accessed: {:,}".format(num_mem))
            print("")

        return num_mem, tile1, tile2, tile3

    # This is the main function that captures the memory hierarchy impact
    # on the number of accesses to global memory considering not everything fits in
    # L2 cache and also captures the effect of shared memory
    def GEMM(self, order_dims, tile_dims, name):
        dim1_ = order_dims[0]
        dim2_ = order_dims[1]
        dim3_ = order_dims[2]
        # dim1 = util.power2RoundUp(dim1_)
        # dim2 = util.power2RoundUp(dim2_)
        # dim3 = util.power2RoundUp(dim3_)
        dim1 = dim1_
        dim2 = dim2_
        dim3 = dim3_

        GEMM_flop = dim1 * dim3 * (dim2 + dim2 - 1)
        # dim2 multiply
        # dim2-1 add

        # X1 = self.L2_tile_dim
        # X2 = self.shared_mem_tile_dim
        # X3 = self.reg_tile_dim

        num_accesses = [0] * self.num_levels
        r1, r2, r3 = 1, 1, 1

        if algByte:
            num_accesses[self.num_levels - 1] = (
                dim1 * dim2 + dim2 * dim3 + dim1 * dim3
            ) * self.precision
        else:
            num_repeat = 1
            for level in range(self.num_levels - 1, 0, -1):
                repeat = (r1, r2, r3)
                num_accesses[level], tile1, tile2, tile3 = self.getNumAccesses(
                    level,
                    dim1,
                    dim2,
                    dim3,
                    tile_dims[level - 1],
                    num_repeat,
                    name,
                    repeat,
                )
                try:
                    num_repeat *= (
                        math.ceil(dim1 / tile1)
                        * math.ceil(dim2 / tile2)
                        * math.ceil(dim3 / tile3)
                    )
                    r1 *= math.ceil(dim1 / tile1)
                    r2 *= math.ceil(dim2 / tile2)
                    r3 *= math.ceil(dim3 / tile3)
                except:
                    num_repeat *= 1

                dim1 = tile1 if tile1 != 0 else dim1
                dim2 = tile2 if tile2 != 0 else dim2
                dim3 = tile3 if tile3 != 0 else dim3

            # assume systolic engine can support n x m x n GEMM  (e.g. 8 x 4 x 8 for A100 tensorcore), which is FLOPs_tile = n^2 * (m-1) FLOPs
            # reuse is the number of n x m x n GEMMs that are performed before stationary values (weight, activations, or output) get swapped
            # every n x n output tile:
            #   1. loads nxm activations and mxn weights -> 2 * reuse * n * m accesses
            #   2. performs reuse * FLOPs_tile computations
            #   3. writes back n^2 output elements

            reuse = 1
            dim1 = dim1_
            dim2 = dim2_
            dim3 = dim3_

            if self.dataflow == "none":
                reuse = 1
            elif self.dataflow == "best":
                reuse = max(
                    math.ceil(dim1 / self.FMA_dims[0]),
                    math.ceil(dim3 / self.FMA_dims[0]),
                    math.ceil(dim2 / self.FMA_dims[1]),
                )
            elif self.dataflow == "wst":  # weight stationary
                reuse = math.ceil(dim1 / self.FMA_dims[0])
            elif self.dataflow == "ast":  # activation stationary
                reuse = math.ceil(dim3 / self.FMA_dims[0])
            elif self.dataflow == "ost":  # output stationary
                reuse = math.ceil(dim2 / self.FMA_dims[1])
            else:
                raise NotImplementedError()

            # TODO: make sure to model underutilized systolic array

            num_accesses[0] = (
                GEMM_flop
                * (
                    2 * reuse * self.FMA_dims[0] * self.FMA_dims[1]
                    + self.FMA_dims[0] ** 2
                )
                / (2 * reuse * self.FMA_dims[0] * (self.FMA_dims[1] - 1))
                * self.precision
            )
            # num_accesses[0]    = GEMM_flop * ((2 * reuse + 1) / (2 * reuse)) * 1/self.FMA_width * self.precision
            # num_accesses[0]    = GEMM_flop * ((2 * reuse + self.FMA_width) / (2 * reuse)) * 1/self.FMA_width * self.precision

            # TODO: do we still need these in new hierarchical version?
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

    def roofline(self, flop, mem_access_, name=""):
        # print("Roofline: entered {}".format(name))
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

        num_level = len(mem_access)
        time = [0] * num_level
        comp_int = [0] * num_level
        inflection_point = [0] * num_level

        try:
            assert mem_access[num_level - 1] > 0, "last_level_mem = 0"
        except Exception as e:
            print(
                "{}: Number of accesses to the last level of memory hierarchy cannot be zero:\n {}".format(
                    name, e
                ),
                flush=True,
            )
            sys.exit(0)

        for i in range(0, num_level):
            time[i] = 0
            mem_bw = self.memLayer[i].getThroughput()
            mem_latency = self.memLayer[i].getLatency()
            num_mem = mem_access[i]
            inflection_point[i] = float("inf") if mem_bw == 0 else self.th / mem_bw
            comp_int[i] = 0 if num_mem == 0 else flop / num_mem

            if comp_int[i] < inflection_point[i]:  # mem-bound
                time[i] = (
                    float("inf")
                    if (mem_bw == 0 or num_mem == 0)
                    else (num_mem / mem_bw)
                ) + mem_latency
            else:  # compute-bound
                time[i] = float("inf") if (self.th == 0) else (flop / self.th)

        max_time = max(time)

        # if self.debug:
        #     print("{}: {}".format(name, max_time))
        #     print("GEMM flops: {:,}".format(flop))
        #     for i in range(0, num_level):
        #         print("L{}".format(i))
        #         print("inflection_point: {:.2f}".format(inflection_point[i]))
        #         print("comp_int: {:.2f}".format(comp_int[i]))
        #         print("time: {}".format(time[i]))
        #         print(
        #             "Throughput = ",
        #             self.th,
        #             "BW = ",
        #             self.memLayer[i].getThroughput(),
        #             "mem_latency=",
        #             self.memLayer[i].getLatency(),
        #         )  ################
        #         print()

        # print("Roofline: exited {}".format(name))
        return max_time



    def getGEMMTime(self, dim1, dim2, dim3, name):
        tile2time = {}
        gemm_dict = {}
        orderSpace = self.generateOrder(dim1, dim2, dim3, name)
        for order_dims in orderSpace:
            if self.debug:
                print("===============================================================")
                print("order: {}".format(order_dims))
                print("===============================================================")
            # print("TILE_SPACE: \n", self.tileSpace)
            for tile_dims in self.tileSpace:
                tile_dims = tile_dims + ((dim1, dim2, dim3),)
                if self.debug:
                    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                    print("tile: {}".format(tile_dims))
                    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                # GEMM_flop, mem_access = self.GEMM(order_dims, tile_dims, name)
                gemm = TiledGEMM(
                    order_dims, tile_dims, self.core, self.precision
                )  # assumes 4 levels of memory hierarchy

                GEMM_flop = gemm.GEMM_flop()
                mem_access = gemm.mem_accesses()
                GEMM_time = self.roofline(GEMM_flop, mem_access, name) + self.O
                tile2time[(order_dims, tile_dims)] = (GEMM_time, mem_access)
                gemm_dict[(order_dims, tile_dims)] = gemm

        best_tile = min(tile2time, key=tile2time.get)
        best_time, mem_access = tile2time[best_tile]
        best_gemm = gemm_dict[best_tile]

        if self.debug:
            print(repr(best_gemm))
            print(
                "{}: Best Time: {:,} ms, Best Order: {}, Best Tile: {}\n".format(
                    name, best_time * 1e3, best_tile[0], best_tile[1]
                )
            )

        return best_time, best_tile[0], best_tile[1], mem_access

    def generateOrder(self, dim1, dim2, dim3, name):
        if self.dataflow == "best":  # best stationary
            if dim1 >= max(dim2, dim3):
                self.dataflow = "wst"
            elif dim2 >= max(dim1, dim3):
                self.dataflow = "ost"
            elif dim3 >= max(dim1, dim2):
                self.dataflow = "ast"

        order = []
        if self.dataflow == "wst":  # weight stationary
            # order.append((dim2, dim3, dim1))
            order.append("mnk")
            if dim2 != dim3:
                # order.append((dim3, dim2, dim1))
                order.append("mkn")
        elif self.dataflow == "ast":  # activation stationary
            # order.append((dim1, dim2, dim3))
            order.append("nmk")
            if dim2 != dim1:
                # order.append((dim2, dim1, dim3))
                order.append("nkm")
        elif self.dataflow == "ost":  # output stationary
            # order.append((dim1, dim3, dim2))
            order.append("knm")
            if dim1 != dim3:
                # order.append((dim3, dim1, dim2))
                order.append("kmn")
        elif self.dataflow == "none":  # not stationary
            if dim1 != dim2 and dim2 != dim3 and dim1 != dim3:
                # order=list(itertools.permutations([dim1, dim2, dim3]))
                order = ["".join(i) for i in list(itertools.permutations("mnk"))]
            elif dim1 == dim2 and dim2 != dim3:
                # order = [(dim1, dim2, dim3), (dim1, dim3, dim2), (dim3, dim1, dim2)]
                order = ["nkm", "knm", "kmn"]
            elif dim1 == dim3 and dim2 != dim1:
                # order = [(dim1, dim2, dim3), (dim1, dim3, dim2), (dim2, dim1, dim3)]
                order = ["nkm", "knm", "nmk"]
            elif dim2 == dim3 and dim1 != dim2:
                # order = [(dim1, dim2, dim3), (dim2, dim1, dim3), (dim2, dim3, dim1)]
                order = ["nkm", "nmk", "mnk"]

        order = ["".join(i) for i in list(itertools.permutations("mnk"))]

        return order

    def getCf(self, m, k, n):
        # Add Biad adds
        """Get Cell Time on Forward Path"""
        GEMM_time = self.getGEMMTime(m, k, n, "Cf")

        point_flop = m * n * 5
        # 1: add bias
        # 5: add nonlinearities, there is one more than the number of gates (self.G)
        # 1: pointwise muliply and add
        point_mem = self.precision * m * n * (3 * 3 + 2 * 2)
        # 3: 3 memory accesses for operands with two inputs and one output
        # 2: 1 for bias add + 1 for pointwise mul
        # 2: 2 memory accesses for operands with one input and one output
        # 1: 5/4 non-linearities per gate

        point_time = (
            self.roofline(point_flop, point_mem, name="pointwise_Cf") + 5 * self.O
        )

        if self.debug:
            gigaByte = 1024 * 1024 * 1024
            print(
                "Hidden point_flop: {:,}, point_mem: {:,}\n".format(
                    int(point_flop / 1e9), int(point_mem / gigaByte)
                )
            )
            print("Hidden point_time: {:,}\n".format(point_time))

        if self.validating_GEMM:
            return GEMM_time
        else:
            return GEMM_time[0] + point_time
        

    def get_node_f(self, gemm, name):
        
        """Get node Time on Forward Path"""
        m = gemm[0]
        k = gemm[1]
        n = gemm[2]
        GEMM_time = self.getGEMMTime(m, k, n, name)
        point_time = (
            self.O # self.roofline(point_flop, point_mem, name="pointwise_Cf") +
        )
        if self.debug:
            print("{} GEMM_time: {:,}\n".format(name, GEMM_time[0]))
        if self.validating_GEMM:
            return GEMM_time
        else:
            return GEMM_time[0] + point_time
        
    def get_node_b(self, gemm, name):
        """Get node Time on Backward Path"""
        m = gemm[0]
        k = gemm[1]
        n = gemm[2]
        grad_act_time, _, _, _ = self.getGEMMTime(
            m, n, k, "{}_act".format(name)
        )
        grad_wt_time, _, _, _ = self.getGEMMTime(
            k, m, n, "{}_wt".format(name)
        )

        GEMM_time = grad_act_time + grad_wt_time

        point_time = (
            self.O # self.roofline(point_flop, point_mem, name="pointwise_Cf") +
        )

        if self.debug:
            print("Hidden point_time: {:,}\n".format(point_time))

        return GEMM_time + point_time
        
    def getEmbedding_f(self):
        embedding_mem = 2 * (self.miniB * self.hidden_dim * self.precision)
        # embedding_time = (embedding_mem)/ (self.mem_bw) + self.mem_latency + self.O
        embedding_time = self.roofline(0, embedding_mem, name="embedding_f") + self.O
        embedding_transfer_time = 2 * self.miniB * self.hidden_dim * self.precision / self.H2Dbw
        if self.debug:
            print("Embedding_mem: {:,}".format(int(embedding_mem / 1e9)))
        return embedding_time + embedding_transfer_time
    def getLinearSoftmax_f(self):
        GEMM_time, _, _, _ = self.getGEMMTime(
            self.miniB,  self.hidden_dim, self.vocab_size, "linear_softmax_f"
        )
        point_flop = self.miniB * (3 * self.vocab_size - 1)
        point_mem = self.precision * self.miniB * (7 * self.vocab_size)
        point_time = (
            self.roofline(point_flop, point_mem, name="pointwise-linear-softmax-f")
            + 4 * self.O
        )

        if self.debug:
            print(
                "Linear Softmax point_flop: {:,}, point_mem: {:,}".format(
                    int(point_flop / 1e9), int(point_mem / 1e9)
                )
            )
            print("point_time: {:,}\n".format(point_time))

        return GEMM_time + point_time
    def getScaleSoftmax_f(self, gemm):
        m = gemm[0]
        n = gemm[2]
        scale_flop = m * n
        scale_mem = self.precision * scale_flop * 2
        softmax_flop = m * n * 3
        softmax_mem = self.precision * m * n * 7
        scale_time = (
            self.roofline(scale_flop, scale_mem, name="pointwise-scale-f")
            + 1 * self.O
        )
        softmax_time = (
            self.roofline(softmax_flop, softmax_mem, name="pointwise-softmax-f")
            + 4 * self.O
        )

        if self.debug:
            print(
                "Scale Softmax point_flop: {:,}, point_mem: {:,}".format(
                    int(scale_flop / 1e9), int(scale_mem / 1e9)
                )
            )
            print("scale point_time: {:,}\n".format(scale_time))
            print("softmax point_flop: {:,}, softmax point_mem: {:,}".format(
                int(softmax_flop / 1e9), int(softmax_mem / 1e9)
            ))
            print("softmax point_time: {:,}\n".format(softmax_time))
            
            

        return scale_time + softmax_time
    def getScaleSoftmax_b(self, gemm):
        m = gemm[0]
        n = gemm[2]
        scale_flop = m * n
        scale_mem = self.precision * scale_flop * 2
        softmax_flop = m * n * 5
        softmax_mem = self.precision * m * n * 11
        scale_time = (
            self.roofline(scale_flop, scale_mem, name="pointwise-scale-f")
            + 1 * self.O
        )
        softmax_time = (
            self.roofline(softmax_flop, softmax_mem, name="pointwise-softmax-f")
            + 4 * self.O
        )

        if self.debug:
            print(
                "(gr)Scale Softmax point_flop: {:,}, point_mem: {:,}".format(
                    int(scale_flop / 1e9), int(scale_mem / 1e9)
                )
            )
            print("(gr)scale point_time: {:,}\n".format(scale_time))
            print("(gr)softmax point_flop: {:,}, softmax point_mem: {:,}".format(
                int(softmax_flop / 1e9), int(softmax_mem / 1e9)
            ))
            print("(gr)softmax point_time: {:,}\n".format(softmax_time))
            
        return scale_time + softmax_time
    def getResidual_f(self, gemm):
        m = gemm[0]
        n = gemm[2]
        residual_flop = m * n
        residual_mem = self.precision * residual_flop * 3
    
        residual_time = (
            self.roofline(residual_flop, residual_mem, name="pointwise-scale-f")
            + 1 * self.O
        )
    

        if self.debug:
            print(
                "Residual point_flop: {:,}, point_mem: {:,}".format(
                    int(residual_flop / 1e9), int(residual_mem / 1e9)
                )
            )
            print("Residual point_time: {:,}\n".format(residual_time))
            
        return residual_time
    def getResidual_b(self, gemm):
        m = gemm[0]
        n = gemm[2]
        residual_flop = m * n
        residual_mem = self.precision * residual_flop * 3
        residual_time = (
            self.roofline(residual_flop, residual_mem, name="pointwise-scale-f")
            + 1 * self.O
        )
        if self.debug:
            print(
                "(gr)Residual point_flop: {:,}, point_mem: {:,}".format(
                    int(residual_flop / 1e9), int(residual_mem / 1e9)
                )
            )
            print("(gr)Residual point_time: {:,}\n".format(residual_time))
            
        return residual_time
    def getLayernorm_f(self, gemm):
        m = gemm[0]
        n = gemm[2]
        flops = m * n * 5
        mem = self.precision * m * n * 9
        time = (
            self.roofline(flops, mem, name="pointwise-scale-f")
            + 3 * self.O
        )
        if self.debug:
            print(
                "Layernorm point_flop: {:,}, point_mem: {:,}".format(
                    int(flops / 1e9), int(mem / 1e9)
                )
            )
            print("Layernorm point_time: {:,}\n".format(time))
            
        return time
    def getLayernorm_b(self, gemm):
        m = gemm[0]
        n = gemm[2]
        flops = m * n * 7
        mem = self.precision * m * n * 11
        time = (
            self.roofline(flops, mem, name="pointwise-scale-f")
            + 4 * self.O
        )
        if self.debug:
            print(
                "(gr)Layernorm point_flop: {:,}, point_mem: {:,}".format(
                    int(flops / 1e9), int(mem / 1e9)
                )
            )
            print("(gr)Layernorm point_time: {:,}\n".format(time))
            
        return time
    def getLinearSoftmax_b(self):
        grad_wt_time, _, _, _ = self.getGEMMTime(
            self.hidden_dim, self.miniB, self.vocab_size, "linear_softmax_b_wt"
        )
        grad_act_time, _, _, _ = self.getGEMMTime(
            self.miniB, self.vocab_size, self.hidden_dim, "linear_softmax_b_act"
        )

        GEMM_time = grad_wt_time + grad_act_time
        point_flop = self.miniB * self.vocab_size * 5
        # 1: one for one of the divisions, grad(A) (y=A/B)
        # 2: one for division and multiplication, grad(B)
        # 1: one for addition, copies turn into add
        # 1: one for sigmoid

        point_mem = self.precision * self.miniB * self.vocab_size * 11
        # 3: grad(A) in pointwise division
        # 3: grad(B) in pointwise division
        # 3: addition in copy backprop
        # 2: sigmoid

        point_time = (
            self.roofline(point_flop, point_mem, name="pointwise-linear-softmax-b")
            + 4 * self.O
        )

        if self.debug:
            print(
                "(gr) Linear Softmax point_flop: {:,}, point_mem: {:,}".format(
                    int(point_flop / 1e9), int(point_mem / 1e9)
                )
            )
            print("(gr) Linear Softmax point_time: {:,}\n".format(point_time))

        return GEMM_time + point_time
    
    def getEmbedding_b(self):
        # p2p_data_transfer = (self.precision * self.miniB * self.D)
        # data_transfer_time  = 0 if (self.dp == 1) else (float("inf") if (self.IBD == 0) else (((p2p_data_transfer) / self.IBD + self.LLD) * 2 * (self.dp -1 )))

        embedding_mem = 2 * self.miniB * self.hidden_dim * self.precision
        # embedding_mem_time = (embedding_mem / self.mem_bw) + self.mem_latency + self.O
        embedding_mem_time = (
            self.roofline(0, embedding_mem, name="embedding_b") + self.O
        )

        if self.debug:
            print("(gr) Embedding_mem: {:,}".format(int(embedding_mem / 1e9)))
        # return data_transfer_time + embedding_mem_time
        return embedding_mem_time

    def calcTime_LLM(self):
        """Calculate time for LLM model."""
        batch_size = self.batch_size
        vocab_size = self.vocab_size
        num_layers = self.num_layers
        hidden_dim = self.hidden_dim
        seq_len = self.seq_len
        num_heads = self.num_heads
        ffn_mult = self.ffn_mult
        if ffn_mult is not None:
            ffn_dim = self.hidden_dim * ffn_mult
        else:
            ffn_dim = self.model.ffn_dim
        n_tokens = self.n_tokens
        communication_time = self.communication_time
        N_PP = self.N_PP
        

        lp = self.lp
        dp = self.dp

        self.readjust_type()
        gemm_3d=process_gemm_shapes(batch_size, seq_len, hidden_dim, num_heads, ffn_dim, option="multiply_batch_into_m")
        # print(gemm_3d) #m,k,n
        gemm_qkv_proj = gemm_3d[0]
        gemm_attention_score = gemm_3d[1]
        gemm_attention_output = gemm_3d[2]
        gemm_output_proj = gemm_3d[3]
        gemm_ffn1 = gemm_3d[4]
        gemm_ffn2 = gemm_3d[5]
        if self.kp_hidden_type == -1:
            embedding_f = self.getEmbedding_f()
            embedding_b = self.getEmbedding_b()
            qkv_proj_f, qkv_proj_b = self.get_node_f(gemm=gemm_qkv_proj, name="qkv_projection_f"), self.get_node_b(gemm=gemm_qkv_proj, name="qkv_projection_b")
            attention_score_f, attention_score_b = self.get_node_f(gemm=gemm_attention_score, name="attention_score_f"), self.get_node_b(gemm=gemm_attention_score, name="attention_score_b")
            attention_scale_softmax_f, attention_scale_softmax_b = self.getScaleSoftmax_f(gemm=gemm_attention_score), self.getScaleSoftmax_b(gemm=gemm_attention_score)  # Not used in this context
            attention_output_f, attention_output_b = self.get_node_f(gemm=gemm_attention_output, name="attention_output_f"), self.get_node_b(gemm=gemm_attention_output, name="attention_output_b")
            output_proj_f, output_proj_b = self.get_node_f(gemm=gemm_output_proj, name="output_projection_f"), self.get_node_b(gemm=gemm_output_proj, name="output_projection_b")
            residual1_f, residual1_b = self.getResidual_f(gemm=gemm_output_proj), self.getResidual_b(gemm=gemm_output_proj)
            layernorm1_f, layernorm1_b = self.getLayernorm_f(gemm=gemm_output_proj), self.getLayernorm_b(gemm=gemm_output_proj)
            ffn1_f, ffn1_b = self.get_node_f(gemm=gemm_ffn1, name="ffn_f"), self.get_node_b(gemm=gemm_ffn1, name="ffn_b")
            ffn2_f, ffn2_b = self.get_node_f(gemm=gemm_ffn2, name="ffn2_f"), self.get_node_b(gemm=gemm_ffn2, name="ffn2_b")
            residual2_f, residual2_b = self.getResidual_f(gemm=gemm_ffn2), self.getResidual_b(gemm=gemm_ffn2)
            layernorm2_f, layernorm2_b = self.getLayernorm_f(gemm=gemm_ffn2), self.getLayernorm_b(gemm=gemm_ffn2)
            linear_softmax_f, linear_softmax_b = self.getLinearSoftmax_f(), self.getLinearSoftmax_b()  
            
            # QK_scaling_f = self.get_linear_f(

            # Tf = self.getInterLayerCommLatency(B, D)
        # elif self.kp_hidden_type == 1:  # CR
        #     Cf = self.getCf_kp1()
        #     Cb = self.getCb_kp1()
        #     Tf = self.getInterLayerCommLatency(B, D / self.kp_hidden_dim1)
        # elif self.kp_hidden_type == 2:  # RC
        #     Cf = self.getCf_kp2()
        #     Cb = self.getCb_kp2()
        #     Tf = self.getInterLayerCommLatency(
        #         B / self.kp_hidden_dim1, D / self.kp_hidden_dim2
        #     )
        else:
            print("Incorrect distributed GEMM type, 1: Column-Row, 2: Row-Column")
            sys.exit()

        if self.lp == 1:
            Tf = 0

        Tb = Tf

        # if self.kp_softmax_type == -1:
        #     Sf = self.getLinearSoftmax_f()
        #     Sb = self.getLinearSoftmax_b()
        # elif self.kp_softmax_type == 1:  # CR
        #     Sf = self.getSoftmax_f_kp1()
        #     Sb = self.getSoftmax_b_kp1()
        # elif self.kp_softmax_type == 2:  # RC
        #     Sf = self.getSoftmax_f_kp2()
        #     Sb = self.getSoftmax_b_kp2()
        # else:
        #     print("Incorrect distributed GEMM type, 1: Column-Row, 2: Row-Column")
        #     sys.exit()

        # if self.kp_embedding_type == -1:
        #     Ef = self.getEmbedding_f()
        #     Eb = self.getEmbedding_b()
        # elif self.kp_embedding_type == 1:  # CR
        #     Ef = self.getEmbedding_f_kp1()
        #     Eb = self.getEmbedding_b_kp1()
        # elif self.kp_embedding_type == 2:  # RC
        #     Ef = self.getEmbedding_f_kp2()
        #     Eb = self.getEmbedding_b_kp2()
        # else:
        #     print("Incorrect distributed GEMM type, 1: Column-Row, 2: Row-Column")
        #     sys.exit()

        # Rc = self.getDataParallelReduction(
        #     k=2 * D,
        #     n=G * D,
        #     dim1=self.kp_hidden_dim1,
        #     dim2=self.kp_hidden_dim2,
        #     name="Hidden Reduction",
        # )
        # Rs = self.getDataParallelReduction(
        #     k=D,
        #     n=V,
        #     dim1=self.kp_softmax_dim1,
        #     dim2=self.kp_softmax_dim2,
        #     name="Softmax Reduction",
        # )
        # Re = self.getDataParallelReduction(
        #     k=V,
        #     n=D,
        #     dim1=self.kp_embedding_dim1,
        #     dim2=self.kp_embedding_dim2,
        #     name="Embedding Reduction",
        # )

        if self.debug:
            print(
                "dp: {}, lp: {}, kp_hidden_dim1: {}, kp_hidden_dim2: {}, kp_softmax_dim1: {}, kp_softmax_dim2: {}, kp_embedding_dim1: {}, kp_embedding_dim2: {},  kp_hidden_type: {}, kp_softmax_type: {}, kp_embedding_type: {}\n".format(
                    dp,
                    lp,
                    self.kp_hidden_dim1,
                    self.kp_hidden_dim2,
                    self.kp_softmax_dim1,
                    self.kp_softmax_dim2,
                    self.kp_embedding_dim1,
                    self.kp_embedding_dim2,
                    self.kp_hidden_type,
                    self.kp_softmax_type,
                    self.kp_embedding_type,
                )
            )

            # print(
            #     "Cf: {} Cb: {} softmax_f: {} softmax_b: {} embedding_f: {} embedding_b: {} "
            #     "Rs: {} Rc: {} Re: {}\n".format(Cf, Cb, Sf, Sb, Ef, Eb, Rs, Rc, Re)
            # )
            print(
                "embedding_f: {}, embedding_b: {}, qkv_proj_f: {}, qkv_proj_b: {}, attention_score_f: {}, attention_score_b: {}, attention_scale_softmax_f: {}, attention_scale_softmax_b: {}, attention_output_f: {}, attention_output_b: {}, output_proj_f: {}, output_proj_b: {}, residual1_f: {}, residual1_b: {}, layernorm1_f: {}, layernorm1_b: {}, ffn1_f: {}, ffn1_b: {}, ffn2_f: {}, ffn2_b: {}".format(
                embedding_f, embedding_b,
                
                qkv_proj_f, qkv_proj_b, attention_score_f, attention_score_b,
                attention_scale_softmax_f, attention_scale_softmax_b,
                attention_output_f, attention_output_b, output_proj_f, output_proj_b,
                residual1_f, residual1_b, layernorm1_f, layernorm1_b,
                ffn1_f, ffn1_b, ffn2_f, ffn2_b
            ))
        print("Calculating LLM time...")
        g = Graph(
            num_seq=seq_len,
            num_layer=num_layers,
            lp=lp,
            
            T_embedding_f=embedding_f,
            T_qkv_projection_f= qkv_proj_f,
            T_attention_score_f= attention_score_f,
            T_attention_scale_softmax_f= attention_scale_softmax_f,
            T_attention_output_f= attention_output_f,
            T_out_proj_f = output_proj_f,
            
            T_residual1_f= residual1_f,
            T_layer_norm1_f= layernorm1_f,
            T_ffn1_f= ffn1_f,
            T_ffn2_f= ffn2_f,
            T_residual2_f= residual2_f,
            T_layer_norm2_f= layernorm2_f,
            T_linear_softmax_f= linear_softmax_f,
            
            T_embedding_b=embedding_b,
            T_qkv_projection_b= qkv_proj_b,
            T_attention_score_b= attention_score_b,
            T_attention_scale_softmax_b= attention_scale_softmax_b,
            T_attention_output_b= attention_output_b,
            T_out_proj_b = output_proj_b,
            T_residual1_b= residual1_b,
            T_layer_norm1_b= layernorm1_b,
            T_ffn1_b= ffn1_b,
            T_ffn2_b= ffn2_b,
            T_residual2_b= residual2_b,
            T_layer_norm2_b= layernorm2_b,
            T_linear_softmax_b= linear_softmax_b,
            Tb=0,  
            Tf=0,
            
            
            
            
            
            
        
        )

        # g = Graph(
        #     num_seq=S,
        #     num_layer=L,
        #     lp=lp,
        #     Ef=Ef,
        #     Cf=Cf,
        #     Sf=Sf,
        #     Tf=Tf,
        #     Eb=Eb,
        #     Cb=Cb,
        #     Sb=Sb,
        #     Tb=Tb,
        #     Re=Re,
        #     Rc=Rc,
        #     Rs=Rs,
        # )
        fw_roots = g.construct_fwd_graph()
        time_fw = g.simulate(fw_roots[0], 0)
        
        filename = "fwd_graph_s%s_l%s_lp%s" % (g.num_seq, g.num_layer, g.lp)
        dot_fw = visualize_graph(fw_roots[0], filename=filename)
        dot_fw.render(filename, format="png", cleanup=True)
        print("Forward graph saved to %s.png" % filename)
        print("Forward simulation time: {}".format(time_fw))
        
        bw_roots = g.construct_bwd_graph()
        time_bw = g.simulate(bw_roots[0], g.lp - 1)   
        dot_bw = visualize_graph(bw_roots[0], filename=filename + "_bwd")
        dot_bw.render(filename + "_bwd", format="png", cleanup=True)
        print("Backward graph saved to %s_bwd.png" % filename)
        
        print("Backward simulation time: {}".format(time_bw))
        
        self.tot_time = time_fw + time_bw
    
        # fw_roots = g.construct_fwd_graph()
        # bw_roots = g.construct_bwd_graph()

        # time_fw = g.simulate(fw_roots[0], 0)
        # time_bw = g.simulate(bw_roots[g.num_seq - 1], g.lp - 1)

        # self.tot_time = time_fw + time_bw
        # tot_param = self.tot_param()
        
        return time_fw, time_bw#, tot_param

    def getTime(self):
        return self.tot_time
    
if __name__ == "__main__":
        # Example usage
    batch_size = 32
    vocab_size = 10000
    num_layers = 12
    hidden_dim = 768
    seq_len = 128
    num_heads = 12
    ffn_mult = 4  # Example multiplier for feed-forward network dimension
    ffn_dim = hidden_dim * ffn_mult
    n_tokens = 1000000  # Example number of tokens
    communication_time = 0.1  # Example communication time
    N_PP = 8  # Example number of parallel processing units
    