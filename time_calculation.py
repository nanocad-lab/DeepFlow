#!/tools/lm-venv/py3.6-tf-1.3.0-svail/bin/python

import math
import os
import sys
import config
import shutil
import itertools
from typing import Optional
# import numpy as np

from parallelism import Parallelism
from topology import Topology
from simulate import Graph
import util
from hw_component import Core, MemoryHierarchy, Network, DRAM
from astrasim_lib import run_cache_astrasim
from model import Model_LSTM, Model_GEMM, Model_LLM
from tile import TiledGEMM, formatBytes

algByte = False  # algorithmic ops false
proj = False  # consider projection layer, turn off for end-2-end validation, as baeline model does not have projection layer
validating_v100 = False

class NetworkModel:
    def __init__(self, hw_config, precision, kernel_overhead, roofline_cb, astra_policy: Optional[str] = None):
        self.hw_config = hw_config
        self.precision = precision
        self.O = kernel_overhead
        self._roofline = roofline_cb
        self._astra_policy = astra_policy or "analytical"

    def _astra_collective(self, kind: str, participants: int, size_bytes: int) -> float:
        part = int(participants)
        if part <= 1 or size_bytes <= 0:
            return 0.0
        byte_count = int(math.ceil(size_bytes))
        _, max_sec = run_cache_astrasim(
            self.hw_config,
            comm=kind,
            npus_count=part,
            size_bytes=byte_count,
            astra_config_dir="./astra_cache",
            cache_path="./astra_cache/cache.json",
        )
        return float(max_sec)


    def collective(
        self,
        *,
        kind: str,
        size_bytes: float,
        participants: int,
        ib: float,
        ll: float,
        local_bytes: float = 0.0,
        local_ops: float = 0.0,
        debug_label: str = "",
    ) -> float:
        if size_bytes <= 0:
            return 0.0

        # Pipeline and point-to-point operations now handled through unified path

        if participants is None:
            return 0.0

        part = int(participants)
        if part <= 1:
            return 0.0

        collective_ops = {"all_reduce", "reduce_scatter", "all_gather", "all_to_all"}

        network_bytes = float(math.ceil(size_bytes))
        local_bytes_int = int(math.ceil(local_bytes)) if local_bytes else 0

        if self._astra_policy in {"hybrid", "full"}:
            if kind in collective_ops or kind == "pipeline":
                # Pipeline uses 2 NPUs for point-to-point, others use part
                npus = 2 if kind == "pipeline" else part
                network_time = self._astra_collective(kind, npus, network_bytes)
            else:
                raise ValueError(f"Unsupported collective operation: {kind}")
        else:
            network_time = self._analytical_collective(
                kind=kind,
                size_bytes=network_bytes,
                participants=part,
                ib=ib,
                ll=ll,
                debug_label=debug_label,
            )

        local_time = 0.0
        if local_ops or local_bytes:
            local_time = self._roofline(
                local_ops,
                local_bytes_int,
                name=f"{debug_label}-local",
            )

        overhead_kinds = {"all_reduce", "reduce_scatter", "all_gather"}
        overhead = self.O if (network_bytes and kind in overhead_kinds) else 0.0

        return network_time + local_time + overhead


    def _analytical_collective(
        self,
        *,
        kind: str,
        size_bytes: float,
        participants: int,
        ib: float,
        ll: float,
        debug_label: str,
    ) -> float:
        if kind == "all_reduce":
            return self._analytical_all_reduce(size_bytes, participants, ib, ll, debug_label)
        if kind == "reduce_scatter":
            return self._analytical_reduce_scatter(size_bytes, participants, ib, ll, debug_label)
        if kind == "all_gather":
            return self._analytical_all_gather(size_bytes, participants, ib, ll, debug_label)
        if kind == "all_to_all":
            return self._analytical_all_to_all(size_bytes, participants, ib, ll, debug_label)
        if kind == "pipeline":
            return self._analytical_point_to_point(size_bytes, ib, ll)
        # Default fallback for unknown patterns
        raise ValueError(f"Unsupported collective operation: {kind}")

    def _analytical_all_reduce(self, size_bytes, participants, ib, ll, label):
        if ib == 0:
            return float("inf")
        per_rank = size_bytes / participants
        mem_access = self._roofline(
            0,
            int(math.ceil(2 * size_bytes / participants)),
            name=f"{label}-mem",
        )
        data_transfer = ((per_rank / ib) + mem_access + ll) * 2 * (participants - 1)
        prep_comp = per_rank
        prep_mem = int(math.ceil(3 * size_bytes / participants))
        data_prep = (
            self._roofline(prep_comp, prep_mem, name=f"{label}-prep") + self.O
        ) * (participants - 1)
        return data_transfer + data_prep

    def _analytical_reduce_scatter(self, size_bytes, participants, ib, ll, label):
        if ib == 0:
            return float("inf")
        per_rank = size_bytes / participants
        mem_access = self._roofline(
            0,
            int(math.ceil(2 * size_bytes / participants)),
            name=f"{label}-mem",
        )
        data_transfer = ((per_rank / ib) + mem_access + ll) * (participants - 1)
        prep_comp = per_rank
        prep_mem = int(math.ceil(3 * size_bytes / participants))
        data_prep = (
            self._roofline(prep_comp, prep_mem, name=f"{label}-prep") + self.O
        ) * (participants - 1)
        return data_transfer + data_prep

    def _analytical_all_gather(self, size_bytes, participants, ib, ll, label):
        if ib == 0:
            return float("inf")
        mem_access = self._roofline(
            0,
            int(math.ceil(2 * size_bytes / participants)),
            name=f"{label}-mem",
        )
        data_transfer = ((size_bytes / ib) + mem_access + ll) * (participants - 1)
        return data_transfer

    def _analytical_all_to_all(self, size_bytes, participants, ib, ll, label):
        if ib == 0:
            return float("inf")
        return ((size_bytes / participants) / ib + ll) * (participants - 1)

    def _analytical_point_to_point(self, size_bytes, ib, ll):
        if size_bytes <= 0:
            return 0.0
        if ib == 0:
            return float("inf")
        return size_bytes / ib + ll


class TimeCalculation:
    def __init__(self, hw_config, model_config, mode, *, astra_policy_override: Optional[str] = None):
# Mode parameter
        

        # Software Parameters
        self.O = hw_config.sw_config.kernel_launch_overhead
        self.precision = hw_config.sw_config.precision
        self.h2d_bandwidth = getattr(hw_config.sw_config, "h2d_bandwidth", -1)
        self.attached = True

        # Hardware Parameters
        self.hw_config = hw_config
        self.core = Core(hw_config)
        self.th = self.core.getThroughput()
        self.FMA_dims = self.core.FMA_dims  # (FMA_x, FMA_y)
        self.dataflow = self.core.dataflow

        self.memoryHierarchy = MemoryHierarchy(hw_config)
        self.num_levels = self.memoryHierarchy.num_levels
        self.memLayer = self.memoryHierarchy.memLayer
        self.tileSpace = None

        # TODO: move this to config file
        self.H2Dbw = self.h2d_bandwidth


        # System Parameters
        self.num_wafer = hw_config.system_config.num_wafers
        self.num_workers = hw_config.system_config.num_workers



        level = 0
        mem_config = hw_config.memory_hierarchy.mem_hr[level]
        self.DRAM = DRAM(hw_config, mem_config, level)
        self.memory_capacity = self.DRAM.size * self.num_workers# in bytes

        # self.memory_capacity = hw_config.perimeter_breakdown.DRAM

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
                # print(f'intra_throughput / intra_derate: {intra_throughput / intra_derate}, inter_throughput / inter_derate: {inter_throughput / inter_derate}')
            else:
                derated_inter_throughput = inter_throughput / inter_derate
                # print(f'inter_throughput / inter_derate: {inter_throughput / inter_derate}')
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
        self.IBD, self.LLD = ( #interconnect bandwidth and latency for data parallelism
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
        self.mb = par.mb
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
        if self.num_workers % (self.kp_hidden_dim1 * self.kp_hidden_dim2) != 0:
            raise ValueError("num_workers must be divisible by (kp_hidden_dim1 * kp_hidden_dim2)")
        num_workers = self.num_workers/(self.kp_hidden_dim1*self.kp_hidden_dim2)
        # print(f'num_workers after kp_hidden_dim1 and kp_hidden_dim2: {num_workers}')

        if num_workers % self.dp != 0:
            raise ValueError("num_workers must be divisible by dp")
        self.num_workers_dp = num_workers / self.dp # number of workers for each data parallelism batch
        # print(f'num_workers_dp after dividing by dp: {self.num_workers_dp}')

        if self.num_workers_dp % self.lp != 0:
            raise ValueError("num_workers_dp must be divisible by lp")
        self.num_workers_lp = self.num_workers_dp / self.lp if self.lp > 1 else self.num_workers_dp #number of workers per pipeline stage
        # print(f'num_workers_lp after dividing by lp: {self.num_workers_lp}')
        # print(f'lp: {self.lp}')
        if self.num_workers_lp != 1:
            raise ValueError("num_workers_lp must be equal to 1")
        
        
        #check parallelism parameters
        if self.kp1 != None and self.kp2 != None:
            if self.dp * self.lp * self.kp1 * self.kp2 != self.num_workers :
                raise ValueError("Product of dp, lp, kp1 and kp2 must be equal to number of workers")
        else:
            if self.dp * self.lp  != self.num_workers :
                raise ValueError("Product of dp, lp must be equal to number of workers")
            
        # Allow data parallelism to stay intra-node when dp_inter is False
        # (previously this raised an error). We now support dp using intra links.
        
        
        # Statistics Param
        self.tot_flop = 0
        self.tot_mem = 0
        self.tot_time = 0
        self.debug = False
        
        self.mode = mode
        default_policy = 'analytical'
        eb = getattr(hw_config, "execution_backend", None)
        if eb and getattr(eb, "model", "analytical") == "astra":
            default_policy = 'hybrid'
        self._astra_policy = astra_policy_override or default_policy

        self.network_model = NetworkModel(
            hw_config,
            self.precision,
            self.O,
            self.roofline,
            astra_policy=self._astra_policy,
        )
        
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
            if self.batch_size % self.dp != 0:
                raise ValueError("Batch size must be divisible by data parallelism degree")
            self.miniB = math.ceil(self.batch_size / self.dp) # mini-batch size for each data parallel node
            if self.miniB % self.mb != 0:
                print(f"miniB: {self.miniB}, mb: {self.mb}")
                raise ValueError("Batch size must be divisible by micro-batch size")
            self.microB = math.ceil(self.miniB / self.mb) if self.lp > 1 else self.miniB # micro-batch size for each pipeline stage

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

        # print("kp1: {}".format(self.kp_hidden_dim1))
        # print("kp2: {}".format(self.kp_hidden_dim2))

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

        # print("kp1: {}".format(self.kp_hidden_dim1))
        # print("kp2: {}".format(self.kp_hidden_dim2))
        # TODO: It is a hacky way of capturing assymetry across links within V100
        # move this to network topology and distinguish between inter and intra network
        if validating_v100:
            self.IBK1 = util.scale_down(self.IBK1, self.kp_hidden_dim1, "kp1")
            self.IBK2 = util.scale_down(self.IBK2, self.kp_hidden_dim2, "kp2")
            self.IBD = util.scale_down(self.IBD, self.dp, "dp")
            self.IBL = util.scale_down(self.IBL, self.lp, "lp")

    # Number of parameters
    def tot_param(self):
        embedding = self.V * self.D
        hidden = (2 * self.D + 1) * (self.G * self.D) * self.L
        projection = self.D * self.projection
        softmax = ((self.projection if proj else self.D) + 1) * self.V

        tot_param = embedding + hidden + projection + softmax
        return tot_param

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
                batch_size=self.B,
                hidden_dim=self.D,
                vocab_size=self.V,
                seq_len=self.S,
                num_layer=self.L,
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
                batch_size=self.B,
                hidden_dim=self.D,
                vocab_size=self.V,
                seq_len=self.S,
                num_layer=self.L,
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

    def roofline(self, flop, mem_access_, name="", info=False):
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

        if info:
            print(f"--- {name}: {(max_time * 1e3):.2f} ms, {flop:,} ---")
            for i in range(0, num_level):
                compare = ("memory", "<") if comp_int[i] < inflection_point[i] else ("compute", ">")
                print(f"L{i}: {comp_int[i]:.2f} {compare[1]} {inflection_point[i]} [{compare[0]}-bound]")
                print(f"time: {time[i] * 1e3} ms")
                print(
                    "Throughput = ",
                    self.th,
                    "BW = ",
                    self.memLayer[i].getThroughput(),
                    "mem_latency=",
                    self.memLayer[i].getLatency(),
                )  ################
                print()

        # print("Roofline: exited {}".format(name))
        return max_time


    def getGEMMTime(self, dim1, dim2, dim3, name, original=False):
        # Streaming best selection to avoid building large dicts
        best_time = float("inf")
        best_choice = None  # type: Optional[tuple]
        best_mem_access = None  # type: Optional[tuple]
        best_gemm = None  # type: Optional[TiledGEMM]
        best_metric = float("inf")

        # Iterate directly over candidates from tile module; no string orders
        for gemm in TiledGEMM.enumerate_candidates(self.core, self.memLayer, dim1, dim2, dim3, self.precision, original=False):
            if self.debug:
                print("===============================================================")
                print(f"inner_code: {gemm._inner_code}")
                print("===============================================================")

            GEMM_flop = gemm.GEMM_flop
            mem_access = gemm.mem_accesses

            # Adjust shared accesses per effective SMs
            mem_access_per_sm = list(mem_access)
            reuse_M = (dim1 + gemm.l2_M - 1) // gemm.l2_M
            reuse_K = (dim2 + gemm.l2_K - 1) // gemm.l2_K
            reuse_N = (dim3 + gemm.l2_N - 1) // gemm.l2_N
            eff_sm = min(self.core.num_bundle, reuse_M * reuse_K * reuse_N)
            if eff_sm > 0:
                mem_access_per_sm[1] = mem_access_per_sm[1] / eff_sm

            GEMM_time = self.roofline(GEMM_flop, mem_access_per_sm, name) + self.O

            tile_dims = (
                (gemm.l0_M, gemm.l0_K, gemm.l0_N),
                (gemm.l1_M, gemm.l1_K, gemm.l1_N),
                (gemm.l2_M, gemm.l2_K, gemm.l2_N),
            )
            key = (gemm._inner_code, tile_dims)

            # Tie-breaker metric identical to previous selection: hypot(dram, l2)
            metric = math.hypot(mem_access[3], mem_access[2])

            if (GEMM_time < best_time) or (GEMM_time == best_time and metric < best_metric):
                best_time = GEMM_time
                best_choice = key
                best_mem_access = mem_access
                best_gemm = gemm
                best_metric = metric

        # best_choice, best_mem_access must be set if there was at least one candidate
        mem_access = best_mem_access  # type: ignore

        if self.debug:
            print(repr(best_gemm))
            print(
                f"{name}: Best Time: {best_time * 1e3:,} ms, Best Inner: {best_choice[0]}, Best Tile: {best_choice[1]}\n"
            )

        # Inner code mapping for loop order (no strings):
        # 0 -> inner 'm' (weight stationary)
        # 1 -> inner 'k' (output stationary)
        # 2 -> inner 'n' (activation stationary)
        best_inner_code = best_choice[0]  # type: ignore[index]
        best_tile_dims = best_choice[1]  # type: ignore[index]
        return best_time, best_inner_code, best_tile_dims, mem_access

    def generateTileSpace(self, dim1=None, dim2=None, dim3=None, original=False):
        tile_space = []
        tiles = [None] * self.num_levels

        for level in range(0, self.num_levels - 1):
            memory = self.memLayer[level]
            # tiles[level] = self.getTileDims(memory)
            if not original and level == 2:
                tiles[level] = memory.getGEMMBasedTileDims(dim1, dim2, dim3)
            else:
                tiles[level] = memory.getTileDims()

        if self.num_levels == 1:
            tile_space = []
        elif self.num_levels == 2:
            tile_space = tiles[0]
        elif self.num_levels == 3:
            tile_space = [(x, y) for x in tiles[0] for y in tiles[1]]
        elif self.num_levels == 4:
            tile_space = []
            for x in tiles[2]:
                t1, t2, t3 = x
                if not original:
                    tiles[1] = self.memLayer[1].getGEMMBasedTileDims(t1, t2, t3)
                tile_strategy = [
                    (x, y, z) for y in tiles[1] for z in tiles[2]
                ]
                tile_space.extend(tile_strategy)
        else:
            raise NotImplementedError()
        
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
            num_accesses[0] = GEMM_flop * ((2 * reuse + 1) / (2 * reuse)) * 1/(2 * self.FMA_dims[0]) * self.precision

            # num_accesses[0] = (
            #     GEMM_flop
            #     * (
            #         2 * reuse * self.FMA_dims[0] * self.FMA_dims[1]
            #         + self.FMA_dims[0] ** 2
            #     )
            #     / (2 * reuse * self.FMA_dims[0] * (self.FMA_dims[1] - 1))
            #     * self.precision
            # )
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

    # Column-Row MM
    def getCf_kp1(self):
        # Multiply
        assert self.kp_hidden_type == 1
        assert self.kp_hidden_dim1 > 1
        assert (
            self.kp_hidden_dim1 % 4 == 0 or self.kp_hidden_dim1 == 2
        )  # 4 bc it is LSTM cell
        assert (2 * self.D) % self.kp_hidden_dim1 == 0
        GEMM_time, reduction_time = self.getDistGEMM_f_kp1(
            self.miniB, 2 * self.D, self.G * self.D, self.kp_hidden_dim1, "Cf_kp1"
        )

        # Pointwise ops: all the linear/non-linear ops after MM
        point_flop = self.miniB * (self.G * self.D / self.kp_hidden_dim1) * 5
        # 4 refers to the number of pointwise ops (mul + add +tanh + mul + tanh) on
        # the critical path
        point_mem = (
            self.precision
            * self.miniB
            * (self.G * self.D / self.kp_hidden_dim1)
            * (3 * 3 + 2 * 2)
        )
        # 3(3 memory access per operation with two input and one output)
        # 3(mul +  add + mul) on critical path
        # 2(2 memory access per operation with one input and one output)
        # 1(tanh) on critical path

        data_size = (
            4 * self.miniB * (self.G * self.D / self.kp_hidden_dim1) * self.precision
        )
        # 4 refers to the number of pointwise ops (mul + add + mul + tanh) on the
        # critical path whose inputs are located across different GPUs
        # NOTE:Assuming all communications can happpen in parallel
        mem_transfer = self.roofline(
            0,
            2 * data_size,
            name="Cf_kp1: memory accesses before and after data transfer over network",
        )
        # 2:  one read from the source and one write to the destination memory
        data_transfer = data_size / self.IBK1
        point_comm = mem_transfer + data_transfer

        point_time = (
            self.roofline(point_flop, point_mem, name="pointwise_cf_kp1")
            + 5 * self.O
            + point_comm
        )

        # print("GEMM_time = ", GEMM_time, "point_time= ", point_time)
        return GEMM_time + reduction_time + point_time

    # def getCb_kp1(self):
    #     # TODO:Add local accumulation of weights at every time step
    #     # Pointwise
    #     point_flop = (self.miniB) * (self.G * self.D / self.kp_hidden_dim1) * 5 + (
    #         2 * self.D * self.G * self.D / self.kp_hidden_dim1
    #     )  # local accumulation of wts
    #     # 4 refers to the number of pointwise ops (mul + add +tanh + mul) on
    #     # the critical path
    #     point_mem = (
    #         self.precision
    #         * self.miniB
    #         * (self.G * self.D / self.kp_hidden_dim1)
    #         * (3 * 3 + 2 * 2)
    #         + (2 * self.precision * self.D * self.G * self.D / self.kp_hidden_dim1) * 3
    #     )  # local accumulation of wts
    #     # 3(3 memory access per operation with two input and one output)
    #     # 3(mul +  add + mul) on critical path
    #     return GEMM_time + reduction_time + point_time

    def getCb_kp1(self):
        # TODO:Add local accumulation of weights at every time step
        # Pointwise
        point_flop = (self.miniB) * (self.G * self.D / self.kp_hidden_dim1) * 5 + (
            2 * self.D * self.G * self.D / self.kp_hidden_dim1
        )  # local accumulation of wts
        # 4 refers to the number of pointwise ops (mul + add +tanh + mul) on
        # the critical path
        point_mem = (
            self.precision
            * self.miniB
            * (self.G * self.D / self.kp_hidden_dim1)
            * (3 * 3 + 2 * 2)
            + (2 * self.precision * self.D * self.G * self.D / self.kp_hidden_dim1) * 3
        )  # local accumulation of wts
        # 3(3 memory access per operation with two input and one output)
        # 3(mul +  add + mul) on critical path
        # 2(2 memory access per operation with one input and one output)
        # 1(tanh) on critical path

        data_size = (
            4 * self.miniB * (self.G * self.D / self.kp_hidden_dim1) * self.precision
        )
        mem_transfer = self.roofline(
            0,
            2 * data_size,
            name="Cb_kp1: memory accesses before and after data transfer over network",
        )
        data_transfer = data_size / self.IBK1
        point_comm = mem_transfer + data_transfer
        # 3 refers to the number of pointwise ops (mul + tanh + mul) on
        # critical path whose inputs are located across different GPUs
        # NOTE:Assuming all communications can happpen in parallel

        point_time = (
            self.roofline(point_flop, point_mem, name="pointwise_Cb_kp1")
            + 5 * self.O
            + point_comm
        )

        # GEMM_wrt_act and wt is calculated under getDistGEMM_b_kp1
        GEMM_time, reduction_time = self.getDistGEMM_b_kp1(
            self.miniB, 2 * self.D, self.G * self.D, self.kp_hidden_dim1, "Cb_kp1"
        )

        if self.debug:
            print(
                "(gr) Hidden point_flop: {:,}, point_mem: {:,}\n".format(
                    int(point_flop / 1e9), int(point_mem / 1e9)
                )
            )

        return GEMM_time + reduction_time + point_time

    # Row-Column MM
    def getCf_kp2(self):
        # Multiply
        assert self.kp_hidden_type == 2
        assert self.kp_hidden_dim1 > 1 or self.kp_hidden_dim2 > 1
        assert (
            self.kp_hidden_dim2 % self.G == 0
            or self.kp_hidden_dim2 == 2
            or self.kp_hidden_dim2 == 1
        )
        assert self.miniB % self.kp_hidden_dim1 == 0 
        assert self.G * self.D % self.kp_hidden_dim2 == 0

        GEMM_time, reduction_time = self.getDistGEMM_f_kp2(
            self.miniB,
            2 * self.D,
            self.G * self.D,
            self.kp_hidden_dim1,
            self.kp_hidden_dim2,
            "Cf_kp2",
        )

        # Pointwise ops
        point_flop = (
            (self.miniB / self.kp_hidden_dim1)
            * (self.G * self.D / self.kp_hidden_dim2)
            * 5
        )
        # 4 refers to the number of pointwise ops (mul + add +tanh + mul) on
        # the critical path
        point_mem = int(
            self.precision
            * (self.miniB / self.kp_hidden_dim1)
            * (self.G * self.D / self.kp_hidden_dim2)
            * (3 * 3 + 2 * 2)
        )
        # 3(3 memory access per operation with two input and one output)
        # 3(mul +  add + mul) on critical path
        # 2(2 memory access per operation with one input and one output)
        # 1(tanh) on critical path
        data_size = (
            (self.miniB / self.kp_hidden_dim1)
            * (self.G * self.D / self.kp_hidden_dim2)
            * 4
            * self.precision
        )
        # 4 refers to the number of pointwise ops (mul + add + tanh + mul) whose inputs
        # across different GPU

        point_comm = 0
        if self.kp_softmax_dim2 > 1:
            mem_transfer = self.roofline(
                0,
                2 * data_size,
                name="Cf_kp2: memory accesses before and after data transfer over network",
            )
            data_transfer = data_size / self.IBK2
            point_comm = mem_transfer + data_transfer

        point_time = (
            self.roofline(point_flop, point_mem, name="pointwise_Cf_kp2")
            + 5 * self.O
            + point_comm
        )

        return GEMM_time + reduction_time + point_time

    def getCb_kp2(self):
        # Pointwise ops
        point_flop = (self.miniB / self.kp_hidden_dim1) * (
            self.G * self.D / self.kp_hidden_dim2
        ) * 5 + (
            2 * self.D * self.G * self.D / self.kp_hidden_dim2
        )  # local accumulation of wts
        # 4 refers to the number of pointwise ops (mul + add +tanh + mul) on
        # the critical path
        # kp_hidden_dim2 is for the reduction sum operation after doing outer product
        # for (B,4D)x(4D,2D).This is outerproduct due to the data distribution.
        point_mem = int(
            (
                self.precision
                * (self.miniB / self.kp_hidden_dim1)
                * (self.G * self.D / self.kp_hidden_dim2)
                * (3 * 3 + 2 * 2)
            )
            + (2 * self.precision * self.D * self.G * self.D / self.kp_hidden_dim2) * 3
        )  # local accumulation of wts
        # 3(3 memory access per operation with two input and one output)
        # 3(mul +  add + mul) on critical path
        # 2(2 memory access per operation with one input and one output)
        # 1(tanh) on critical path

        data_size = int(
            self.miniB * (self.G * self.D / self.kp_hidden_dim2) * 4 * self.precision
        )
        # 3 refers to the number of pointwise ops (mul + add +tanh + mul) on
        # 3 refers to the number of hops to gather i,f, o and c in each GPU
        # in order to perform (B,4D)x(4D,2D)

        point_comm = 0
        if self.kp_softmax_dim2 > 1:
            mem_transfer = self.roofline(
                0,
                2 * data_size,
                name="Cb_kp2:memory accesses before and after data transfer over network",
            )
            data_transfer = data_size / self.IBK2
            point_comm = mem_transfer + data_transfer

        point_time = (
            self.roofline(point_flop, point_mem, name="pointwise_Cb_kp2")
            + 5 * self.O
            + point_comm
        )

        GEMM_time, reduction_time = self.getDistGEMM_b_kp2(
            self.miniB,
            2 * self.D,
            self.G * self.D,
            self.kp_hidden_dim1,
            self.kp_hidden_dim2,
            "Cb_kp2",
        )

        if self.debug:
            print(
                "(gr) Hidden point_flop: {:,}, point_mem: {:,}\n".format(
                    int(point_flop / 1e9), int(point_mem / 1e9)
                )
            )

        return GEMM_time + reduction_time + point_time

    def getCf(self, m, k, n):
        # Add Biad adds
        """Get LSTM Cell Time on Forward Path"""
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

        return GEMM_time[0] + point_time

    def getCb(self):
        """Get LSTM Cell Time on Backward Path"""
        grad_act_time, _, _, _ = self.getGEMMTime(
            self.miniB, self.G * self.D, 2 * self.D, "Cb_act"
        )
        grad_wt_time, _, _, _ = self.getGEMMTime(
            2 * self.D, self.miniB, self.G * self.D, "Cb_wt"
        )

        GEMM_time = grad_act_time + grad_wt_time

        point_flop = (self.miniB * self.D * 5) + (
            2 * self.D * self.G * self.D
        )  # local accumulation of wts
        point_mem = (self.precision * self.miniB * self.D * (3 * 3 + 2 * 2)) + (
            2 * self.precision * self.D * self.G * self.D
        ) * 3  # local accumulation of wts
        point_time = (
            self.roofline(point_flop, point_mem, name="pointwise_Cb") + 5 * self.O
        )

        if self.debug:
            print(
                "(gr) Hidden/ point_flop: {:,}, point_mem: {:,} ".format(
                    int(point_flop / 1e9), int(point_mem / 1e9)
                )
            )
            print("Hidden point_time: {:,}\n".format(point_time))

        return GEMM_time + point_time

    # Reduction and all-gather time estimation

    def gradClipping(self, Dim0=None, Dim1=None, name=None):
        if Dim0 == None:
            Dim0 = 2 * self.D
        if Dim1 == None:
            Dim1 = self.G * self.D
        if name == None:
            name = "Hidden"
        # t_list[i] * clip_norm / max(global_norm, clip_norm)
        # where:
        # global_norm = sqrt(sum([l2norm(t)**2 for t in t_list]))

        norm_comp = Dim0 * Dim1 * 2
        # 1: power 2
        # 1: summ
        norm_mem = (Dim0 * Dim1 * 1) * self.precision
        # 1: one read per element and power it by 2 in local registers  anfd
        # summing to local acc

        clip_comp = Dim0 * Dim1 * 2
        # 1: pointwise mul
        # 1: pointwise div

        clip_mem = (Dim0 * Dim1 * 2) * self.precision
        # 1: one read for pointwise mul
        # 1: one write for pointwise div

        gradclip_mem = norm_mem + clip_mem
        gradclip_comp = norm_comp + clip_comp

        gradclip_time = self.roofline(
            gradclip_comp, gradclip_mem, name="pointwise-grad-clipping"
        )

        if self.debug:
            print(
                "({}) gradclip_flop: {:,}, gradclip_mem: {:,}".format(
                    name, gradclip_comp, gradclip_mem
                )
            )
            print("({}) gradclip_time: {:,}\n".format(name, gradclip_time))

        return gradclip_time

    def applyGrad(self, Dim0=None, Dim1=None, name=None):
        if Dim0 == None:
            Dim0 = 2 * self.D
        if Dim1 == None:
            Dim1 = self.G * self.D
        if name == None:
            name = "Hidden"

        applyGrad_comp = Dim0 * Dim1 * 3
        # 3: one pointwise division by  scalar after reducing all the gradients,
        #   one final addition of gradients to the weights
        #   one multiply by learning rate
        applyGrad_mem = (
            (1 * Dim0 * Dim1 * self.precision)
            + (2 * Dim0 * Dim1 * self.precision)
            + (1 * Dim0 * Dim1 * self.precision)
        )
        # 1: read for pointiwse div
        # 2: 1 reads and one write for pointwise add
        # 1: one write for multiplication by lr
        applyGrad_time = self.roofline(
            applyGrad_comp, applyGrad_mem, name="pointwise-applyGrad"
        )

        clip_time = self.gradClipping(Dim0, Dim1, name)

        grad_time = applyGrad_time + clip_time

        if self.debug:
            print(
                "({}) applyGrad_flop: {:,}, applyGrad_mem: {:,}".format(
                    name, applyGrad_comp, applyGrad_mem
                )
            )
            print("({}) applyGrad_time: {:,}\n".format(name, applyGrad_time))

        return grad_time



    # NOTE: for KP1/KP2 TP, CR AND RC:
    # In the flattening case for full_astrasim_flattened we assume that the backward pass does ALL GATHER as the last collective.
    # This is the default below. If this is changed, you will need to update the cross layer cross rank communication pipeline comms.
    # TODO(getDistGEMM_f_kp1):
    # - Confirm the intended collective is REDUCE-SCATTER (partial=True, allReduce=True).
    #   * Use RS if downstream consumers expect a sharded C[m, n] along kp1.
    #   * If the next op needs full C immediately, switch to ALL-REDUCE (partial=False, allReduce=True).
    def getDistGEMM_f_kp1(self, m, k, n, dim1, name):
        gemm_time = self.getGEMMTime(m, k // dim1, n, name)[0]

        # Sum-Reduce within each row for use in the next time step
        total_bytes = math.ceil(self.precision * m * n)
        reduction_time = self.network_model.collective(
            kind="reduce_scatter",
            size_bytes=total_bytes,
            participants=int(dim1),
            ib=self.IBK1,
            ll=self.LLK1,
            local_bytes=0.0,
            local_ops=0.0,
            debug_label=name or "comm",
        )
        return gemm_time, reduction_time

    def getDistGEMM_b_kp1(self, m, k, n, dim1, name):
        # calculate grad wrt. act (A'. W^T)
        # gather whole(A') before MM
        # A' is distibuted as columns across different nodes
        total_bytes = math.ceil(self.precision * m * n)
        size_bytes = math.ceil(total_bytes / dim1)
        reduction_time = self.network_model.collective(
            kind="all_gather",
            size_bytes=size_bytes,
            participants=int(dim1),
            ib=self.IBK1,
            ll=self.LLK1,
            local_bytes=3 * total_bytes,
            debug_label=name or "comm",
        )
        grad_wt_time, _, _, _ = self.getGEMMTime(k, (m // dim1), n, name + "wt")
        grad_act_time, _, _, _ = self.getGEMMTime(m, (n // dim1), k, name + "act")
        gemm_time = grad_wt_time + grad_act_time
        return gemm_time, reduction_time

    # all-reduce across M // kp1 GPUs
    def getDistGEMM_f_kp2(self, m, k, n, dim1, dim2, name):
        gemm_time = self.getGEMMTime(m // dim1, k, n // dim2, name)[0]
        total_bytes = math.ceil(self.precision * (m // dim1) * n)
        size_bytes = math.ceil(total_bytes / dim2)
        reduction_time = self.network_model.collective(
            kind="all_gather",
            size_bytes=size_bytes,
            participants=int(dim2),
            ib=self.IBK2,
            ll=self.LLK2,
            local_bytes=3 * total_bytes,
            debug_label=name or "comm",
        )
        return gemm_time, reduction_time


    # (getDistGEMM_b_kp2):
    # - Collectives used here should all be ALL-GATHERs (partial=False, allReduce=False):
    #   * wt1: gather row(A^T) across dim1
    #   * wt2: gather column grad(A') across dim1
    #   * act1: gather row grad(A') across dim2
    #   * act2: gather column(w^T) across dim2
    # - Remove heuristic "/2" scaling on wt1 and act2. !!!!
    # TODO TODO TODO: We have removed the heuristic /2 scaling for ANALYTICAL *AND* ASTRA mode. This will change results.
    # BEFORE MERGE: CAREFULLY CONSIDER AND ADDRESS THIS.
    # IF YOU SEE THIS MESSAGE IN MAINLINE DEEPFLOW PLEASE LET ME KNOW. -GK
    def getDistGEMM_b_kp2(self, m, k, n, dim1, dim2, name):
        ######################################################################################
        # calculate grad wrt. weights (A^T. grad(A'))
        # gather row(A^T)
        total_bytes = math.ceil(self.precision * k * m)
        size_bytes = math.ceil(total_bytes / dim1)
        reduction_time_wt1 = self.network_model.collective(
            kind="all_gather",
            size_bytes=size_bytes,
            participants=int(dim1),
            ib=self.IBK1,
            ll=self.LLK1,
            local_bytes=3 * total_bytes,
            debug_label=name or "comm",
        )
        # To calculate grad wrt weights (A^T, grad(A')),
        # gather column grad(A')
        total_bytes = math.ceil(self.precision * m * (n / dim2))
        size_bytes = math.ceil(total_bytes / dim1)
        reduction_time_wt2 = self.network_model.collective(
            kind="all_gather",
            size_bytes=size_bytes,
            participants=int(dim1),
            ib=self.IBK1,
            ll=self.LLK1,
            local_bytes=3 * total_bytes,
            debug_label=name or "comm",
        )

        ########################################################################################
        # calculate grad wrt. act (grad(A'). w^T)
        # gather row grad(A')
        total_bytes = math.ceil(self.precision * (m / dim1) * n)
        size_bytes = math.ceil(total_bytes / dim2)
        reduction_time_act1 = self.network_model.collective(
            kind="all_gather",
            size_bytes=size_bytes,
            participants=int(dim2),
            ib=self.IBK2,
            ll=self.LLK2,
            local_bytes=3 * total_bytes,
            debug_label=name or "comm",
        )
        # calculate grad wrt. act (grad(A'). w^T)
        # gather col(w^T)
        total_bytes = math.ceil(self.precision * k * n)
        size_bytes = math.ceil(total_bytes / dim2)
        reduction_time_act2 = self.network_model.collective(
            kind="all_gather",
            size_bytes=size_bytes,
            participants=int(dim2),
            ib=self.IBK2,
            ll=self.LLK2,
            local_bytes=3 * total_bytes,
            debug_label=name or "comm",
        )

        reduction_time = (
            reduction_time_wt1
            + reduction_time_wt2
            + reduction_time_act1
            + reduction_time_act2
        )

        # Multiply full grad_activation with shards of weights
        grad_wt_time, _, _, _ = self.getGEMMTime(k / dim1, m, n / dim2, name + "wt")
        # Multiply full grad-activation with shards of activations
        grad_act_time, _, _, _ = self.getGEMMTime(m / dim1, n, k / dim2, name + "act")

        GEMM_time = grad_wt_time + grad_act_time

        return GEMM_time, reduction_time


    def getDataParallelReduction(self, k, n, dim1, dim2, name):
        # k = 2 * self.D
        # n = 4 * self.D
        # dim1 = self.kp_hidden_dim1
        # dim2 = self.kp_hidden_dim2

        reduction_time_wt_kp = 0
        reduction_time_wt_dp = 0
        apply_grad_time = 0

        if self.kp_hidden_type == 1:  # CR
            reduction_time_wt_kp = 0
            total_bytes = math.ceil(self.precision * (k / dim1) * n)
            reduction_time_wt_dp = self.network_model.collective(
                kind="all_reduce",
                size_bytes=total_bytes,
                participants=int(self.dp),
                ib=self.IBD,
                ll=self.LLD,
                local_bytes=0.0,
                debug_label=name or "comm",
            )
            apply_grad_time = self.applyGrad(Dim0=k / dim1, Dim1=n, name=name)

        elif self.kp_hidden_type == 2:  # RC
            total_bytes = math.ceil(self.precision * (k / dim1) * (n / dim2))
            reduction_time_wt_dp = self.network_model.collective(
                kind="all_reduce",
                size_bytes=total_bytes,
                participants=int(self.dp),
                ib=self.IBD,
                ll=self.LLD,
                local_bytes=0.0,
                debug_label=name or "comm",
            )

            # gather col(w)
            total_bytes = math.ceil(self.precision * k * (n / dim2))
            size_bytes = math.ceil(total_bytes / dim1)
            reduction_time_wt_kp = self.network_model.collective(
                kind="all_gather",
                size_bytes=size_bytes,
                participants=int(dim1),
                ib=self.IBK1,
                ll=self.LLK1,
                local_bytes=3 * total_bytes,
                debug_label=name or "comm",
            )
            apply_grad_time = self.applyGrad(Dim0=k, Dim1=n / dim2, name=name)
        else:
            reduction_time_wt_kp = 0
            total_bytes = math.ceil(self.precision * k * n)
            reduction_time_wt_dp = self.network_model.collective(
                kind="all_reduce",
                size_bytes=total_bytes,
                participants=int(self.dp),
                ib=self.IBD,
                ll=self.LLD,
                local_bytes=0.0,
                debug_label=name or "comm",
            )
            apply_grad_time = self.applyGrad(Dim0=k, Dim1=n, name=name)
        if self.debug:
            print(f"reduction_time_wt_kp: {reduction_time_wt_kp}")
            print(f"reduction_time_wt_dp: {reduction_time_wt_dp}")
            print(f"apply_grad_time: {apply_grad_time}")
        reduction_time = reduction_time_wt_kp + reduction_time_wt_dp + apply_grad_time
        return reduction_time

    def getProjection_f(self):
        GEMM_time, _, _, _ = self.getGEMMTime(
            self.miniB, self.D, self.projection, "projection"
        )
        return GEMM_time

    def getProjection_b(self):
        grad_wt_time, _, _, _ = self.getGEMMTime(
            self.projection, self.miniB, self.D, "projection_b_wt"
        )
        grad_act_time, _, _, _ = self.getGEMMTime(
            self.miniB, self.projection, self.D, "projection_b_act"
        )

        GEMM_time = grad_wt_time + grad_act_time
        return GEMM_time

    def getProjection_f_kp1(self):
        assert self.kp_projection_type == 1
        assert self.kp_projection_dim1 > 1
        assert self.D % self.kp_projection_dim1 == 0

        GEMM_time, reduction_time = self.getDistGEMM_f_kp1(
            self.miniB, self.D, self.projection, self.kp_projection_dim1, "projection_f"
        )
        return GEMM_time + reduction_time

    def getProjection_b_kp1(self):
        assert self.kp_projection_type == 1
        assert self.kp_projection_dim1 > 1
        assert self.D % self.kp_projection_dim1 == 0

        GEMM_time, reduction_time = self.getDistGEMM_b_kp1(
            self.miniB, self.D, self.projection, self.kp_projection_dim1, "projection_b"
        )
        return GEMM_time + reduction_time

    def getProjection_f_kp2(self):
        assert self.kp_projection_type == 2
        assert self.kp_projection_dim1 > 1 or self.kp_projection_dim2 > 1
        assert (self.miniB) % self.kp_projection_dim1 == 0
        assert self.projection % self.kp_projection_dim2 == 0

        GEMM_time, reduction_time = self.getDistGEMM_f_kp2(
            self.miniB,
            self.D,
            self.projection,
            self.kp_projection_dim1,
            self.kp_projection_dim2,
            "projection_f",
        )
        return GEMM_time + reduction_time

    def getProjection_b_kp2(self):
        assert self.kp_projection_type == 2
        assert self.kp_projection_dim1 > 1 or self.kp_projection_dim2 > 1
        assert (self.miniB) % self.kp_projection_dim1 == 0
        assert self.projection % self.kp_projection_dim2 == 0

        GEMM_time, reduction_time = self.getDistGEMM_f_kp2(
            self.miniB,
            self.D,
            self.projection,
            self.kp_projection_dim1,
            self.kp_projection_dim2,
            "projection_b",
        )
        return GEMM_time + reduction_time

    def getSoftmax_f(self):
        GEMM_time, _, _, _ = self.getGEMMTime(
            self.miniB, (self.projection if proj else self.D), self.V, "softmax_f"
        )

        # Final matrix after GEMM has (B, V) dimensionality
        # We get exponential on each of the elements in a row
        # and then normalize them across the row
        # Therefore for each row we do V sigmoids and V-1 additions and V divisions
        # For each row, we read V sigmoids and write one
        # Up to here is 3 operations
        point_flop = self.miniB * (3 * self.V - 1)

        point_mem = self.precision * self.miniB * (7 * self.V)
        # 2: one read and one write for sigmoid
        # 1: one read for reduction
        # 1: one write for extension
        # 2: for pointwise division

        point_time = (
            self.roofline(point_flop, point_mem, name="pointwise-softmax-f")
            + 4 * self.O
        )

        if self.debug:
            print(
                "Softmax point_flop: {:,}, point_mem: {:,}".format(
                    int(point_flop / 1e9), int(point_mem / 1e9)
                )
            )
            print("point_time: {:,}\n".format(point_time))

        return GEMM_time + point_time

    # FIXME: where is the reduction time?
    def getSoftmax_b(self):
        grad_wt_time, _, _, _ = self.getGEMMTime(
            (self.projection if proj else self.D), self.miniB, self.V, "softmax_b_wt"
        )
        grad_act_time, _, _, _ = self.getGEMMTime(
            self.miniB, self.V, (self.projection if proj else self.D), "softmax_b_act"
        )

        GEMM_time = grad_wt_time + grad_act_time
        point_flop = self.miniB * self.V * 5
        # 1: one for one of the divisions, grad(A) (y=A/B)
        # 2: one for division and multiplication, grad(B)
        # 1: one for addition, copies turn into add
        # 1: one for sigmoid

        point_mem = self.precision * self.miniB * self.V * 11
        # 3: grad(A) in pointwise division
        # 3: grad(B) in pointwise division
        # 3: addition in copy backprop
        # 2: sigmoid

        point_time = (
            self.roofline(point_flop, point_mem, name="pointwise-softmax-b")
            + 4 * self.O
        )

        if self.debug:
            print(
                "(gr) Softmax point_flop: {:,}, point_mem: {:,}".format(
                    int(point_flop / 1e9), int(point_mem / 1e9)
                )
            )
            print("(gr) Softmax point_time: {:,}\n".format(point_time))

        return GEMM_time + point_time

    # Column-Row MM
    def getSoftmax_f_kp1(self):
        # Multiply
        assert self.kp_softmax_type == 1
        assert self.kp_softmax_dim1 > 1
        assert (self.projection if proj else self.D) % self.kp_softmax_dim1 == 0

        GEMM_time, reduction_time = self.getDistGEMM_f_kp1(
            self.miniB,
            self.projection if proj else self.D,
            self.V,
            self.kp_softmax_dim1,
            "softmax_f_kp1",
        )

        # Final matrix after GEMM has (B, V) dimensionality
        # We get exponential on each of the elements in a row
        # and then reduce all elements in the row to one
        # Therefore for each row we do V sigmoids and V-1 additions and V divisions
        # For each row, we read V sigmoids and write one
        # After GEMM reduction, each matrix has the full (B,V)
        # but each needs to only operate on 1/dim1 rows to get the reduction
        point_flop = (self.miniB / self.kp_softmax_dim1) * self.V * 3
        point_mem = self.precision * (self.miniB / self.kp_softmax_dim1) * self.V * 7
        # 2: sigmoid
        # 1: one read for reduction, the accumulate is a register
        # 1: one for write/extend the reduction result into all cells
        # 3: division needs one for read and one for write.

        total_bytes = math.ceil(self.precision * self.miniB * 1)
        size_bytes = math.ceil(total_bytes / self.kp_softmax_dim1)
        point_comm = self.network_model.collective(
            kind="all_gather",
            size_bytes=size_bytes,
            participants=int(self.kp_softmax_dim1),
            ib=self.IBK1,
            ll=self.LLK1,
            local_bytes=3 * total_bytes,
            debug_label="getSoftmax_f_kp1",
        )
        # communicating partail sum per row from one GPU to all others to perform sum reduce

        point_time = (
            self.roofline(point_flop, point_mem, name="pointwise-softmax-f-kp1")
            + self.O
            + point_comm
        )

        if self.debug:
            print(
                "Softmax point_flop: {:,}, point_mem: {:,}".format(
                    int(point_flop / 1e9), int(point_mem / 1e9)
                )
            )
            print(
                "Softmax GEMM_time: {:,}, point_time: {:,}\n".format(
                    GEMM_time, point_time
                )
            )

        return GEMM_time + reduction_time + point_time

    def getSoftmax_b_kp1(self):
        point_flop = (self.miniB) * (self.V / self.kp_softmax_dim1) * 5
        # 1: one for one of the divisions, grad(A) (y=A/B)
        # 2: one for division and multiplication, grad(B)
        # 1: one for addition, copies turn into add
        # 1: one for sigmoid

        point_mem = (
            self.precision * (self.miniB) * ((11 * self.V) / self.kp_softmax_dim1)
        )
        # 3: grad(A) in pointwise division
        # 3: grad(B) in pointwise division
        # 3: addition in copy backprop
        # 2: sigmoid

        point_comm = 0

        point_time = (
            self.roofline(point_flop, point_mem, name="pointwise-softmax-b-kp1")
            + self.O
            + point_comm
        )

        GEMM_time, reduction_time = self.getDistGEMM_b_kp1(
            self.miniB,
            self.projection if proj else self.D,
            self.V,
            self.kp_softmax_dim1,
            "softmax_b_kp1",
        )

        if self.debug:
            print(
                "(gr) Softmax point_flop: {:,}, point_mem: {:,}\n".format(
                    int(point_flop / 1e9), int(point_mem / 1e9)
                )
            )

        return reduction_time + GEMM_time + point_time

    # Row-Column MM
    def getSoftmax_f_kp2(self):
        # Multiply
        assert self.kp_softmax_type == 2
        assert self.kp_softmax_dim1 > 1 or self.kp_softmax_dim2 > 1
        assert (self.miniB) % self.kp_softmax_dim1 == 0
        assert (self.projection if proj else self.D) % self.kp_softmax_dim2 == 0

        GEMM_time, reduction_time = self.getDistGEMM_f_kp2(
            self.miniB,
            self.projection if proj else self.D,
            self.V,
            self.kp_softmax_dim1,
            self.kp_softmax_dim2,
            "softmax_f_kp2",
        )

        # Final matrix after GEMM has (B X S, V) dimensionality
        # We get exponential on each of the elements in a row
        # and then reduce all elements in the row to one
        # Therefore for each row we do V sigmoids and V-1 additions and V divisions
        # For each row, we read V sigmoids and write one

        point_flop = (
            (self.miniB / self.kp_softmax_dim1) * (self.V / self.kp_softmax_dim2) * 3
        )
        point_mem = (
            self.precision
            * (self.miniB / self.kp_softmax_dim1)
            * (self.V / self.kp_softmax_dim2)
            * 7
        )
        # 2: sigmoid
        # 1: one read for reduction, the accumulate is a register
        # 1: one for write/broadcast the reduction result into all cells
        # 3: division needs one for read and one for write.

        data_size = (
            self.precision
            * (self.miniB / self.kp_softmax_dim1)
            * (self.kp_softmax_dim2)
        )

        point_comm = 0
        if self.kp_softmax_dim2 > 1:
            point_comm = self.network_model.collective(
                kind="all_to_all",
                size_bytes=data_size,
                participants=self.kp_softmax_dim2,
                ib=self.IBK2,
                ll=0.0,
                local_bytes=2 * data_size,
                debug_label="softmax_kp2",
            )

        point_time = (
            self.roofline(point_flop, point_mem, name="pointwise-Softmax_f_kp2")
            + self.O
            + point_comm
        )

        if self.debug:
            print(
                "Softmax point_flop: {:,}, point_mem: {:,}".format(
                    int(point_flop / 1e9), int(point_mem / 1e9)
                )
            )
            print(
                "Softmax GEMM_time: {:,}, point_time: {:,}\n".format(
                    GEMM_time, point_time
                )
            )

        return GEMM_time + point_time + reduction_time

    def getSoftmax_b_kp2(self):
        point_flop = (
            (self.miniB / self.kp_softmax_dim1) * (self.V / self.kp_softmax_dim2) * 5
        )
        # 1: one for one of the divisions, grad(A) (y=A/B)
        # 2: one for division and multiplication, grad(B)
        # 1: one for addition, copies turn into add
        # 1: one for sigmoid

        point_mem = (
            self.precision
            * (self.miniB / self.kp_softmax_dim1)
            * ((11 * self.V) / self.kp_softmax_dim2)
        )
        # 3: grad(A) in pointwise division
        # 3: grad(B) in pointwise division
        # 3: addition in copy backprop
        # 2: sigmoid

        point_comm = 0

        point_time = (
            self.roofline(point_flop, point_mem, name="pointwise-Softmax_b_kp2")
            + self.O
            + point_comm
        )

        GEMM_time, reduction_time = self.getDistGEMM_b_kp2(
            self.miniB,
            self.projection if proj else self.D,
            self.V,
            self.kp_softmax_dim1,
            self.kp_softmax_dim2,
            "softmax_b_kp2",
        )

        if self.debug:
            print(
                "(gr) Softmax point_flop: {:,}, point_mem: {:,}\n".format(
                    int(point_flop / 1e9), int(point_mem / 1e9)
                )
            )

        return reduction_time + GEMM_time + point_time

    def getEmbedding_f(self):
        embedding_mem = 2 * (self.miniB * self.D * self.precision)
        embedding_time = self.roofline(0, embedding_mem, name="embedding_f") + self.O
        if self.H2Dbw and self.H2Dbw > 0:
            embedding_transfer_time = embedding_mem / self.H2Dbw
        else:
            embedding_transfer_time = 0.0
        if self.debug:
            print("Embedding_mem: {:,}".format(int(embedding_mem / 1e9)))
        return embedding_time + embedding_transfer_time

    def getEmbedding_b(self):
        # p2p_data_transfer = (self.precision * self.miniB * self.D)
        # data_transfer_time  = 0 if (self.dp == 1) else (float("inf") if (self.IBD == 0) else (((p2p_data_transfer) / self.IBD + self.LLD) * 2 * (self.dp -1 )))

        embedding_mem = 2 * self.miniB * self.D * self.precision
        embedding_mem_time = self.roofline(0, embedding_mem, name="embedding_b") + self.O

        if self.debug:
            print("(gr) Embedding_mem: {:,}".format(int(embedding_mem / 1e9)))
        # return data_transfer_time + embedding_mem_time
        return embedding_mem_time

    def getEmbedding_f_kp1(self):
        # Each GPU has only a portion of the activations since each GPU had only a row of the weights
        total_bytes = math.ceil(self.precision * self.miniB * self.D)
        size_bytes = math.ceil(total_bytes / self.kp_embedding_dim1)
        reduction_time_act = self.network_model.collective(
            kind="all_gather",
            size_bytes=size_bytes,
            participants=int(self.kp_embedding_dim1),
            ib=self.IBK1,
            ll=self.LLK1,
            local_bytes=3 * total_bytes,
            debug_label="getEmbedding_f_kp1",
        )
        embedding_mem = 2 * (self.miniB * self.D * self.precision)
        # embedding_time = (embedding_mem)/ (self.mem_bw) + self.mem_latency + self.O
        embedding_time = self.roofline(0, embedding_mem, name="embedding_f") + self.O
        if self.debug:
            print("Embedding_mem: {:,}".format(int(embedding_mem / 1e9)))
        return embedding_time + reduction_time_act

    def getEmbedding_b_kp1(self):
        # Activations from previous row arrive in column fasion, they need to be gathered
        # before applying them to the local portion of the embeddings
        total_bytes = math.ceil(self.precision * self.miniB * self.D)
        size_bytes = math.ceil(total_bytes / self.kp_embedding_dim1)
        reduction_time_act = self.network_model.collective(
            kind="all_gather",
            size_bytes=size_bytes,
            participants=int(self.kp_embedding_dim1),
            ib=self.IBK1,
            ll=self.LLK1,
            local_bytes=3 * total_bytes,
            debug_label="getEmbedding_b_kp1",
        )
        # Each GPU would read through the entire actication and write as many at most as many of B rows
        embedding_mem = 2 * self.miniB * self.D * self.precision
        embedding_mem_time = (
            self.roofline(0, embedding_mem, name="embedding_b") + self.O
        )

        if self.debug:
            print("(gr) Embedding_mem: {:,}".format(int(embedding_mem / 1e9)))
        return embedding_mem_time + reduction_time_act

    def getEmbedding_f_kp2(self):
        embedding_mem = 2 * (
            (self.miniB / self.kp_embedding_dim1)
            * (self.D / self.kp_embedding_dim2)
            * self.precision
        )
        embedding_time = self.roofline(0, embedding_mem, name="embedding_f") + self.O
        if self.debug:
            print("Embedding_mem: {:,}".format(int(embedding_mem / 1e9)))
        return embedding_time

    def getEmbedding_b_kp2(self):
        # Every GPU will update a little tile of the embedding
        # need to be gathered after the update across the rows of each column
        total_bytes = math.ceil(self.precision * self.miniB * (self.D / self.kp_embedding_dim2))
        size_bytes = math.ceil(total_bytes / self.kp_embedding_dim1)
        reduction_time_act = self.network_model.collective(
            kind="all_gather",
            size_bytes=size_bytes,
            participants=int(self.kp_embedding_dim1),
            ib=self.IBK1,
            ll=self.LLK1,
            local_bytes=3 * total_bytes,
            debug_label="getEmbedding_b_kp2",
        )

        embedding_mem = (
            2
            * (self.miniB / self.kp_embedding_dim1)
            * (self.D / self.kp_embedding_dim2)
            * self.precision
        )
        embedding_mem_time = (
            self.roofline(0, embedding_mem, name="embedding_b") + self.O
        )

        if self.debug:
            print("(gr) Embedding_mem: {:,}".format(int(embedding_mem / 1e9)))
        return embedding_mem_time + reduction_time_act


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
            Cf = self.getCf(m=B, k=2 * D, n=G * D)
            Cb = self.getCb()
            w_size = self.precision * B * D
            Tf = self.network_model.collective(
                kind="pipeline",
                size_bytes=w_size,
                participants=1,
                ib=self.IBL,
                ll=self.LLL,
                local_bytes=2 * w_size,  # Memory modeling: read + write
                debug_label="inter_layer",
            ) if self.lp > 1 else 0
        elif self.kp_hidden_type == 1:  # CR
            Cf = self.getCf_kp1()
            Cb = self.getCb_kp1()
            w_size = self.precision * B * (D / self.kp_hidden_dim1)
            Tf = self.network_model.collective(
                kind="pipeline",
                size_bytes=w_size,
                participants=1,
                ib=self.IBL,
                ll=self.LLL,
                local_bytes=2 * w_size,  # Memory modeling: read + write
                debug_label="inter_layer",
            ) if self.lp > 1 else 0
        elif self.kp_hidden_type == 2:  # RC
            Cf = self.getCf_kp2()
            Cb = self.getCb_kp2()
            w_size = self.precision * (B / self.kp_hidden_dim1) * (D / self.kp_hidden_dim2)
            Tf = self.network_model.collective(
                kind="pipeline",
                size_bytes=w_size,
                participants=1,
                ib=self.IBL,
                ll=self.LLL,
                local_bytes=2 * w_size,  # Memory modeling: read + write
                debug_label="inter_layer",
            ) if self.lp > 1 else 0
        else:
            print("Incorrect distributed GEMM type, 1: Column-Row, 2: Row-Column")
            sys.exit()

        if self.lp == 1:
            Tf = 0

        Tb = Tf

        if self.kp_softmax_type == -1:
            Sf = self.getSoftmax_f()
            Sb = self.getSoftmax_b()
        elif self.kp_softmax_type == 1:  # CR
            Sf = self.getSoftmax_f_kp1()
            Sb = self.getSoftmax_b_kp1()
        elif self.kp_softmax_type == 2:  # RC
            Sf = self.getSoftmax_f_kp2()
            Sb = self.getSoftmax_b_kp2()
        else:
            print("Incorrect distributed GEMM type, 1: Column-Row, 2: Row-Column")
            sys.exit()

        if self.kp_embedding_type == -1:
            Ef = self.getEmbedding_f()
            Eb = self.getEmbedding_b()
        elif self.kp_embedding_type == 1:  # CR
            Ef = self.getEmbedding_f_kp1()
            Eb = self.getEmbedding_b_kp1()
        elif self.kp_embedding_type == 2:  # RC
            Ef = self.getEmbedding_f_kp2()
            Eb = self.getEmbedding_b_kp2()
        else:
            print("Incorrect distributed GEMM type, 1: Column-Row, 2: Row-Column")
            sys.exit()

        Rc = self.getDataParallelReduction(
            k=2 * D,
            n=G * D,
            dim1=self.kp_hidden_dim1,
            dim2=self.kp_hidden_dim2,
            name="Hidden Reduction",
        )
        Rs = self.getDataParallelReduction(
            k=D,
            n=V,
            dim1=self.kp_softmax_dim1,
            dim2=self.kp_softmax_dim2,
            name="Softmax Reduction",
        )
        Re = self.getDataParallelReduction(
            k=V,
            n=D,
            dim1=self.kp_embedding_dim1,
            dim2=self.kp_embedding_dim2,
            name="Embedding Reduction",
        )

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

            print(
                "Cf: {} Cb: {} softmax_f: {} softmax_b: {} embedding_f: {} embedding_b: {} "
                "Rs: {} Rc: {} Re: {}\n".format(Cf, Cb, Sf, Sb, Ef, Eb, Rs, Rc, Re)
            )

        g = Graph(
            num_seq=S,
            num_layer=L,
            lp=lp,
            Ef=Ef,
            Cf=Cf,
            Sf=Sf,
            Tf=Tf,
            Eb=Eb,
            Cb=Cb,
            Sb=Sb,
            Tb=Tb,
            Re=Re,
            Rc=Rc,
            Rs=Rs,
        )

        fw_roots = g.construct_fwd_graph()
        bw_roots = g.construct_bwd_graph()

        time_fw = g.simulate(fw_roots[0], 0)
        time_bw = g.simulate(bw_roots[g.num_seq - 1], g.lp - 1)

        self.tot_time = time_fw + time_bw
        tot_param = self.tot_param()

        return self.tot_time, tot_param

    def getTime(self):
        return self.tot_time
