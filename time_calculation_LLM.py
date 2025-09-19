#!/tools/lm-venv/py3.6-tf-1.3.0-svail/bin/python

# import click
import math
import os
import pickle
import sys
import config
import shutil
import itertools
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Tuple, Optional, List, Set
# import numpy as np
import simulate_LLM
from parallelism import Parallelism
from topology import Topology
from simulate_LLM import Graph
import LLM_util
from hw_component import Core, MemoryHierarchy, Network
from model import Model_LSTM, Model_GEMM, Model_LLM
from tile import TiledGEMM, formatBytes
from astrasim_integration import run_astrasim_graph
from astra_comparison import run_astra_simulation_only_onepath

from simulate_LLM import visualize_graph
from time_calculation import TimeCalculation
# algByte = False  # algorithmic ops false
# proj = False  # consider projection layer, turn off for end-2-end validation, as baeline model does not have projection layer
validating_v100 = True

debug = True
all_reduce = "every layer"  #"the end"  "every layer"


showing_ms = False # Show time in ms if True    show time in us if False
if showing_ms:
    m=1e3
    second = "ms"
else:
    m=1e6
    second = "us"


class ExecutionMode(Enum):
    ANALYTICAL = "analytical"
    HYBRID = "hybrid"
    HYBRID_CONGESTION = "hybrid_congestion"
    FULL_ASTRASIM = "full_astrasim"


@dataclass
class ExecutionResult:
    total_time: float
    graph_root: Any
    mode: ExecutionMode


class TimeCalculationLLM(TimeCalculation):
    def __init__(self, hw_config, model_config, mode):
# Mode parameter
        execution_mode = self._derive_execution_mode(hw_config)
        astra_policy = self._map_execution_mode_to_policy(execution_mode)

        super().__init__(
            hw_config,
            model_config,
            mode,
            astra_policy_override=astra_policy,
        )
        self.execution_mode = execution_mode
        self._reduction_total_llm = 0.0
        self.all_reduce = all_reduce # when the reduce happens in data parallelism options: "the end"  "every layer"
        self.pipeline_graph: Optional[Graph] = None
        self.pipeline_root: Optional[Any] = None
        self.pipeline_interconnect: Optional[Dict[str, Tuple[float, float]]] = None
        self.transformer_graph: Optional[Graph] = None
        self.transformer_graph_root: Optional[Any] = None
        self.transformer_analytical_time: Optional[float] = None
        self.transformer_astrasim_time: Optional[float] = None
        self.transformer_astrasim_per_rank: Optional[List[float]] = None
        self.transformer_time_scale: Optional[float] = None
        self.pipeline_astrasim_time: Optional[float] = None
        self.pipeline_astrasim_per_rank: Optional[List[float]] = None

    @staticmethod
    def _derive_execution_mode(hw_config) -> ExecutionMode:
        backend = getattr(hw_config, "execution_backend", None)
        if not backend or getattr(backend, "model", "analytical").lower() != "astra":
            return ExecutionMode.ANALYTICAL

        mode_str = "hybrid"
        astra_cfg = getattr(backend, "astra", None)
        if astra_cfg and getattr(astra_cfg, "mode", None):
            mode_str = str(astra_cfg.mode).lower()

        for candidate in ExecutionMode:
            if candidate.value == mode_str:
                return candidate
        print(f"[WARN] Unknown execution mode '{mode_str}', defaulting to 'hybrid'.")
        return ExecutionMode.HYBRID

    @staticmethod
    def _map_execution_mode_to_policy(mode: ExecutionMode) -> str:
        if mode == ExecutionMode.ANALYTICAL:
            return 'analytical'
        if mode == ExecutionMode.FULL_ASTRASIM:
            return 'full'
        # Treat hybrid congestion the same as hybrid for now
        return 'hybrid'

    def _distributed_gemm_forward(self, m: int, k: int, n: int, name: str) -> Tuple[float, float]:
        if self.t is None or (self.kp1 == 1 and self.kp2 == 1):
            gemm_time = self.getGEMMTime(m, k, n, name)[0]
            return gemm_time, 0.0
        if self.t == "CR":
            return self.getDistGEMM_f_kp1(m, k, n, self.kp1, name)
        if self.t == "RC":
            return self.getDistGEMM_f_kp2(m, k, n, self.kp1, self.kp2, name)
        raise ValueError(f"Invalid tensor parallel strategy: {self.t}")

    def _distributed_gemm_backward(self, m: int, k: int, n: int, name: str) -> Tuple[float, float]:
        if self.t is None or (self.kp1 == 1 and self.kp2 == 1):
            grad_act_time, _, _, _ = self.getGEMMTime(m, n, k, f"{name}_act")
            grad_wt_time, _, _, _ = self.getGEMMTime(k, m, n, f"{name}_wt")
            return grad_act_time + grad_wt_time, 0.0
        if self.t == "CR":
            return self.getDistGEMM_b_kp1(m, k, n, self.kp1, name)
        if self.t == "RC":
            return self.getDistGEMM_b_kp2(m, k, n, self.kp1, self.kp2, name)
        raise ValueError(f"Invalid tensor parallel strategy: {self.t}")

    def get_node_f(self, gemm, name):

        """Get node Time on Forward Path
            For GEMM operations
        """
        m = gemm[0]
        k = gemm[1]
        n = gemm[2]
        gemm_time, reduction_time = self._distributed_gemm_forward(m, k, n, name)

        if self.debug:
            print(f"{name} GEMM_time: {gemm_time:,}; reduction_time: {reduction_time:,}")

        total = gemm_time + reduction_time
        self._reduction_total_llm += reduction_time
        return total

    def get_node_b(self, gemm, name):
        """Get node Time on Backward Path
            For GEMM operations
        """
        m = gemm[0]
        k = gemm[1]
        n = gemm[2]
        gemm_time, reduction_time = self._distributed_gemm_backward(m, k, n, name)

        total = gemm_time + reduction_time
        self._reduction_total_llm += reduction_time
        return total

    def getEmbedding_f(self):
        """
        Calculates the total time required for embedding operations, including computation and data transfer.
        """
        batch = self._effective_transformer_batch()
        embedding_mem = 2 * self.seq_len * batch * self.hidden_dim * self.precision
        embedding_time = self.roofline(0, embedding_mem, name="embedding_f") + self.O
        if self.h2d_bandwidth and self.h2d_bandwidth > 0:
            embedding_transfer_time = embedding_mem / self.h2d_bandwidth
        else:
            embedding_transfer_time = 0.0
        if self.debug:
            print(
                "Embedding_mem: {:,}, transfer_time: {:.6f}".format(
                    int(embedding_mem / 1e9), embedding_transfer_time
                )
            )
        return embedding_time + embedding_transfer_time

    def getLinearSoftmax_f(self, gemm):
        """
        Calculates the total computation time for a linear softmax operation using GEMM (General Matrix Multiply) and pointwise operations.
        """
        m = gemm[0]
        k = gemm[1]
        n = gemm[2]
        
        gemm_time, reduction_time = self._distributed_gemm_forward(m, k, n, "linear_softmax_f")
        point_flop = m * (3 * n - 1)
        point_mem = self.precision * m * (7 * n)
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

        total = gemm_time + reduction_time + point_time
        self._reduction_total_llm += reduction_time
        return total
    def getScaleSoftmax_f(self, gemm):

        m = gemm[0] # get the gemm shape of former layer which is attention_score layer here
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
        m = gemm[0] # get the gemm shape of former layer which is output_proj layer here
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
        m = gemm[0] # get the gemm shape of former layer 
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
    def getLinearSoftmax_b(self, gemm):
        m = gemm[0]
        k = gemm[1]
        n = gemm[2]
        gemm_time, reduction_time = self._distributed_gemm_backward(m, k, n, "linear_softmax_b")
        point_flop = m * n * 5
        # 1: one for one of the divisions, grad(A) (y=A/B)
        # 2: one for division and multiplication, grad(B)
        # 1: one for addition, copies turn into add
        # 1: one for sigmoid

        point_mem = self.precision * m * 11
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

        total = gemm_time + reduction_time + point_time
        self._reduction_total_llm += reduction_time
        return total
    
    def getEmbedding_b(self):
        batch = self._effective_transformer_batch()
        embedding_mem = 2 * self.seq_len * batch * self.hidden_dim * self.precision
        embedding_mem_time = self.roofline(0, embedding_mem, name="embedding_b") + self.O

        if self.debug:
            print("(gr) Embedding_mem: {:,}".format(int(embedding_mem / 1e9)))
        return embedding_mem_time
    
    def check_memory(self, hw_config, model_config): #check whether memory usage exceeds capacity
        """Check memory usage."""
        total_mem_capacity = self.memory_capacity # in bytes
        print(f"Total Memory Capacity: {total_mem_capacity / 1e9} GB")
        total_mem = LLM_util.getTotMemReq(hw_config, model_config)[0]
        print(f"Total Memory Usage estimation: {total_mem / 1e9} GB")
        if total_mem > total_mem_capacity:
            print("Warning: Total memory usage exceeds memory capacity!")
            sys.exit("Program terminated due to memory capacity exceeded.")

        return
    def getDPOverhead(self, d, ffn_dim, n_layers): #calculate the reduction time and apply_grad_time for data parallelism 

        reduction_time = 0
        apply_grad_time = 0

        reduction_time += self.getR(Dim0=d, Dim1=3*d, p=self.dp, ib=self.IBD, ll=self.LLD, partial=False, allReduce=True, name="qkv_proj reduction") #get the reduction time between all nodes in one data parallel group
        apply_grad_time += self.applyGrad(Dim0=d, Dim1=3*d, name="qkv_proj grad")

        reduction_time += self.getR(Dim0=d, Dim1=d, p=self.dp, ib=self.IBD, ll=self.LLD, partial=False, allReduce=True, name="output_proj reduction")
        apply_grad_time += self.applyGrad(Dim0=d, Dim1=d, name="output_proj grad")

        reduction_time += 2*self.getR(Dim0=ffn_dim, Dim1=d, p=self.dp, ib=self.IBD, ll=self.LLD, partial=False, allReduce=True, name="ffn reduction")
        apply_grad_time += 2*self.applyGrad(Dim0=ffn_dim, Dim1=d, name="ffn grad")
        print(f"reduction_time: {reduction_time}")
        print(f"apply_grad_time: {apply_grad_time}")
        
        if all_reduce == "the end":
            reduction_time *= n_layers
            apply_grad_time *= n_layers
        elif all_reduce == "every layer": #return the reduction time for each layer
            pass
        else:
            sys.exit("Invalid all_reduce option")



        return reduction_time , apply_grad_time
    
    
    def getInterLayerCommLatency_LLM(self, batch_size, hidden_dim, seq_len): #calculate the cross-layer communication latency
        w = 0
        w_size = 0
        if self.lp > 1:
            w_size = self.precision * batch_size * hidden_dim * seq_len
            transfer_time = w_size / self.IBL + self.LLL
            mem_time = self.roofline(0, 2 * w_size, name="inter_layer")
            # 2: read from memory of previous layer and write to the memory of the next layer
            w = mem_time + transfer_time
        return w, w_size
    def getDataParallelReductionSizes(self, d, ffn_dim):
        """Calculate communication sizes for data parallel reductions (no timing)."""
        if not getattr(self, "dp", 1) or self.dp <= 1:
            # No communication needed for dp=1
            return {
                'qkv_size': 0,
                'output_size': 0,
                'ffn_size': 0,
                'total_size': 0
            }

        # Calculate sizes only (no timing)
        qkv_size = math.ceil(self.precision * d * 3 * d)
        output_size = math.ceil(self.precision * d * d)
        ffn_size = math.ceil(self.precision * ffn_dim * d)
        total_size = qkv_size + output_size + 2 * ffn_size  # FFN appears twice

        return {
            'qkv_size': qkv_size,
            'output_size': output_size,
            'ffn_size': ffn_size,
            'total_size': total_size
        }

    def getDataParallelLocalComputation(self, d, ffn_dim):
        """Calculate local computation times for apply_grad operations."""
        qkv_local = self.applyGrad(Dim0=d, Dim1=3*d, name="qkv_proj reduction")
        output_local = self.applyGrad(Dim0=d, Dim1=d, name="output_proj reduction")
        ffn_local = 2 * self.applyGrad(Dim0=ffn_dim, Dim1=d, name="ffn reduction")

        return {
            'qkv_local': qkv_local,
            'output_local': output_local,
            'ffn_local': ffn_local,
            'total_local': qkv_local + output_local + ffn_local
        }

    def getDataParallelReduction_LLM(self, d, ffn_dim):
        # If no data parallelism, still apply gradients locally but no cross-device reduction
        if not getattr(self, "dp", 1) or self.dp <= 1:
            apply_grad_time = 0.0
            apply_grad_time += self.applyGrad(Dim0=d, Dim1=3*d, name="qkv_proj reduction")
            apply_grad_time += self.applyGrad(Dim0=d, Dim1=d, name="output_proj reduction")
            apply_grad_time += 2 * self.applyGrad(Dim0=ffn_dim, Dim1=d, name="ffn reduction")
            if self.debug:
                print(f"(dp=1) apply_grad_time: {apply_grad_time}")
            return apply_grad_time

        # k = 2 * self.D
        # n = 4 * self.D
        # dim1 = self.kp_hidden_dim1
        # dim2 = self.kp_hidden_dim2
        w_data = 4*d*d + 2*ffn_dim*d # total parameters need to be reduced
        reduction_time = 0.0
        apply_grad_time = 0.0

        total_bytes = math.ceil(self.precision * d * 3 * d)
        reduction_time += self.network_model.collective(
            kind="all_reduce",
            size_bytes=total_bytes,
            participants=int(self.dp),
            ib=self.IBD,
            ll=self.LLD,
            local_bytes=0.0,
            debug_label="qkv_proj reduction",
        )
        apply_grad_time += self.applyGrad(Dim0=d, Dim1=3*d, name="qkv_proj reduction")

        total_bytes = math.ceil(self.precision * d * d)
        reduction_time += self.network_model.collective(
            kind="all_reduce",
            size_bytes=total_bytes,
            participants=int(self.dp),
            ib=self.IBD,
            ll=self.LLD,
            local_bytes=0.0,
            debug_label="output_proj reduction",
        )
        apply_grad_time += self.applyGrad(Dim0=d, Dim1=d, name="output_proj reduction")

        total_bytes = math.ceil(self.precision * ffn_dim * d)
        reduction_time += 2 * self.network_model.collective(
            kind="all_reduce",
            size_bytes=total_bytes,
            participants=int(self.dp),
            ib=self.IBD,
            ll=self.LLD,
            local_bytes=0.0,
            debug_label="ffn reduction",
        )
        apply_grad_time += 2 * self.applyGrad(Dim0=ffn_dim, Dim1=d, name="ffn reduction")

        if self.debug:
            print(f"reduction_time: {reduction_time}")
            print(f"apply_grad_time: {apply_grad_time}")

        return reduction_time + apply_grad_time
    
    
    
    def getNodeLatency(self, batch_size, vocab_size, hidden_dim, seq_len, num_heads, ffn_dim):
        """Calculate latency for each node in the computation graph."""

        # Process GEMM shapes
        gemm_3d = LLM_util.process_gemm_shapes( #get gemm shape for each layer in transformer and linear softmax layer
            batch_size, seq_len, hidden_dim, num_heads, ffn_dim, vocab_size, option="multiply_batch_into_m"
        )
        gemm_qkv_proj, gemm_attention_score, gemm_attention_output, gemm_output_proj, gemm_ffn1, gemm_ffn2, gemm_linear = gemm_3d

        # Calculate time for each node
        embedding_f = self.getEmbedding_f()
        embedding_b = self.getEmbedding_b()

        qkv_proj_f, qkv_proj_b = (
            self.get_node_f(gemm=gemm_qkv_proj, name="qkv_projection_f"),
            self.get_node_b(gemm=gemm_qkv_proj, name="qkv_projection_b")
        )
        attention_score_f, attention_score_b = (
            self.get_node_f(gemm=gemm_attention_score, name="attention_score_f"),
            self.get_node_b(gemm=gemm_attention_score, name="attention_score_b")
        )
        attention_scale_softmax_f, attention_scale_softmax_b = (
            self.getScaleSoftmax_f(gemm=gemm_attention_score),
            self.getScaleSoftmax_b(gemm=gemm_attention_score)
        )
        attention_output_f, attention_output_b = (
            self.get_node_f(gemm=gemm_attention_output, name="attention_output_f"),
            self.get_node_b(gemm=gemm_attention_output, name="attention_output_b")
        )
        output_proj_f, output_proj_b = (
            self.get_node_f(gemm=gemm_output_proj, name="output_projection_f"),
            self.get_node_b(gemm=gemm_output_proj, name="output_projection_b")
        )
        residual1_f, residual1_b = (
            self.getResidual_f(gemm=gemm_output_proj),
            self.getResidual_b(gemm=gemm_output_proj)
        )
        layernorm1_f, layernorm1_b = (
            self.getLayernorm_f(gemm=gemm_output_proj),
            self.getLayernorm_b(gemm=gemm_output_proj)
        )
        ffn1_f, ffn1_b = (
            self.get_node_f(gemm=gemm_ffn1, name="ffn_f"),
            self.get_node_b(gemm=gemm_ffn1, name="ffn_b")
        )
        ffn2_f, ffn2_b = (
            self.get_node_f(gemm=gemm_ffn2, name="ffn2_f"),
            self.get_node_b(gemm=gemm_ffn2, name="ffn2_b")
        )
        residual2_f, residual2_b = (
            self.getResidual_f(gemm=gemm_ffn2),
            self.getResidual_b(gemm=gemm_ffn2)
        )
        layernorm2_f, layernorm2_b = (
            self.getLayernorm_f(gemm=gemm_ffn2),
            self.getLayernorm_b(gemm=gemm_ffn2)
        )
        linear_softmax_f, linear_softmax_b = (
            self.getLinearSoftmax_f(gemm=gemm_linear),
            self.getLinearSoftmax_b(gemm=gemm_linear)
        )

        # Calculate MHA and FFN times
        mha_time_f = (
            qkv_proj_f + attention_score_f + attention_output_f + output_proj_f + attention_scale_softmax_f
        )
        ffn_time_f = ffn1_f + ffn2_f
        mha_time_b = (
            qkv_proj_b + attention_score_b + attention_output_b + output_proj_b + attention_scale_softmax_b
        )
        ffn_time_b = ffn1_b + ffn2_b

        # Calculate transformer times
        transformer_time_f = (
            mha_time_f + ffn_time_f + residual1_f + residual2_f + layernorm1_f + layernorm2_f
        )
        transformer_time_b = (
            mha_time_b + ffn_time_b + residual1_b + residual2_b + layernorm1_b + layernorm2_b
        )

        # Write results to file
        output_file = "Transformer_breakdown_results.txt"
        with open(output_file, "w") as f:
            f.write("\n\n==============================================\n")
            f.write("Performance Results\n")
            f.write("==============================================\n")
            f.write(f"qkv_proj_f: {qkv_proj_f * m}{second}\n")
            f.write(f"attention_score_f: {attention_score_f * m}{second}\n")
            f.write(f"attention_scale_softmax_f: {attention_scale_softmax_f * m}{second}\n")
            f.write(f"attention_output_f: {attention_output_f * m}{second}\n")
            f.write(f"output_proj_f: {output_proj_f * m}{second}\n")
            f.write(f"residual1_f: {residual1_f * m}{second}\n")
            f.write(f"layernorm1_f: {layernorm1_f * m}{second}\n")
            f.write(f"ffn1_f: {ffn1_f * m}{second}\n")
            f.write(f"ffn2_f: {ffn2_f * m}{second}\n")
            f.write(f"residual2_f: {residual2_f * m}{second}\n")
            f.write(f"layernorm2_f: {layernorm2_f * m}{second}\n")
            f.write(f"linear_softmax_f: {linear_softmax_f * m}{second}\n")
            f.write(f"MHA Time: {mha_time_f * m}{second}\n")
            f.write(f"FFN Time: {ffn_time_f * m}{second}\n")
            f.write(f"Transformer Time (1 layer): {transformer_time_f * m}{second}\n")

        # Debugging output
        if debug:
            print(f"embedding_f: {embedding_f * m:.1f}{second}")
            print(f"qkv_proj_f: {qkv_proj_f * m:.1f}{second}")
            print(f"attention_score_f: {attention_score_f * m:.1f}{second}")
            print(f"attention_output_f: {attention_output_f * m:.1f}{second}")
            print(f"output_proj_f: {output_proj_f * m:.1f}{second}")
            print(f"residual1_f: {residual1_f * m:.1f}{second}")
            print(f"layernorm1_f: {layernorm1_f * m:.1f}{second}")
            print(f"ffn1_f: {ffn1_f * m:.1f}{second}")
            print(f"ffn2_f: {ffn2_f * m:.1f}{second}")
            print(f"residual2_f: {residual2_f * m:.1f}{second}")
            print(f"layernorm2_f: {layernorm2_f * m:.1f}{second}")
            print(f"linear_softmax_f: {linear_softmax_f * m:.1f}{second}")
            print(f"MHA Time: {mha_time_f * m:.1f}{second}")
            print(f"FFN Time: {ffn_time_f * m:.1f}{second}")
            print(f"Transformer Time (1 layer): {transformer_time_f * m:.1f}{second}")

        return (
            transformer_time_f, transformer_time_b, embedding_f, embedding_b,
            linear_softmax_f, linear_softmax_b
        )


    def _effective_transformer_batch(self) -> int:
        if self.lp > 1:
            return self.microB
        if self.dp > 1:
            return self.miniB
        return self.batch_size

    def _build_comm_metadata(
        self,
        reduction_sizes: Dict[str, int],
        local_comp: Dict[str, float],
        embedding_size: int,
        softmax_size: int,
        cross_layer_bytes: int,
    ) -> Dict[str, Dict[str, Any]]:
        return {
            'transformer': {
                'size': reduction_sizes['total_size'],
                'type': 'all_reduce',
                'participants': self.dp,
                'interconnect_type': 'dp',
                'local_comp_time': local_comp['total_local']
            },
            'embedding': {
                'size': embedding_size,
                'type': 'all_reduce',
                'participants': self.dp,
                'interconnect_type': 'dp',
                'local_comp_time': 0
            },
            'softmax': {
                'size': softmax_size,
                'type': 'all_reduce',
                'participants': self.dp,
                'interconnect_type': 'dp',
                'local_comp_time': 0
            },
            'cross_layer': {
                'size': cross_layer_bytes,
                'type': 'pipeline',
                'participants': 2,
                'interconnect_type': 'lp',
                'local_comp_time': 0
            }
        }

    def _populate_transformer_comm_metadata(
        self,
        entry: Dict[str, Any],
        metadata: Dict[str, Dict[str, Any]],
        m: int,
        k: int,
        n: int,
    ) -> None:
        """Attach tensor-parallel collectives for a GEMM to metadata and entry."""

        tp_mode = self.t
        kp1 = int(self.kp1) if self.kp1 else 1
        kp2 = int(self.kp2) if self.kp2 else 1
        precision = self.precision

        if not tp_mode or (kp1 <= 1 and kp2 <= 1):
            return

        def add_comm(direction: str, suffix: str, kind: str, size_bytes: float, participants: int, interconnect: str) -> None:
            if participants <= 1:
                return
            bytes_int = int(math.ceil(size_bytes))
            if bytes_int <= 0:
                return
            key = f"{entry['name']}_{direction}_{suffix}"
            # ensure uniqueness when multiple collectives share the same suffix
            unique_key = key
            counter = 1
            while unique_key in metadata:
                counter += 1
                unique_key = f"{key}_{counter}"

            metadata[unique_key] = {
                'size': bytes_int,
                'type': kind,
                'participants': int(participants),
                'interconnect_type': interconnect,
            }
            entry[direction]['comm_keys'].append(unique_key)

        if tp_mode == "CR":
            if kp1 <= 1:
                return
            total_bytes = math.ceil(precision * m * n)
            add_comm('forward', 'reduce_scatter', 'reduce_scatter', total_bytes, kp1, 'kp1')

            # Backward all-gather mirrors forward reduce-scatter.
            total_bytes = math.ceil(precision * m * n)
            size_bytes = math.ceil(total_bytes / kp1)
            add_comm('backward', 'all_gather', 'all_gather', size_bytes, kp1, 'kp1')
            return

        if tp_mode == "RC":
            if kp2 > 1:
                total_bytes = math.ceil(precision * (m // max(1, kp1)) * n)
                size_bytes = math.ceil(total_bytes / kp2)
                add_comm('forward', 'all_gather', 'all_gather', size_bytes, kp2, 'kp2')

            if kp1 > 1:
                total_bytes_row = math.ceil(precision * k * m)
                size_row = math.ceil(total_bytes_row / kp1)

                total_bytes_col = math.ceil(precision * m * (n / max(1, kp2)))
                size_col = math.ceil(total_bytes_col / kp1)

                add_comm(
                    'backward',
                    'wt_all_gather',
                    'all_gather',
                    size_row + size_col,
                    kp1,
                    'kp1',
                )

            if kp2 > 1:
                total_bytes_row = math.ceil(precision * (m / max(1, kp1)) * n)
                size_row = math.ceil(total_bytes_row / kp2)

                total_bytes_col = math.ceil(precision * k * n)
                size_col = math.ceil(total_bytes_col / kp2)

                add_comm(
                    'backward',
                    'act_all_gather',
                    'all_gather',
                    size_row + size_col,
                    kp2,
                    'kp2',
                )

    def _build_interconnect_params(self) -> Dict[str, Tuple[float, float]]:
        return {
            'dp': (self.IBD, self.LLD),
            'lp': (self.IBL, self.LLL),
            'kp1': (self.IBK1, self.LLK1),
            'kp2': (self.IBK2, self.LLK2)
        }

    def _build_pipeline_graph(
        self,
        num_micro_batches: int,
        num_layers: int,
        transformer_time_f: float,
        transformer_time_b: float,
        embedding_f: float,
        embedding_b: float,
        linear_softmax_f: float,
        linear_softmax_b: float,
        comm_metadata: Dict[str, Dict[str, Any]],
    ) -> Tuple[Graph, Any, Dict[str, Tuple[float, float]]]:
        comp_times = {
            "embedding_f": embedding_f,
            "embedding_b": embedding_b,
            "linear_softmax_f": linear_softmax_f,
            "linear_softmax_b": linear_softmax_b,
            "transformer_f": transformer_time_f,
            "transformer_b": transformer_time_b,
            "cross_layer_f": 0.0,
            "cross_layer_b": 0.0,
        }
        misc_metadata = {
            "num_batch": num_micro_batches,
            "num_layer": num_layers,
            "all_reduce": self.all_reduce,
        }

        graph = simulate_LLM.Graph(
            mode="pipeline",
            dp=self.dp,
            lp=self.lp,
            kp1=self.kp1,
            kp2=self.kp2,
            tp_mode=self.t,
            comp_times=comp_times,
            comm_metadata=comm_metadata,
            misc_metadata=misc_metadata,
        )
        root = graph.construct_fwd_bwd_graph()
        interconnect_params = self._build_interconnect_params()
        return graph, root, interconnect_params

    def _tp_degree(self) -> int:
        kp1 = self.kp1 if self.kp1 else 1
        kp2 = self.kp2 if self.kp2 else 1
        return int(kp1 * kp2)
    
    def calcTime_LLM(self):
        """Calculate time for LLM model."""
        # Extract model parameters
        batch_size = self._effective_transformer_batch()
        vocab_size = self.vocab_size
        num_layers = self.num_layers
        hidden_dim = self.hidden_dim
        seq_len = self.seq_len
        num_heads = self.num_heads
        ffn_mult = self.ffn_mult
        ffn_dim = self.hidden_dim * ffn_mult if ffn_mult else self.ffn_dim
        num_micro_batches = self.mb

        # Adjust types and calculate node latencies
        self.readjust_type()
        transformer_time_f, transformer_time_b, embedding_f, embedding_b, linear_softmax_f, linear_softmax_b = self.getNodeLatency(batch_size, vocab_size, hidden_dim, seq_len, num_heads, ffn_dim)

        if self.debug:
            print(
                "dp: {}, lp: {}, kp_hidden_dim1: {}, kp_hidden_dim2: {}, kp_softmax_dim1: {}, kp_softmax_dim2: {}, kp_embedding_dim1: {}, kp_embedding_dim2: {},  kp_hidden_type: {}, kp_softmax_type: {}, kp_embedding_type: {}\n".format(
                    self.dp, self.lp, self.kp_hidden_dim1, self.kp_hidden_dim2,
                    self.kp_softmax_dim1, self.kp_softmax_dim2, self.kp_embedding_dim1,
                    self.kp_embedding_dim2, self.kp_hidden_type, self.kp_softmax_type,
                    self.kp_embedding_type,
                )
            )


            # print(
            #     "embedding_f: {}, embedding_b: {}, qkv_proj_f: {}, qkv_proj_b: {}, attention_score_f: {}, attention_score_b: {}, attention_scale_softmax_f: {}, attention_scale_softmax_b: {}, attention_output_f: {}, attention_output_b: {}, output_proj_f: {}, output_proj_b: {}, residual1_f: {}, residual1_b: {}, layernorm1_f: {}, layernorm1_b: {}, ffn1_f: {}, ffn1_b: {}, ffn2_f: {}, ffn2_b: {}".format(
            #     embedding_f, embedding_b,
                
            #     qkv_proj_f, qkv_proj_b, attention_score_f, attention_score_b,
            #     attention_scale_softmax_f, attention_scale_softmax_b,
            #     attention_output_f, attention_output_b, output_proj_f, output_proj_b,
            #     residual1_f, residual1_b, layernorm1_f, layernorm1_b,
            #     ffn1_f, ffn1_b, ffn2_f, ffn2_b
            # ))
        print("Calculating LLM time...")

        # Create graph and calculate times

        print("simulating parallelism with dp = {}, lp = {}, total data batch = {}, "
            "for each dp node, data batch = {}, for each pipeline stage, data batch = {}".format(self.dp, self.lp, self.batch_size, self.miniB, self.microB))
        print("total number of workers: {}".format(self.num_workers))
        print("number of workers for each data parallelism batch: {}".format(self.num_workers_dp))
        print("number of workers for each pipeline stage: {}".format(self.num_workers_lp))

        reduction_sizes = self.getDataParallelReductionSizes(hidden_dim, ffn_dim)
        local_comp = self.getDataParallelLocalComputation(hidden_dim, ffn_dim)
        embedding_size = math.ceil(self.precision * vocab_size * hidden_dim) + math.ceil(self.precision * seq_len * hidden_dim)
        softmax_size = math.ceil(self.precision * hidden_dim * vocab_size)
        cross_layer_bytes = self.getInterLayerCommLatency_LLM(batch_size, hidden_dim, seq_len)[1]

        comm_metadata = self._build_comm_metadata(
            reduction_sizes=reduction_sizes,
            local_comp=local_comp,
            embedding_size=embedding_size,
            softmax_size=softmax_size,
            cross_layer_bytes=cross_layer_bytes,
        )

        shapes = LLM_util.process_gemm_shapes(
            batch_size,
            seq_len,
            hidden_dim,
            num_heads,
            ffn_dim,
            vocab_size,
            option="multiply_batch_into_m",
        )
        (
            gemm_qkv_proj,
            gemm_attention_score,
            gemm_attention_output,
            gemm_output_proj,
            gemm_ffn1,
            gemm_ffn2,
            _,
        ) = shapes

        gemm_specs = [
            ("gemm_qkv_proj", gemm_qkv_proj),
            ("gemm_attn_score", gemm_attention_score),
            ("gemm_attn_output", gemm_attention_output),
            ("gemm_output_proj", gemm_output_proj),
            ("gemm_ffn1", gemm_ffn1),
            ("gemm_ffn2", gemm_ffn2),
        ]

        transformer_gemm_entries = []
        transformer_comm_metadata: Dict[str, Dict[str, Any]] = {}

        for base_name, (m, k, n) in gemm_specs:
            fwd_time, fwd_red = self._distributed_gemm_forward(m, k, n, base_name + "_fwd")
            bwd_time, bwd_red = self._distributed_gemm_backward(m, k, n, base_name + "_bwd")

            entry = {
                "name": base_name,
                "forward": {
                    "duration": fwd_time,
                    "reduction": fwd_red,
                    "comm_keys": [],
                },
                "backward": {
                    "duration": bwd_time,
                    "reduction": bwd_red,
                    "comm_keys": [],
                },
            }

            self._populate_transformer_comm_metadata(
                entry=entry,
                metadata=transformer_comm_metadata,
                m=m,
                k=k,
                n=n,
            )

            transformer_gemm_entries.append(entry)

        tp_degree = self._tp_degree()

        transformer_comp_times = {
            "transformer": {
                "gemms": transformer_gemm_entries,
                "tp_degree": tp_degree,
            }
        }
        self.transformer_graph = simulate_LLM.Graph(
            mode="transformer",
            dp=self.dp,
            lp=self.lp,
            kp1=self.kp1,
            kp2=self.kp2,
            tp_mode=self.t,
            comp_times=transformer_comp_times,
            comm_metadata=transformer_comm_metadata,
            misc_metadata={},
        )
        self.transformer_graph_root = self.transformer_graph.construct_transformer_graph()

        analytical_time = 0.0
        for entry in transformer_gemm_entries:
            analytical_time += entry["forward"]["duration"] + entry["forward"].get("reduction", 0.0)
            analytical_time += entry["backward"]["duration"] + entry["backward"].get("reduction", 0.0)
        self.transformer_analytical_time = analytical_time

        pipeline_graph_obj, graph_root, interconnect_params = self._build_pipeline_graph(
            num_micro_batches=num_micro_batches,
            num_layers=num_layers,
            transformer_time_f=transformer_time_f,
            transformer_time_b=transformer_time_b,
            embedding_f=embedding_f,
            embedding_b=embedding_b,
            linear_softmax_f=linear_softmax_f,
            linear_softmax_b=linear_softmax_b,
            comm_metadata=comm_metadata,
        )

        self.pipeline_graph = pipeline_graph_obj
        self.pipeline_root = graph_root
        self.pipeline_interconnect = interconnect_params

        dispatcher = LLMExecutionDispatcher(
            time_calc=self,
            pipeline_graph=self.pipeline_graph,
            pipeline_root=self.pipeline_root,
            interconnect_params=self.pipeline_interconnect,
            transformer_graph=self.transformer_graph,
            transformer_root=self.transformer_graph_root,
        )
        mode = self.execution_mode
        try:
            result = dispatcher.run(mode)
        except NotImplementedError as exc:
            raise NotImplementedError(f"{exc}. Selected execution mode '{mode.value}'.") from exc

        time_fw_bw = result.total_time

        pipeline_root = result.graph_root
        self.pipeline_graph = dispatcher.pipeline_graph
        self.pipeline_root = pipeline_root
        self.pipeline_interconnect = dispatcher.interconnect_params

        self.pipeline_graph.save_graph(pipeline_root, "output_graph/", "fw_bw_graph")

        if self.transformer_analytical_time is not None:
            print(f"Analytical transformer time: {self.transformer_analytical_time:.1f}s")

        self.transformer_graph.save_graph(self.transformer_graph_root, "output_graph/", "transformer_graph")

        self.tot_time = time_fw_bw

        output_file = "LLM_time_results.txt"
        
        with open(output_file, "w") as f:
            f.write("\n\n==============================================\n")
            f.write("Performance Results\n")
            f.write("==============================================\n")
            # f.write("Forward Time: {0:.8f} {1}\n".format(time_fw * m, second))
            # f.write("Backward Time: {0:.8f} {1}\n".format(time_bw * m, second))
            f.write("Forward + Backward Time: {0:.8f} {1}\n".format(time_fw_bw * m, second))

            # f.write("Total Time: {0:.8f}\n".format(TC.getTime()))

        return time_fw_bw

    def getTime(self):
        return self.tot_time

    def getReductionTotal(self):
        return getattr(self, "_reduction_total_llm", 0.0)


class LLMExecutionDispatcher:
    def __init__(
        self,
        time_calc: TimeCalculationLLM,
        pipeline_graph: Graph,
        pipeline_root: Any,
        interconnect_params: Dict[str, Tuple[float, float]],
        transformer_graph: Optional[Graph] = None,
        transformer_root: Optional[Any] = None,
    ) -> None:
        self.time_calc = time_calc
        self.pipeline_graph = pipeline_graph
        self.pipeline_root = pipeline_root
        self.interconnect_params = interconnect_params
        self.transformer_graph = transformer_graph
        self.transformer_root = transformer_root

    def run(self, mode: ExecutionMode) -> ExecutionResult:
        if mode == ExecutionMode.ANALYTICAL:
            return self._run_pipeline_with_analytical_comm(ExecutionMode.ANALYTICAL)
        if mode == ExecutionMode.HYBRID:
            # internally, .collective() distinguishes between ANALYTICAL and HYBRID execution modes.
            return self._run_pipeline_with_analytical_comm(ExecutionMode.HYBRID)
        if mode == ExecutionMode.HYBRID_CONGESTION:
            return self._run_hybrid_congestion()
        if mode == ExecutionMode.FULL_ASTRASIM:
            return self._run_full_astrasim()
        raise ValueError(f"Unsupported execution mode: {mode}")

    def _run_pipeline_with_analytical_comm(self, declared_mode: ExecutionMode) -> ExecutionResult:
        timed_root = self.pipeline_graph.convert_comm_sizes_to_times(
            self.pipeline_root,
            self.time_calc.network_model,
            self.interconnect_params,
        )
        # Persist timed root for any downstream consumer
        self.pipeline_root = timed_root
        total_time = self.pipeline_graph.simulate(timed_root)
        return ExecutionResult(total_time=total_time, graph_root=timed_root, mode=declared_mode)

    def _run_hybrid_congestion(self) -> ExecutionResult:
        scale = self._run_transformer_astrasim(ExecutionMode.HYBRID_CONGESTION)
        if scale is not None:
            self._apply_transformer_scale(scale)
        return self._run_pipeline_with_analytical_comm(ExecutionMode.HYBRID_CONGESTION)

    def _run_full_astrasim(self) -> ExecutionResult:
        scale = self._run_transformer_astrasim(ExecutionMode.FULL_ASTRASIM)
        if scale is not None:
            self._apply_transformer_scale(scale)

        dp_count = getattr(self.time_calc, "dp", 1) or 1
        if not self.pipeline_root:
            raise RuntimeError("Pipeline graph root is not available for AstraSim execution")

        # per_rank_sec, max_sec = run_astrasim_graph(
        #     graph_root=self.pipeline_root,
        #     dp_count=int(dp_count),
        #     hw_obj=self.time_calc.hw_config,
        #     tag="pipeline_full",
        # )

        per_rank_sec, max_sec = run_astra_simulation_only_onepath(self.pipeline_root, self.time_calc, "./astra_pipeline_output")
        self.time_calc.pipeline_astrasim_per_rank = per_rank_sec
        self.time_calc.pipeline_astrasim_time = max_sec
        if max_sec <= 0:
            raise RuntimeError("AstraSim pipeline execution returned non-positive duration")
        return ExecutionResult(total_time=max_sec, graph_root=self.pipeline_root, mode=ExecutionMode.FULL_ASTRASIM)

    def _run_transformer_astrasim(self, mode: ExecutionMode) -> Optional[float]:
        if not self.transformer_root:
            return None

        # per_rank_sec, max_sec = run_astrasim_graph(
        #     graph_root=self.transformer_root,
        #     dp_count=1,
        #     hw_obj=self.time_calc.hw_config,
        #     tag=f"transformer_{mode.value}",
        # )

        per_rank_sec, max_sec = run_astra_simulation_only_onepath(self.transformer_root, self.time_calc, "./astra_transformer_output", dp_override=1)
        
        self.time_calc.transformer_astrasim_per_rank = per_rank_sec
        self.time_calc.transformer_astrasim_time = max_sec
        if max_sec <= 0:
            raise RuntimeError("AstraSim transformer execution returned non-positive duration")

        analytical_total = getattr(self.time_calc, "transformer_analytical_time", None)
        if not analytical_total or analytical_total <= 0:
            raise RuntimeError("Analytical transformer time is unavailable for scaling")

        scale = max_sec / analytical_total
        self.time_calc.transformer_time_scale = scale
        return scale

    def _apply_transformer_scale(self, scale: float) -> None:
        if scale <= 0:
            raise ValueError("Transformer scaling factor must be positive")

        comp_times = getattr(self.pipeline_graph, "comp_times", None)
        if isinstance(comp_times, dict):
            if "transformer_f" in comp_times:
                comp_times["transformer_f"] *= scale
            if "transformer_b" in comp_times:
                comp_times["transformer_b"] *= scale

        visited: Set[int] = set()
        roots: List[Any]
        if isinstance(self.pipeline_root, (list, tuple)):
            roots = list(self.pipeline_root)
        else:
            roots = [self.pipeline_root]

        for root in roots:
            self._scale_transformer_nodes(root, visited, scale)

    def _scale_transformer_nodes(self, node: Any, visited: Set[int], scale: float) -> None:
        if node is None:
            return
        node_id = id(node)
        if node_id in visited:
            return
        visited.add(node_id)

        if isinstance(node, simulate_LLM.Node):
            if node.name in {"transformer", "transformer_b"}:
                node.duration *= scale

        for child in getattr(node, "children", []):
            self._scale_transformer_nodes(child, visited, scale)
