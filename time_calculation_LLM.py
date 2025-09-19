#!/tools/lm-venv/py3.6-tf-1.3.0-svail/bin/python

# import click
import math
import os
import pickle
import sys
import config
import shutil
import itertools
# import numpy as np
import simulate_LLM
from parallelism import Parallelism
from topology import Topology
from simulate_LLM import Graph
from astra_comparison import run_astra_simulation_only, run_astra_simulation_only_onepath
import LLM_util
from hw_component import Core, MemoryHierarchy, Network
from model import Model_LSTM, Model_GEMM, Model_LLM 
from tile import TiledGEMM, formatBytes

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

class TimeCalculationLLM(TimeCalculation):
    def __init__(self, hw_config, model_config, mode):
# Mode parameter
        
        super().__init__(hw_config, model_config, mode)
        self._reduction_total_llm = 0.0
        self.all_reduce = all_reduce # when the reduce happens in data parallelism options: "the end"  "every layer"
    def getGEMMTime_RC(self, m, k, n, name): #kp2
        # Get the GEMM time for the RC (Row-Column) data layout
        GEMM_time = self.getGEMMTime(m // self.kp_hidden_dim2, k, n // self.kp_hidden_dim1, name)
        reduction_time = 0 #todo All-Reduce across GPUs.
        return GEMM_time[0] + reduction_time
    def getGEMMTime_CR(self, m, k, n, dim1, dim2, name): #kp1
        # Get the GEMM time for the CR data layout
        GEMM_time = self.getGEMMTime(m, k // self.kp_hidden_dim1, n, name)
        reduction_time = 0 #todo All-Gather of input activations before GEMM.
        return GEMM_time[0] + reduction_time



    def get_node_f(self, gemm, name):

        """Get node Time on Forward Path
            For GEMM operations
        """
        m = gemm[0]
        k = gemm[1]
        n = gemm[2]
        
        if self.t == None:
            
            GEMM_time = self.getGEMMTime(m, k, n, name)[0]
            print(f"Calculating GEMM for {name} with shape ({m}, {k}, {n})")
        elif self.t == "RC":
            GEMM_time = self.getGEMMTime_RC(m, k, n, name)
            print(f"Calculating distributed GEMM for {name} with shape ({m}, {k}, {n})")
        elif self.t == "CR":
            GEMM_time = self.getGEMMTime_CR(m, k, n, name)
            print(f"Calculating distributed GEMM for {name} with shape ({m}, {k}, {n})")
        else:
            raise ValueError("Invalid tp strategy")

        if self.debug:
            print("{} GEMM_time: {:,}\n".format(name, GEMM_time[0]))
        if self.validating_GEMM:
            return GEMM_time
        else:
            return GEMM_time + self.O #gemm time plus kernel launch overhead

    def get_node_b(self, gemm, name):
        """Get node Time on Backward Path
            For GEMM operations
        """
        m = gemm[0]
        k = gemm[1]
        n = gemm[2]
        if self.t == None:
            grad_act_time, _, _, _ = self.getGEMMTime( # Get activation gradient time
                m, n, k, "{}_act".format(name)
            )
            grad_wt_time, _, _, _ = self.getGEMMTime( # Get weight gradient time
                k, m, n, "{}_wt".format(name)
            )
        elif self.t == "RC":
            grad_act_time = self.getGEMMTime_RC(
                m, n, k, "{}_act".format(name)
            )
            grad_wt_time = self.getGEMMTime_RC(
                k, m, n, "{}_wt".format(name)
            )
        elif self.t == "CR":
            grad_act_time = self.getGEMMTime_CR(
                m, n, k, "{}_act".format(name)
            )
            grad_wt_time = self.getGEMMTime_CR(
                k, m, n, "{}_wt".format(name)
            )

        GEMM_time = grad_act_time + grad_wt_time


        return GEMM_time + self.O #gemm time plus kernel launch overhead

    def getEmbedding_f(self):
        """
        Calculates the total time required for embedding operations, including computation and data transfer.
        """
        # Get embedding time
        embedding_mem = 2 * (self.seq_len * self.miniB * self.hidden_dim * self.precision)
        embedding_time = self.roofline(0, embedding_mem, name="embedding_f") + self.O
        embedding_transfer_time = 2 * self.seq_len * self.miniB * self.hidden_dim * self.precision / self.H2Dbw
        if self.debug:
            print("Embedding_mem: {:,}".format(int(embedding_mem / 1e9)))
        return embedding_time + embedding_transfer_time

    def getLinearSoftmax_f(self, gemm):
        """
        Calculates the total computation time for a linear softmax operation using GEMM (General Matrix Multiply) and pointwise operations.
        """
        m = gemm[0]
        k = gemm[1]
        n = gemm[2]
        
        if self.t == None:
            GEMM_time = self.getGEMMTime(
                m, k, n, "linear_softmax_f"
            )
            print(f"Calculating GEMM for linear_softmax_f with shape ({m}, {k}, {n})")
        elif self.t == "RC":
            GEMM_time = self.getGEMMTime_RC(
                m, k, n, "linear_softmax_f"
            )
            print(f"Calculating distributed GEMM for linear_softmax_f with shape ({m}, {k}, {n})")
        elif self.t == "CR":
            GEMM_time = self.getGEMMTime_CR(
                m, k, n, "linear_softmax_f"
            )
            print(f"Calculating distributed GEMM for linear_softmax_f with shape ({m}, {k}, {n})")
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

        return GEMM_time + point_time
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
        if self.t == None:
            grad_wt_time, _, _, _ = self.getGEMMTime(
                k, m, n, "linear_softmax_b_wt"
            )
            grad_act_time, _, _, _ = self.getGEMMTime(
                m, n, k, "linear_softmax_b_act"
            )
        elif self.t == "RC":
            grad_act_time = self.getGEMMTime_RC(
                m, n, k, "linear_softmax_b_act"
            )
            grad_wt_time = self.getGEMMTime_RC(
                k, m, n, "linear_softmax_b_wt"
            )
        elif self.t == "CR":
            grad_act_time = self.getGEMMTime_CR(
                m, n, k, "linear_softmax_b_act"
            )
            grad_wt_time = self.getGEMMTime_CR(
                k, m, n, "linear_softmax_b_wt"
            )

        GEMM_time = grad_wt_time + grad_act_time
        point_flop = m * n * 5
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

        embedding_mem = 2 * self.seq_len * self.miniB * self.hidden_dim * self.precision
        embedding_mem_time = (
            self.roofline(0, embedding_mem, name="embedding_b") + self.O
        )

        if self.debug:
            print("(gr) Embedding_mem: {:,}".format(int(embedding_mem / 1e9)))
        # return data_transfer_time + embedding_mem_time
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
        
        
    def calcTime_LLM(self):
        """Calculate time for LLM model."""
        # Extract model parameters
        if self.lp > 1: #simulate transformer for each pipeline stage with micro-batchsize
            batch_size = self.microB
        elif self.dp > 1: #simulate transformer for each data parallel worker with mini-batchsize
            batch_size = self.miniB
        else:
            batch_size = self.batch_size
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

        # Calculate reduction times
        # get the reduction time and weight update time for transformer, reduction occurs between data parallel (dp) GPUs of different nodes, all-reduce
        # (total number of GPUs for all-reduce equals dp)
        # R_transformer, G_transformer = self.getDPOverhead(
        #     d=hidden_dim, ffn_dim=ffn_dim, n_layers=num_layers
        # )



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
        # Calculate communication sizes and local computation times separately
        reduction_sizes = self.getDataParallelReductionSizes(hidden_dim, ffn_dim)
        local_comp = self.getDataParallelLocalComputation(hidden_dim, ffn_dim)

        # Calculate embedding and softmax sizes
        embedding_size = math.ceil(self.precision * vocab_size * hidden_dim) + math.ceil(self.precision * seq_len * hidden_dim)
        softmax_size = math.ceil(self.precision * hidden_dim * vocab_size)

        comm_metadata = {
            'transformer': {
                'size': reduction_sizes['total_size'],
                'type': 'all_reduce',
                'participants': self.dp,
                'interconnect_type': 'dp',  # Data parallel
                'local_comp_time': local_comp['total_local']
            },
            'embedding': {
                'size': embedding_size,
                'type': 'all_reduce',
                'participants': self.dp,
                'interconnect_type': 'dp',  # Data parallel
                'local_comp_time': 0  # No local computation for embedding reduction
            },
            'softmax': {
                'size': softmax_size,
                'type': 'all_reduce',
                'participants': self.dp,
                'interconnect_type': 'dp',  # Data parallel
                'local_comp_time': 0  # No local computation for softmax reduction
            },
            'cross_layer': {
                'size': self.getInterLayerCommLatency_LLM(batch_size, hidden_dim, seq_len)[1],
                'type': 'pipeline',
                'participants': 2,
                'interconnect_type': 'lp',  # Pipeline parallel
                'local_comp_time': 0  # No local computation for cross-layer latency
            }
        }

        # g = Graph(
        g = simulate_LLM.Graph(
            num_batch=num_micro_batches, num_layer=num_layers, lp=self.lp, dp=self.dp, all_reduce=self.all_reduce,
            T_embedding_f=embedding_f, T_linear_softmax_f=linear_softmax_f,
            T_embedding_b=embedding_b, T_linear_softmax_b=linear_softmax_b,
            # Tb=0, Tf=0, T_reduction_transformer=0, T_grad_transformer=0,
            # T_reduction_embedding=0, T_reduction_linear_softmax=0,
            T_transformer_f=transformer_time_f, T_transformer_b=transformer_time_b,
            comm_metadata=comm_metadata,
        )

        # Phase 2: Convert communication sizes to times using network model
        interconnect_params = {
            'dp': (self.IBD, self.LLD),
            'lp': (self.IBL, self.LLL),
            'kp1': (self.IBK1, self.LLK1),
            'kp2': (self.IBK2, self.LLK2)
        }
        # fw_roots = g.construct_fwd_graph()
        # bw_roots = g.construct_bwd_graph()
        fw_bw_root = g.construct_fwd_bwd_graph()
        # fw_root = g.convert_comm_sizes_to_times(fw_roots[0], self.network_model, interconnect_params)
        # bw_root = g.convert_comm_sizes_to_times(bw_roots[0], self.network_model, interconnect_params)
        fw_bw_root = g.convert_comm_sizes_to_times(fw_bw_root, self.network_model, interconnect_params)
        # Avoid graph rendering when running under astra_test or when disabled explicitly
        # time_fw = g.simulate(fw_root)
        # time_bw = g.simulate(bw_root)
        time_fw_bw = g.simulate(fw_bw_root)

        # print(f"time_fw: {time_fw}\ntime_bw: {time_bw}")
        print(f'time_fw_bw: {time_fw_bw}')

    

        if os.environ.get("ASTRA_TEST") or os.environ.get("DISABLE_LLM_GRAPH"):
            # simulate wiht astrasim too
            # write the graph to a file as a pkl file
            # with open("fw_bw_graph.pkl", "wb") as f:
            #     pickle.dump(fw_bw_root, f)  
            run_astra_simulation_only_onepath(fw_bw_root, self, "./astra_comparison_output")\
            
            g.save_graph(fw_bw_root, "output_graph/","fw_bw_graph")
        else:
            g.save_graph(fw_bw_root, "output_graph/","fw_bw_graph")

        # self.tot_time = time_fw + time_bw
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
