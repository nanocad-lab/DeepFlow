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
from LLM_util import process_gemm_shapes, linear_gemm
from simulate_LLM import visualize_graph
from time_calculation import TimeCalculation
algByte = False  # algorithmic ops false
proj = False  # consider projection layer, turn off for end-2-end validation, as baeline model does not have projection layer
validating_v100 = True
showing_ms = True
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
        


    def get_node_f(self, gemm, name):
        
        """Get node Time on Forward Path"""
        m = gemm[0]
        k = gemm[1]
        n = gemm[2]
        print(f"Calculating GEMM for {name} with shape ({m}, {k}, {n})")
        
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
    def getLinearSoftmax_f(self, gemm):
        m = gemm[0]
        k = gemm[1]
        n = gemm[2]
        print(f"Calculating GEMM for linear_softmax_f with shape ({m}, {k}, {n})")

        GEMM_time, _, _, _ = self.getGEMMTime(
            m, k, n, "linear_softmax_f"
        )
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
    def getLinearSoftmax_b(self, gemm):
        m = gemm[0]
        k = gemm[1]
        n = gemm[2]

        grad_wt_time, _, _, _ = self.getGEMMTime(
            k, m, n, "linear_softmax_b_wt"
        )
        grad_act_time, _, _, _ = self.getGEMMTime(
            m, n, k, "linear_softmax_b_act"
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
    def getDataParallelReduction_LLM(self, d, ffn_dim):
        # k = 2 * self.D
        # n = 4 * self.D
        # dim1 = self.kp_hidden_dim1
        # dim2 = self.kp_hidden_dim2
        w_data = 4*d*d + 2*ffn_dim*d #total parameters need to be reduced
        reduction_time = 0
        apply_grad_time = 0

        reduction_time += self.getR(Dim0=d, Dim1=3*d, p=self.dp, ib=self.IBD, ll=self.LLD, partial=False, allReduce=True, name="qkv_proj reduction")
        apply_grad_time += self.applyGrad(Dim0=d, Dim1=3*d, name="qkv_proj reduction")
        
        reduction_time += self.getR(Dim0=d, Dim1=d, p=self.dp, ib=self.IBD, ll=self.LLD, partial=False, allReduce=True, name="output_proj reduction")
        apply_grad_time += self.applyGrad(Dim0=d, Dim1=d, name="output_proj reduction")

        reduction_time += 2*self.getR(Dim0=ffn_dim, Dim1=d, p=self.dp, ib=self.IBD, ll=self.LLD, partial=False, allReduce=True, name="ffn reduction")
        apply_grad_time += 2*self.applyGrad(Dim0=ffn_dim, Dim1=d, name="ffn reduction")

        if self.debug:
            print(f"reduction_time: {reduction_time}")
            print(f"apply_grad_time: {apply_grad_time}")

        return reduction_time + apply_grad_time
    def calcTime_LLM(self):
        """Calculate time for LLM model."""
        batch_size = self.miniB
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
        gemm_linear = linear_gemm(batch_size, seq_len, hidden_dim, vocab_size)
        # if self.kp_hidden_type == -1:
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
        linear_softmax_f, linear_softmax_b = self.getLinearSoftmax_f(gemm=gemm_linear), self.getLinearSoftmax_b(gemm=gemm_linear)
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


        R_transformer = self.getDataParallelReduction_LLM(
            d = hidden_dim,
            ffn_dim = ffn_dim

        )
        R_embedding = self.getR(Dim0=vocab_size, Dim1=hidden_dim, p=self.dp, ib=self.IBD, ll=self.LLD, partial=False, allReduce=True, name="embedding reduction")
        R_embedding += self.getR(Dim0=seq_len, Dim1=hidden_dim, p=self.dp, ib=self.IBD, ll=self.LLD, partial=False, allReduce=True, name="positional encoding reduction")

        R_linear_softmax = self.getR(Dim0=hidden_dim, Dim1=vocab_size, p=self.dp, ib=self.IBD, ll=self.LLD, partial=False, allReduce=True, name="softmax reduction")
        print(f"Linear Softmax Reduction Time: {R_linear_softmax * m:.1f}{second}")
        print(f"Embedding Reduction Time: {R_embedding * m:.1f}{second}")
        print(f"Data Parallel Reduction Time: {R_transformer * m:.1f}{second}")
        if self.lp == 1:
            Tf = 0

        Tb = Tf


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
            T_reduction_transformer=R_transformer,
            T_reduction_embedding=R_embedding,
            T_reduction_linear_softmax=R_linear_softmax,

    
    
        )

        time_fw, time_bw = g.save_graph()
        self.tot_time = time_fw + time_bw
        
        
        
        mha_time = qkv_proj_f + attention_score_f + attention_output_f + output_proj_f
        ffn_time = ffn1_f + ffn2_f
        output_file = "LLM_time_results.txt"
        
        with open(output_file, "w") as f:
            f.write("\n\n==============================================\n")
            f.write("Performance Results\n")
            f.write("==============================================\n")
            f.write(f"qkv_proj_f: {qkv_proj_f* m}{second}\n")
            f.write(f"attention_score_f: {attention_score_f* m}{second}\n")
            f.write(f"attention_output_f: {attention_output_f* m}{second}\n")
            f.write(f"output_proj_f: {output_proj_f* m}{second}\n")
            f.write(f"residual1_f: {residual1_f* m}{second}\n")
            f.write(f"layernorm1_f: {layernorm1_f* m}{second}\n")
            f.write(f"ffn1_f: {ffn1_f* m}{second}\n")
            f.write(f"ffn2_f: {ffn2_f* m}{second}\n")
            f.write(f"residual2_f: {residual2_f* m}{second}\n")
            f.write(f"layernorm2_f: {layernorm2_f* m}{second}\n")
            f.write(f"linear_softmax_f: {linear_softmax_f* m}{second}\n")
            f.write(f"MHA Time: {mha_time* m}{second}\n")
            f.write(f"FFN Time: {ffn_time* m}{second}\n")
            f.write("Forward Time: {0:.8f} {1}\n".format(time_fw * m, second))
            f.write("Backward Time: {0:.8f} {1}\n".format(time_bw * m, second))

            # f.write("Total Time: {0:.8f}\n".format(TC.getTime()))
        
        return time_fw, time_bw#, tot_param

    def getTime(self):
        return self.tot_time
