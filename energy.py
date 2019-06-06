import click
import math

import os
import config

import perf as pf

from parallelism import Parallelism
from topology import Topology
import util

class EnergyCalculation:
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
        par                     = Parallelism(exp_config)
        par.findParallelStrategy()
        self.autoPar            = par.autoPar
        self.lp                 = par.lp
        self.hlp                = par.hlp
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

        #Power Breakdown Parameters
        self.TDP                = exp_config.power_breakdown.TDP
        self.DRAMPower          = exp_config.power_breakdown.DRAM * self.TDP
        self.HBM_stack_bw       = exp_config.tech_config.HBM_stack_bw
        self.HBM_stack_capacity = exp_config.tech_config.HBM_stack_capacity
        self.mem_bw             = self.DRAMPower / self.dram_energy_per_byte * util.mem_scale
        self.mem_size           = (self.mem_bw / util.mem_scale / self.HBM_stack_bw) * self.HBM_stack_capacity

       
        #Define miniBatch size
        self.miniB = math.ceil(self.B / self.dp)

        #Calculate architecture configurations based on power breakdown
        #and technology parameters
        self.corePower = exp_config.power_breakdown.core * self.TDP
        self.L2Power   = exp_config.power_breakdown.L2 * self.TDP
        self.shared_mem_power = exp_config.power_breakdown.shared_mem * self.TDP
        self.IBPower  = exp_config.power_breakdown.IB * self.TDP
        self.th     = self.corePower / self.energy_per_flop * util.th_scale
        
        self.L2_bank_bw  = exp_config.tech_config.L2_bank_bw
        self.L2_bank_capacity = exp_config.tech_config.L2_bank_capacity
        self.L2_bw    = (self.L2Power / self.L2_energy_per_byte)
        self.L2_size  = (self.L2_bw / self.L2_bank_bw) * self.L2_bank_capacity 
      
        self.shared_mem_bank_bw = exp_config.tech_config.shared_mem_bank_bw
        self.shared_mem_bank_capacity = exp_config.tech_config.shared_mem_bank_capacity
        self.shared_mem_bw = (self.shared_mem_power / self.shared_mem_energy_per_byte)
        self.shared_mem_size = (self.shared_mem_bw / self.shared_mem_bank_bw) * self.shared_mem_bank_capacity
       


        # Network Parameters
        # IBK: interconnection bandiwdth across kernel parall dimension
        # IBD: interconnection bandiwdth across data parall dimension
        # IBM: interconnection bandiwdth across model parall dimension
        # We are assuming same network bandwidth along all dimensons
        # TODO: Parameterize this so we can explore different bandwidth along different dimensions
        topology = Topology(exp_config) 
        self.IB = (0 if topology.grid_dim == 0 else self.IBPower / (topology.num_links * self.internode_energy_per_byte)) 
        self.IBK = self.IB
        self.IBM = self.IB
        self.IBD = self.IB

        self.attached = True
        self.debug = False
        self.tot_time = 0

        self.L2_tile_dim = 0
        self.sm_tile_dim = 0
        self.setTileDim()
    
        self.tot_energy = 0


    def setTileDim(self):
        self.L2_tile_dim = math.ceil(math.pow(2, math.floor(math.log(math.sqrt((self.L2_size / self.precision) / 3), 2))))
        self.sm_tile_dim = math.ceil(math.pow(2, math.floor(math.log(math.sqrt((self.shared_mem_size/ self.precision) / 3), 2))))
        #print("L2Tile_dim: {}, SMTile_dim: {}".format(self.L2_tile_dim, self.sm_tile_dim))
    
    def getTileDim(self):
        assert(self.L2_tile_dim != 0)
        assert(self.sm_tile_dim != 0)
        return self.L2_tile_dim, self.sm_tile_dim

    def roofline(self, flop, gmem, l2mem=0):
        inflection_point = self.th / self.mem_bw
        comp_int = flop / gmem
        
        L2_tile_dim, sm_tile_dim = self.getTileDim()
        single_block_comp_time = (2 * L2_tile_dim * L2_tile_dim * L2_tile_dim) / self.th
        single_block_mem_time  = (3 * L2_tile_dim * L2_tile_dim * self.precision) / self.mem_bw

        time  = 0
        if comp_int < inflection_point: #mem-bound
            time = (gmem / self.mem_bw) + (l2mem / self.L2_bw) + single_block_comp_time
        else: #compute-bound
            time = (flop / self.th) + single_block_mem_time
        
        #if self.debug:
        #    print("inflection_point: {}".format(inflection_point))
        #    print("comp_int: {}".format(comp_int))
        #    print("flop: {}".format(flop))
        #    print("mem: {}".format(mem))
        #    print("time: {}".format(time))
        
        return time
      
    def getGEMMCounts(self, A, B, C, name):
        GEMM_flop, GEMM_gmem, GEMM_l2mem = self.GEMM(A, B, C)
        GEMM_flop_t, GEMM_gmem_t, GEMM_l2mem_t = self.GEMM(C, B, A)
        
        GEMM_time = self.roofline(GEMM_flop, GEMM_gmem, GEMM_l2mem) + self.O
        GEMM_time_t = self.roofline(GEMM_flop_t, GEMM_gmem_t, GEMM_l2mem_t) + self.O
        
        transpose = False
        time = GEMM_time
        flop = GEMM_flop
        gmem  = GEMM_gmem
        l2mem  = GEMM_l2mem

        if (GEMM_time > GEMM_time_t):
            transpose = True
            time = GEMM_time_t
            flop = GEMM_flop_t
            gmem  = GEMM_gmem_t
            l2mem  = GEMM_l2mem_t

        if self.debug:
            if transpose:
                print("{} GEMM_flop_t: {:,}, GEMM_gmem_t: {:,}, GEMM_l2mem_t: {:,}".format(name, int(GEMM_flop_t),int(GEMM_gmem_t), int(GEMM_l2mem_t)))
                print("{} GEMM_time_t: {:,}".format(name, GEMM_time_t))
            else:
                print("{} GEMM_flop: {:,}, GEMM_gmem: {:,}, GEMM_l2mem: {:,}".format(name, int(GEMM_flop),int(GEMM_gmem), int(GEMM_l2mem)))
                print("{} GEMM_time: {:,}".format(name, GEMM_time))

        return flop, gmem, l2mem

    def GEMM(self, A, B, C):
        GEMM_flop = 2 * A * B * C
        X1, X2 = self.getTileDim()
       
        #Here we are assuming tiles are in square form
        #Global memory accesses going through L2
        reload_AB = 1
        reload_BC = 1
        reload_AC = 1

        if  B <= X1:
            reload_AB = 1
            reload_BC = math.ceil(A / X1)
            reload_AC = 1
        else:
            reload_AB = math.ceil(C / X1)
            reload_BC = math.ceil(A / X1)
            reload_AC = 1
        
        GEMM_gmem = (A * B * reload_AB + B * C * reload_BC + A * C * reload_AC) * self.precision

        #L2memory accesses going through Shared memory
        reload_AB = 1
        reload_BC = 1
        reload_AC = 1
        
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


        
        return GEMM_flop, GEMM_gmem, GEMM_l2mem

      
    def totEnergy(self, flops, dram_mem, l2_mem=0, bytes_transfered=0):
        flopEnergy = flops * self.energy_per_flop  
        dramEnergy = dram_mem * self.dram_energy_per_byte
        l2Energy = l2_mem * self.L2_energy_per_byte
        netEnergy = bytes_transfered * self.internode_energy_per_byte
        
        totEnergy = flopEnergy + dramEnergy + l2Energy + netEnergy
        return totEnergy

    def getCf(self):
        """Get LSTM Cell Energy on Forward Path"""
        GEMM_flop, GEMM_gmem, GEMM_l2mem = self.getGEMMCounts(self.miniB, 2 * self.D, self.G * self.D, "Cf")
        GEMM_energy = self.totEnergy(GEMM_flop, GEMM_gmem, GEMM_l2mem)

        point_flop = (self.miniB * self.D * self.P)
        point_mem  = (self.precision * self.miniB * self.D * 
                     (3 * self.A + 2 * self.NL ))

        point_energy = self.totEnergy(point_flop, point_mem)

        if self.debug:
            print("Cf point_flop: {:,}, point_mem: {:,}\n".format(int(point_flop), int(point_mem)))

        return GEMM_energy + point_energy

    def getCb(self):
        """Get LSTM Cell Energy on Backward Path"""
        grad_act_flop, grad_act_gmem, grad_act_l2mem = self.getGEMMCounts(self. miniB, self.G * self.D, 2 * self.D, "Cb_act") 
        grad_wt_flop, grad_wt_gmem, grad_wt_l2mem   = self.getGEMMCounts(2 * self.D, self.B, self.G * self.D, "Cb_wt")
        
        GEMM_flop = grad_act_flop + grad_wt_flop
        GEMM_gmem  = grad_act_gmem  + grad_wt_gmem
        GEMM_l2mem  = grad_act_l2mem  + grad_wt_l2mem

        GEMM_energy = self.totEnergy(GEMM_flop, GEMM_gmem, GEMM_l2mem)

        point_flop = (((self.miniB * self.D * self.NL) + #pointwise operations backprop, only nonlinear operations perform computations
                     (self.D * self.D) * self.G * 2)) #acccumulate weight gradient 
        point_mem  = ((self.precision * self.miniB * self.D * 
                     (3 * self.A + 2 * self.NL)) +
                     (3 * (self.D * self.D) * self.G * 2 * self.precision))
        point_energy = self.totEnergy(point_flop, point_mem)
   
        if self.debug:
            print("Cb point_flop: {:,}, point_mem: {:,}\n".format(int(point_flop), int(point_mem)))

        return GEMM_energy + point_energy


    def getR(self, x, y):
        if (self.dp == 1):
            return 0

        data_transfer  = 2 * self.precision * x * y * (self.G * 2) #self.dp / self.dp 
        data_prep_comp = x * y * (self.G * 2)
        data_prep_mem  = 3 * self.precision * x * y * (self.G * 2)
        R_energy = self.totEnergy(data_prep_comp, data_prep_mem, 0, data_transfer) * (self.dp - 1)
        
        if self.debug:
            print("(gr) allReduce_flop: {:,}, allReduce_mem: {:,}".format(int(data_prep_comp), int(data_prep_mem)))

        return R_energy

    def getSoftmax_f(self):
        GEMM_flop, GEMM_gmem, GEMM_l2mem = self.getGEMMCounts(self.miniB * self.S, self.D, self.V, "softmax_f")
        GEMM_energy = self.totEnergy(GEMM_flop, GEMM_gmem, GEMM_l2mem)

        #Final matrix after GEMM has (B X S, V) dimensionality
        #We get exponential on each of the elements in a row
        #and then reduce all elements in the row to one
        #Therefore for each row we do V sigmoids and V-1 additions and V divisions
        #For each row, we read V sigmoids and write one
        point_flop = (self.miniB * self.S * (3 * self.V - 1))
        point_mem  = self.precision * self.miniB * self.S * (5 * self.V + 2) 
        point_energy = self.totEnergy(point_flop, point_mem)
   
        if self.debug:
            print("Softmax point_flop: {:,}, point_mem: {:,}".format(int(point_flop), int(point_mem)))
            print("Softmax GEMM_energy: {:,}, point_energy: {:,}\n".format(GEMM_energy, point_energy))

        return GEMM_energy + point_energy

    def getSoftmax_b(self):
        grad_wt_flop, grad_wt_gmem, grad_wt_l2mem   = self.getGEMMCounts(self.V, self.miniB * self.S, self.D, "softmax_b_wt")
        grad_act_flop, grad_act_gmem, grad_act_l2mem = self.getGEMMCounts(self.miniB * self.S, self.V, self.D, "softmax_b_act")

        GEMM_flop = grad_act_flop + grad_wt_flop
        GEMM_gmem  = grad_act_gmem  + grad_wt_gmem
        GEMM_l2mem  = grad_act_l2mem  + grad_wt_l2mem

        GEMM_energy = self.totEnergy(GEMM_flop, GEMM_gmem, GEMM_l2mem)

        #Going from a (B X S, 1) to (B X S, V)
        point_flop = (self.miniB * self.S * (4 * self.V - 1))
        point_mem  = self.precision * self.miniB * self.S * (5 * self.V + 2) 
        point_energy = self.totEnergy(point_flop, point_mem) 
    
        if self.debug:
            print("(gr) Softmax point_flop: {:,}, point_mem: {:,}\n".format(int(point_flop), int(point_mem)))
        
        return GEMM_energy + point_energy

   
    def getEmbedding_f(self):
        embedding_mem = (self.miniB * self.D * self.S * self.precision)
        embedding_energy = self.totEnergy(0, embedding_mem)
        if self.debug:
            print("Embedding_mem: {:,}".format(int(embedding_mem)))
        return embedding_energy


    def getEmbedding_b(self):
        data_transfer  = self.precision * self.miniB * self.S * self.D * (self.dp - 1)
        
        embedding_flop = self.miniB * self.S * self.D
        embedding_mem = 3 * self.miniB * self.S * self.D * self.precision
        embedding_energy = self.totEnergy(embedding_flop, embedding_mem, 
                                          0, data_transfer) * self.dp

        return embedding_energy



    def getTotMemReq(self):
        hidden_act_mem = (self.miniB * self.D * self.L * self.S * (self.G * 2)) 
        hidden_wt_mem  = self.D * self.D * self.L * (self.G * 2)
        hidden_mem = hidden_act_mem + hidden_wt_mem

        softmax_act_mem = self.miniB * self.S * self.V  
        softmax_wt_mem = self.D * self.V
        softmax_mem = softmax_act_mem + softmax_wt_mem

        embedding_mem = self.miniB * self.S * self.D

        totMem = (hidden_mem + softmax_mem + embedding_mem) * self.precision

        print(totMem)
        return totMem


    def getInterModelCommEnergy(self):
        w = self.precision * self.miniB * self.S * self.D
        energy = w * self.internode_energy_per_byte
        return energy


    def dprint(self, string):
        if self.debug:
            print(string)
    
    def calcEnergy(self):
        W = self.getInterModelCommEnergy() 
        Cf = self.getCf()
        Cb = self.getCb()
        RHidden  = self.getR(self.D, self.D) # Across all the data shards
        RSoftmax = self.getR(self.V, self.D) # Across all the data shards
        REmbedding = self.getEmbedding_b()   # Across all the data shards
        L = self.L
        S = self.S
        lp = self.lp

        softmax_f = self.getSoftmax_f()
        softmax_b = self.getSoftmax_b()
        embedding_f = self.getEmbedding_f() 
        embedding_b = self.getEmbedding_b() 

        if self.debug:
          print("lp: {}, Cf: {}, Cb:; {}, RHidden: {}, RSoftmax: {}, REmbedding: {}, softmax_f: {}, softmax_b:{}, embedding_f: {}, embedding_b: {}\n"
             .format(self.lp, Cf, Cb, RHidden, RSoftmax, REmbedding, softmax_f, softmax_b, embedding_f, embedding_b))

        #Energy per data shard exclusing the all reduce energy
        energy_per_data_shard = (embedding_f + Cf * L * S + softmax_f + 
                                softmax_b  + Cb * L * S + W * (lp - 1))
        energy_all_reduce = RHidden * L + RSoftmax + REmbedding

        #RHidden happens every layer
        #Rsoftmax and REmbedding happen only once
        tot_energy = energy_per_data_shard * self.dp + energy_all_reduce

        return tot_energy

    def getEnergy(self):
        return self.tot_energy

@click.command()
@click.option("--exp_config", help="path to experiment config", required=True)
@click.option("--debug", help="debug", default=False)
def main(exp_config, debug):
    exp_path = os.path.expandvars(os.path.expanduser(exp_config))
    exp_config = config.parse_config(exp_path)

    EC = EnergyCalculation(exp_config)
    PC = pf.TimeCalculation(exp_config)

    EC.debug = debug
    tot_energy = EC.calcEnergy()
    tot_time = PC.calcTime()
    tot_power = tot_energy / tot_time
    
    print("Time: {:.2f} Seconds".format(tot_time))
    print("Energy: {0:.2f} Jouls".format(tot_energy))
    print("Power: {0:.2f} Watt.".format(tot_power))

   
if __name__ == "__main__":
    main()
