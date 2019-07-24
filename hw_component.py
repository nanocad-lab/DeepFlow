import math

import util
from topology import Topology

class Base:  
  def __init__(self, exp_config):
      self.exp_config               = exp_config
      self.precision                = exp_config.sw_config.precision
      self.area_budget              = exp_config.area_breakdown.area_budget
      self.TDP                      = exp_config.power_breakdown.TDP
      self.throughput               = -1
  def calcThroughput(self):
      print("Each class should have its own calcThroughput")
  def getThroughput(self):
      assert(self.throughput != -1)
      return self.throughput

class Memory(Base):
  def __init__(self, exp_config):
      super().__init__(exp_config)
      self.size                     = -1

  def getSize(self):
      assert(self.size != -1)
      return self.size


class Core(Base):
  def __init__(self, exp_config):
      super().__init__(exp_config)
      self.tot_power                    = exp_config.power_breakdown.core * self.TDP
      self.tot_area                     = exp_config.area_breakdown.core * self.area_budget
      
      self.nominal_voltage              = exp_config.tech_config.core.nominal_voltage
      self.nominal_freq                 = exp_config.tech_config.core.nominal_frequency
      self.nominal_area_per_mcu         = exp_config.tech_config.core.nominal_area_per_mcu
      self.nominal_flop_rate_per_mcu    = exp_config.tech_config.core.nominal_flop_rate_per_mcu
      self.nominal_energy_per_mcu       = exp_config.tech_config.core.nominal_energy_per_mcu
      
      self.operating_voltage            = exp_config.tech_config.core.operating_voltage
      #Assumption: frequency scales linearly with voltage
      self.operating_freq               = (self.nominal_freq * self.operating_voltage / 
                                           self.nominal_voltage)
      self.operating_area_per_mcu       = exp_config.tech_config.core.operating_area_per_mcu
      self.num_mcu_per_bundle           = exp_config.tech_config.core.num_mcu_per_bundle
      self.num_mcu                      = self.tot_area // self.operating_area_per_mcu
      self.num_bundle                   = self.num_mcu // self.num_mcu_per_bundle
      self.area_scaling                 = self.operating_area_per_mcu / self.nominal_area_per_mcu
      #Assumption: performance scales linearly with area
      self.operating_flop_rate_per_mcu  = self.nominal_flop_rate_per_mcu * self.area_scaling
      
      self.calcEnergyPerUnit()
      self.calcThroughput()

  def calcEnergyPerUnit(self):
      self.nominal_energy_per_flop      = (self.nominal_energy_per_mcu / 
                                           self.nominal_flop_rate_per_mcu)
      #Assumption: energy per flop does not scale with mcu area
      #TODO: @Saptadeep: does this assumption make sense?
      self.energy_per_flop              = (self.nominal_energy_per_flop * 
                                           (self.operating_voltage / self.nominal_voltage) * 
                                           (self.operating_freq / self.nominal_freq))
  def calcThroughput(self):
      self.nominal_throughput           = min(self.tot_power / self.energy_per_flop, 
                                              self.operating_flop_rate_per_mcu * self.operating_freq * self.num_mcu)
      self.throughput                   = self.nominal_throughput * util.core

class DRAM(Memory):
  def __init__(self, exp_config):
      super().__init__(exp_config)
      self.tot_power                  = exp_config.power_breakdown.DRAM * self.TDP
      self.tot_area                   = exp_config.area_breakdown.DRAM * self.area_budget
      self.dynamic_energy_per_byte    = exp_config.tech_config.DRAM.dynamic_energy_per_bit * 8
      self.static_power_per_byte      = exp_config.tech_config.DRAM.static_power_per_bit * 8
      self.area_per_byte              = exp_config.tech_config.DRAM.area_per_bit * 8
      self.stack_bw                   = exp_config.tech_config.DRAM.stack_bw
      self.stack_capacity             = exp_config.tech_config.DRAM.stack_capacity

      
      self.calcArea()
      self.calcSize()
      self.calcActiveEnergy()
      self.calcThroughput()

  def calcActiveEnergy(self):
        #TODO: @Saptaddeep: Can you verify if this is correct?
      self.dynamic_power             = self.tot_power - self.static_power_per_byte * self.size

  def calcThroughput(self):
      self.dynamic_throughput         = self.dynamic_power / self.dynamic_energy_per_byte
      self.throughput                 = self.dynamic_throughput * util.DRAM
  
  def calcArea(self):
        #I/O, inter-bank, intra-bank overhead
        #TODO: @Saptadeep, do you know how to capture the DRAM circutry overhead
      self.overhead_area              = 0
      self.cell_area                  = self.tot_area - self.overhead_area
  
  def calcSize(self):
      self.nominal_throughput         = self.tot_power / self.dynamic_energy_per_byte
      self.size                       = min((self.nominal_throughput / self.stack_bw) * self.stack_capacity,
                                               self.cell_area / self.area_per_byte)


class L2(Memory):
  def __init__(self, exp_config):
      super().__init__(exp_config)
      self.tot_power                  = exp_config.power_breakdown.L2 * self.TDP
      self.tot_area                   = exp_config.area_breakdown.L2 * self.area_budget
      self.dynamic_energy_per_byte    = exp_config.tech_config.L2.dynamic_energy_per_bit * 8
      self.static_power_per_byte      = exp_config.tech_config.L2.static_power_per_bit * 8
      self.area_per_byte              = exp_config.tech_config.L2.area_per_bit * 8
      self.bank_bw                    = exp_config.tech_config.L2.bank_bw
      self.bank_capacity              = exp_config.tech_config.L2.bank_capacity
      self.controller_area_per_link   = exp_config.tech_config.L2.controller_area_per_link
      
      self.nominal_throughput         = self.tot_power / self.dynamic_energy_per_byte
      self.num_banks                  = self.nominal_throughput // self.bank_bw
      
      self.calcArea()
      self.calcSize()
      self.calcActiveEnergy()
      self.calcThroughput()
      self.calcTileDim()

  def calcArea(self):
      #I/O, inter-bank, intra-bank overhead
      #TODO: @Saptadeep, do you know how to capture the DRAM circutry overhead
      #in over_head area calculation, should it be num_SMS or num_cores?
      core                            = Core(self.exp_config)
      self.overhead_area              = self.num_banks * core.num_bundle * self.controller_area_per_link
      self.cell_area                  = self.tot_area - self.overhead_area
  
  def calcActiveEnergy(self):
        #TODO: @Saptaddeep: Can you verify if this is correct?
      self.dynamic_power             = self.tot_power - self.static_power_per_byte * self.size

  def calcThroughput(self):
      self.dynamic_throughput         = self.dynamic_power / self.dynamic_energy_per_byte
      self.throughput                 = self.dynamic_throughput * util.L2

  def calcSize(self):
      self.size                       = min(self.num_banks * self.bank_capacity,
                                            self.cell_area / self.area_per_byte)
  def calcTileDim(self):
      self.tile_dim = 0
      if (self.size > 0):
          self.tile_dim = math.ceil(math.pow(2, math.floor(math.log(math.sqrt((self.size / self.precision) / 3), 2))))

class SharedMem(Memory):
  def __init__(self, exp_config):
      super().__init__(exp_config)
      self.tot_power                  = exp_config.power_breakdown.shared_mem * self.TDP
      self.tot_area                   = exp_config.area_breakdown.shared_mem * self.area_budget
      self.dynamic_energy_per_byte    = exp_config.tech_config.shared_mem.dynamic_energy_per_bit * 8
      self.static_power_per_byte      = exp_config.tech_config.shared_mem.static_power_per_bit * 8
      self.area_per_byte              = exp_config.tech_config.shared_mem.area_per_bit * 8
      self.bank_bw                    = exp_config.tech_config.shared_mem.bank_bw
      self.bank_capacity              = exp_config.tech_config.shared_mem.bank_capacity
      self.controller_area_per_link   = exp_config.tech_config.shared_mem.controller_area_per_link
      
      self.nominal_throughput         = self.tot_power / self.dynamic_energy_per_byte
      self.num_banks                  = self.nominal_throughput // self.bank_bw
      
      self.calcArea()
      self.calcSize()
      self.calcActiveEnergy()
      self.calcThroughput()
      self.calcTileDim()


  def calcArea(self):
      #I/O, inter-bank, intra-bank overhead
      #TODO: @Saptadeep, do you know how to capture the shared mem circutry overhead
      core                            = Core(self.exp_config)
      self.overhead_area              = self.num_banks * core.num_mcu_per_bundle * self.controller_area_per_link
      self.cell_area                  = self.tot_area - self.overhead_area
  
  def calcActiveEnergy(self):
      #TODO: @Saptaddeep: Can you verify if this is correct?
      self.dynamic_power             = self.tot_power - self.static_power_per_byte * self.size

  def calcThroughput(self):
      self.dynamic_throughput         = self.dynamic_power / self.dynamic_energy_per_byte
      self.throughput                 = self.nominal_throughput * util.shared_mem

  def calcSize(self):
      self.size                       = min(self.num_banks * self.bank_capacity,
                                              self.cell_area / self.area_per_byte)
  def calcTileDim(self):
      self.tile_dim = 0
      if (self.size > 0):
          self.tile_dim = math.ceil(math.pow(2, math.floor(math.log(math.sqrt((self.size / self.precision) / 3), 2))))

class Network(Base):
  def __init__(self, exp_config):
      super().__init__(exp_config)
      self.intra_network              = SubNetwork(exp_config, 
                                                   exp_config.tech_config.network.intra_node,
                                                   exp_config.power_breakdown.network.intra_node,
                                                   exp_config.area_breakdown.network.intra_node)
      self.inter_network              = SubNetwork(exp_config,
                                                   exp_config.tech_config.network.inter_node,
                                                   exp_config.power_breakdown.network.inter_node,
                                                   exp_config.area_breakdown.network.inter_node)


      intra_data                      = exp_config.tech_config.network.intra_node.parallelMap.data
      inter_data                      = exp_config.tech_config.network.inter_node.parallelMap.data
      data_dim                        = exp_config.sch_config.dp
      self.data_throughput             = self.calcThroughput(intra_data, inter_data, data_dim)

      intra_layer                     = exp_config.tech_config.network.intra_node.parallelMap.layer
      inter_layer                     = exp_config.tech_config.network.inter_node.parallelMap.layer
      lp_dim                          = exp_config.sch_config.lp
      self.layer_throughput            = self.calcThroughput(intra_layer, inter_layer, lp_dim)

      intra_kernel                    = exp_config.tech_config.network.intra_node.parallelMap.kernel
      inter_kernel                    = exp_config.tech_config.network.inter_node.parallelMap.kernel
      h1                              = exp_config.sch_config.kp_hidden_dim1
      s1                              = exp_config.sch_config.kp_softmax_dim1
      e1                              = exp_config.sch_config.kp_embedding_dim1
      p1                              = exp_config.sch_config.kp_projection_dim1
      h2                              = exp_config.sch_config.kp_hidden_dim2
      s2                              = exp_config.sch_config.kp_softmax_dim2
      e2                              = exp_config.sch_config.kp_embedding_dim2
      p2                              = exp_config.sch_config.kp_projection_dim2
      kp_dim                          = max(h1, h2, s1, s2, p1, p2, e1, e2)
      self.kernel_throughput           = self.calcThroughput(intra_kernel, inter_kernel, kp_dim)

  def calcThroughput(self,intra_map, inter_map, sch_parallelism):
      if intra_map and sch_parallelism > 1:
          throughput        = self.intra_network.throughput
      elif inter_map and sch_parallelism > 1:
          throughput        = self.inter_network.throughput
      else:
          throughput        = 0
      return throughput 

class SubNetwork(Base):
  def __init__(self, exp_config, net_config, power_breakdown, area_breakdown):
      super().__init__(exp_config)
      self.tot_power                  = power_breakdown * self.TDP
      self.tot_area                   = area_breakdown  * self.area_budget
      self.topology                   = Topology(net_config, exp_config)
      self.latency                    = net_config.latency
      self.nominal_freq               = net_config.nominal_freq
      self.nominal_voltage            = net_config.nominal_voltage
      self.nominal_energy_per_link    = net_config.nominal_energy_per_link
      #self.nominal_area_per_link      = net_config.nominal_area_per_link
      self.operating_freq             = net_config.operating_freq
      self.operating_voltage          = net_config.operating_voltage
      self.num_links_per_mm           = net_config.num_links_per_mm
      self.parallelMap                = net_config.parallelMap

      self.energy_per_bit             = self.calcEnergyPerBit()

      self.throughput                 = self.calcThroughput()

  def calcEnergyPerBit(self):
      self.operating_energy_per_link  = (self.nominal_energy_per_link * 
                                         (self.operating_freq / self.nominal_freq) *
                                         (self.operating_voltage / self.nominal_voltage))
      energy_per_bit                  = self.operating_energy_per_link
      return energy_per_bit

  def calcThroughput(self):
      if self.topology.num_links == 0:
        throughput                = 0
      else:
        self.power_per_p2p2_connection = self.tot_power / self.topology.num_links
        node_width                     = math.sqrt(self.area_budget)
        throughput                = min(self.power_per_p2p2_connection / self.energy_per_bit,
                                             (node_width *
                                              self.num_links_per_mm *
                                              self.frequency)) / 8  #in Bytes/sec
      
      return throughput
