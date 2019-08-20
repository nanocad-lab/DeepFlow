import math

import util
from topology import Topology

class System:
  def __init__(self, exp_config):
      self.num_nodes_per_wafer      = exp_config.system_hierarchy.num_nodes_per_wafer

class Base:  
  def __init__(self, exp_config):
      self.exp_config               = exp_config
      self.precision                = exp_config.sw_config.precision
      self.proc_chip_area_budget    = exp_config.area_breakdown.proc_chip_area_budget
      self.TDP                      = exp_config.power_breakdown.TDP
      self.topology                 = Topology(exp_config)
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
      self.tile_dim                 = -1
      self.latency                  = -1

  def getSize(self):
      assert(self.size != -1)
      return self.size

  def getTileDim(self):
      assert(self.tile_dim != -1)
      return self.tile_dim

  def getLatency(self):
      assert(self.latency != -1)
      return self.latency

class Core(Base):
  def __init__(self, exp_config):
      super().__init__(exp_config)
      self.tot_power                    = exp_config.power_breakdown.core * self.TDP
      self.tot_area                     = exp_config.area_breakdown.core * self.proc_chip_area_budget
      
      self.nominal_voltage              = exp_config.tech_config.core.nominal_voltage
      self.nominal_freq                 = exp_config.tech_config.core.nominal_frequency
      self.nominal_area_per_mcu         = exp_config.tech_config.core.nominal_area_per_mcu
      #TODO: Define it as a function of precision
      self.nominal_flop_rate_per_mcu    = exp_config.tech_config.core.nominal_flop_rate_per_mcu
      self.nominal_power_per_mcu       = exp_config.tech_config.core.nominal_power_per_mcu
      
      #self.operating_voltage            = exp_config.tech_config.core.operating_voltage

      self.threshold_voltage            = exp_config.tech_config.core.threshold_voltage
      #Assumption: frequency scales linearly with voltage
      #SP: Changed the frequency scaling model to the non-linear one. F ~ (Vdd-Vth)^2/Vdd
      #self.operating_freq               = (self.nominal_freq * (self.operating_voltage - self.threshold_voltage)**2 * self.nominal_voltage / (self.operating_voltage * (self.nominal_voltage- self.threshold_voltage)**2))
      self.operating_area_per_mcu       = exp_config.tech_config.core.operating_area_per_mcu
      self.num_mcu_per_bundle           = exp_config.tech_config.core.num_mcu_per_bundle
      self.num_mcu                      = self.tot_area // self.operating_area_per_mcu
      self.num_bundle                   = self.num_mcu // self.num_mcu_per_bundle
      self.area_scaling                 = self.operating_area_per_mcu / self.nominal_area_per_mcu
      #Assumption: performance scales linearly with area
      self.operating_flop_rate_per_mcu  = self.nominal_flop_rate_per_mcu * self.area_scaling
      
      self.calcOperatingVoltageFrequency()
      #self.calcEnergyPerUnit()
      self.calcThroughput()

  def calcOperatingVoltageFrequency(self):
      self.tot_nominal_power_cores      = self.nominal_power_per_mcu * self.num_mcu
      self.operating_voltage            = (math.sqrt(self.tot_power/self.tot_nominal_power_cores))*self.nominal_voltage

      if self.operating_voltage < (self.threshold_voltage + 0.2):
          self.frequency_scaling_factor = ((self.operating_voltage)/(self.threshold_voltage + 0.2))**2
          self.operating_voltage        = self.threshold_voltage + 0.2
      else:
          self.frequency_scaling_factor = 1 

      self.operating_freq               = self.frequency_scaling_factor * (self.nominal_freq * (self.operating_voltage - self.threshold_voltage)**2 * 
                                          self.nominal_voltage / (self.operating_voltage * (self.nominal_voltage- self.threshold_voltage)**2))


  def calcEnergyPerUnit(self):
      self.nominal_energy_per_flop      = (self.nominal_energy_per_mcu / 
                                           self.nominal_flop_rate_per_mcu)
      #Assumption: energy per flop does not scale with mcu area
      #TODO: @Saptadeep: does this assumption make sense? SP: Yes, but I am not sure if we want to change area per MCU as it usually doesn't change with voltage or frequency. 

      #Changed energy per flop scaling model to be proportional to square of voltage
      self.energy_per_flop              = (self.nominal_energy_per_flop * 
                                           ((self.operating_voltage / self.nominal_voltage)**2))
  def calcThroughput(self):
      #self.nominal_throughput           = min(self.tot_power / self.energy_per_flop, 
      #                                        self.operating_flop_rate_per_mcu * self.operating_freq * self.num_mcu)
     self.operating_throughput         = self.operating_flop_rate_per_mcu * self.operating_freq * self.num_mcu
     self.throughput                   = self.operating_throughput * util.core

class DRAM(Memory):
  def __init__(self, exp_config):
      super().__init__(exp_config)
      self.tot_power                  = exp_config.power_breakdown.DRAM * self.TDP
      self.tot_area                   = exp_config.area_breakdown.node_area_budget - self.proc_chip_area_budget
      self.tot_mem_ctrl_area          = self.proc_chip_area_budget * exp_config.area_breakdown.DRAM
      self.mem_ctrl_area              = exp_config.tech_config.DRAM.mem_ctrl_area
      self.dynamic_energy_per_byte    = exp_config.tech_config.DRAM.dynamic_energy_per_bit * 8
      self.static_power_per_byte      = exp_config.tech_config.DRAM.static_power_per_bit * 8
      self.area_per_byte              = exp_config.tech_config.DRAM.area_per_bit * 8
      self.stack_bw                   = exp_config.tech_config.DRAM.stack_bw
      self.stack_capacity             = exp_config.tech_config.DRAM.stack_capacity
      self.area_per_stack             = exp_config.tech_config.DRAM.area_per_stack
      self.latency                    = exp_config.tech_config.DRAM.latency

      self.num_channels               = min(self.tot_area//self.area_per_stack, self.tot_mem_ctrl_area // self.mem_ctrl_area)
      
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
      assert(self.overhead_area < self.tot_area)
  
  def calcSize(self):
      #self.nominal_throughput         = self.tot_power / self.dynamic_energy_per_byte
      #self.size                       = min((self.nominal_throughput / self.stack_bw) * self.stack_capacity,
      #                                         self.cell_area / self.area_per_byte)
      self.size                       = self.num_channels * self.stack_capacity

  def calcTileDim(self):
      self.tile_dim = 0
      if (self.size > 0):
          self.tile_dim = math.ceil(math.pow(2, math.floor(math.log(math.sqrt((self.size / self.precision) / 3), 2))))


class L2(Memory):
  def __init__(self, exp_config):
      super().__init__(exp_config)
      self.tot_power                  = exp_config.power_breakdown.L2 * self.TDP
      self.tot_area                   = exp_config.area_breakdown.L2 * self.proc_chip_area_budget
      self.dynamic_energy_per_byte    = exp_config.tech_config.L2.dynamic_energy_per_bit * 8
      self.static_power_per_byte      = exp_config.tech_config.L2.static_power_per_bit * 8
      self.area_per_byte              = exp_config.tech_config.L2.area_per_bit * 8
      self.bank_bw                    = exp_config.tech_config.L2.bank_bw
      self.bank_capacity              = exp_config.tech_config.L2.bank_capacity
      self.controller_area_per_link   = exp_config.tech_config.L2.controller_area_per_link
      self.latency                    = exp_config.tech_config.L2.latency
      
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
      self.cell_area                  = (self.tot_area - self.overhead_area)*0.8
      assert(self.overhead_area < self.tot_area)
  
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
      #TODO: Does it make sense to keep the tile dim power of 2 or can we allow non-power of 2 tile-dim
      #check http://web.cse.ohio-state.edu/~pouchet.2/doc/hipc-article.11.pdf
      self.tile_dim = 0
      if (self.size > 0):
          self.tile_dim = math.ceil(math.pow(2, math.floor(math.log(math.sqrt((self.size / self.precision) / 3), 2))))

class SharedMem(Memory):
  def __init__(self, exp_config):
      super().__init__(exp_config)
      self.tot_power                  = exp_config.power_breakdown.shared_mem * self.TDP
      self.tot_area                   = exp_config.area_breakdown.shared_mem * self.proc_chip_area_budget
      self.dynamic_energy_per_byte    = exp_config.tech_config.shared_mem.dynamic_energy_per_bit * 8
      self.static_power_per_byte      = exp_config.tech_config.shared_mem.static_power_per_bit * 8
      self.area_per_byte              = exp_config.tech_config.shared_mem.area_per_bit * 8
      self.bank_bw                    = exp_config.tech_config.shared_mem.bank_bw
      self.bank_capacity              = exp_config.tech_config.shared_mem.bank_capacity
      self.controller_area_per_link   = exp_config.tech_config.shared_mem.controller_area_per_link
      self.latency                    = exp_config.tech_config.shared_mem.latency
      
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
      self.cell_area                  = (self.tot_area - self.overhead_area)*0.8
      assert(self.overhead_area < self.tot_area)
  
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
      core                            = Core(self.exp_config)
      self.size_per_bundle            = self.size / core.num_bundle 
      if (self.size > 0):
          self.tile_dim = math.ceil(math.pow(2, math.floor(math.log(math.sqrt((self.size_per_bundle / self.precision) / 3), 2))))

class RegMem(Memory):
  def __init__(self, exp_config):
      super().__init__(exp_config)
      self.tot_power                  = exp_config.power_breakdown.reg_mem * self.TDP
      self.tot_area                   = exp_config.area_breakdown.reg_mem * self.proc_chip_area_budget
      self.dynamic_energy_per_byte    = exp_config.tech_config.reg_mem.dynamic_energy_per_bit * 8
      self.static_power_per_byte      = exp_config.tech_config.reg_mem.static_power_per_bit * 8
      self.area_per_byte              = exp_config.tech_config.reg_mem.area_per_bit * 8
      self.bank_bw                    = exp_config.tech_config.reg_mem.bank_bw
      self.bank_capacity              = exp_config.tech_config.reg_mem.bank_capacity
      self.controller_area_per_link   = exp_config.tech_config.reg_mem.controller_area_per_link
      self.latency                    = exp_config.tech_config.reg_mem.latency
      
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
      #self.overhead_area              = self.num_banks * core.num_mcu_per_bundle * self.controller_area_per_link
      #SP: Usually the overhead gets a bit amortized as the bank size grows, but the sense amps etc. also scale with bitline length, so not a straight-forward model. I am assuming about 25% overhead which is reasonable based on ISSCC 2018 SRAM papers from Intel and Samsung 
      self.overhead_area              = self.tot_area * 0.25
      self.cell_area                  = self.tot_area - self.overhead_area
      assert(self.overhead_area < self.tot_area)
  
  def calcActiveEnergy(self):
      #TODO: @Saptaddeep: Can you verify if this is correct?
      self.dynamic_power             = self.tot_power - self.static_power_per_byte * self.size

  def calcThroughput(self):
      self.dynamic_throughput         = self.dynamic_power / self.dynamic_energy_per_byte
      self.throughput                 = self.nominal_throughput * util.reg_mem

  def calcSize(self):
      self.size                       = min(self.num_banks * self.bank_capacity,
                                            self.cell_area / self.area_per_byte)
  def calcTileDim(self):
      self.tile_dim = 0
      core                            = Core(self.exp_config)
      self.size_per_bundle            = self.size / core.num_bundle 
      if (self.size > 0):
          self.tile_dim = math.ceil(math.pow(2, math.floor(math.log(math.sqrt((self.size_per_bundle / self.precision) / 3), 2))))


class Network(Base):
  def __init__(self, exp_config):
      super().__init__(exp_config)
      self.intra_network              = SubNetwork(exp_config, 
                                                   exp_config.tech_config.network.intra_node,
                                                   exp_config.power_breakdown.network.intra_node,
                                                   exp_config.area_breakdown.network.intra_node,
                                                   'intra')
      self.inter_network              = SubNetwork(exp_config,
                                                   exp_config.tech_config.network.inter_node,
                                                   exp_config.power_breakdown.network.inter_node,
                                                   exp_config.area_breakdown.network.inter_node,
                                                   'inter')


      #intra_data                      = exp_config.tech_config.network.intra_node.parallelMap.data
      #inter_data                      = exp_config.tech_config.network.inter_node.parallelMap.data
      #data_dim                        = exp_config.sch_config.dp
      #self.data_throughput, self.data_latency = self.calcThroughput(intra_data, inter_data, data_dim)
      self.data_throughput, self.data_latency = self.calcThroughput('data')

      #intra_layer                     = exp_config.tech_config.network.intra_node.parallelMap.layer
      #inter_layer                     = exp_config.tech_config.network.inter_node.parallelMap.layer
      #lp_dim                          = exp_config.sch_config.lp
      #self.layer_throughput, self.layer_latency = self.calcThroughput(intra_layer, inter_layer, lp_dim)
      self.layer_throughput, self.layer_latency = self.calcThroughput('layer')

      #intra_kernel                    = exp_config.tech_config.network.intra_node.parallelMap.kernel
      #inter_kernel                    = exp_config.tech_config.network.inter_node.parallelMap.kernel
      h1                              = exp_config.sch_config.kp_hidden_dim1
      s1                              = exp_config.sch_config.kp_softmax_dim1
      e1                              = exp_config.sch_config.kp_embedding_dim1
      p1                              = exp_config.sch_config.kp_projection_dim1
      h2                              = exp_config.sch_config.kp_hidden_dim2
      s2                              = exp_config.sch_config.kp_softmax_dim2
      e2                              = exp_config.sch_config.kp_embedding_dim2
      p2                              = exp_config.sch_config.kp_projection_dim2
      #kp_dim                          = max(h1 * h2, s1 *s2, p1 * p2, e1 * e2)
      #self.kernel_throughput, self.kernel_latency = self.calcThroughput(intra_kernel, inter_kernel, kp_dim)
      self.kernel_throughput, self.kernel_latency = self.calcThroughput('kernel')

  def calcThroughput(self, sch_parallelism):
      if 'data' in sch_parallelism:
          throughput, latency = self.topology.getDataThroughput(self.intra_network.throughput,
                                                                self.inter_network.throughput,
                                                                self.intra_network.latency,
                                                                self.inter_network.latency)
      elif 'kernel' in sch_parallelism:
          throughput, latency = self.topology.getKernelThroughput(self.intra_network.throughput,
                                                                  self.inter_network.throughput,
                                                                  self.intra_network.latency,
                                                                  self.inter_network.latency)
      elif 'layer' in sch_parallelism:
          throughput, latency = self.topology.getLayerThroughput(self.intra_network.throughput,
                                                                 self.inter_network.throughput,
                                                                 self.intra_network.latency,
                                                                 self.inter_network.latency)
      else:
          throughput        = 0
          latency           = 0

      return throughput, latency 

class SubNetwork(Base):
  def __init__(self, exp_config, net_config, power_breakdown, area_breakdown, netLevel):
      super().__init__(exp_config)
      self.tot_power                  = power_breakdown * self.TDP
      self.tot_area                   = area_breakdown  * self.proc_chip_area_budget
      self.latency                    = net_config.latency
      self.nominal_freq               = net_config.nominal_freq
      self.nominal_voltage            = net_config.nominal_voltage
      self.nominal_energy_per_link    = net_config.nominal_energy_per_link
      self.nominal_area_per_link      = net_config.nominal_area_per_link
      self.threshold_voltage          = net_config.threshold_voltage
      #self.operating_freq             = net_config.operating_freq
      #self.operating_voltage          = net_config.operating_voltage
      self.num_links_per_mm           = net_config.num_links_per_mm
      self.inter                      = True if netLevel == 'inter' else False
      self.intra                      = True if netLevel == 'intra' else False
      #Calculate num_links dedicated to inter or intra network
      inter_fraction, intra_fraction  = self.topology.get_fractions()
      perimeter_fraction              = inter_fraction if self.inter else intra_fraction
      node_width                      = math.sqrt(self.proc_chip_area_budget)
      perimeter                       = node_width * 4
      self.num_links                  = int(min(self.tot_area / 
                                                self.nominal_area_per_link, 
                                                perimeter_fraction * 
                                                perimeter * 
                                                self.num_links_per_mm))

      self.calcOperatingVoltageFrequency()

      self.energy_per_bit             = self.calcEnergyPerBit()

      self.throughput                 = self.calcThroughput()

  def calcOperatingVoltageFrequency(self):
      if self.inter and self.topology.interNodeDegree == 0:
          self.operating_freq             = 0
          self.operating_voltage          = 0
      elif self.intra and self.topology.intraNodeDegree == 0:
          self.operating_freq             = 0
          self.operating_voltage          = 0
      else:
          self.tot_nominal_power_links      = self.nominal_energy_per_link * self.num_links * self.nominal_freq
          self.operating_voltage            = (math.sqrt(self.tot_power/self.tot_nominal_power_links))*self.nominal_voltage

          if self.operating_voltage < (self.threshold_voltage + 0.4):
              self.frequency_scaling_factor = ((self.operating_voltage)/(self.threshold_voltage + 0.4))**2
              self.operating_voltage        = self.threshold_voltage + 0.4
          else:
              self.frequency_scaling_factor = 1 

          self.operating_freq               = self.frequency_scaling_factor * (self.nominal_freq * (self.operating_voltage - self.threshold_voltage)**2 * 
                                              self.nominal_voltage / (self.operating_voltage * (self.nominal_voltage- self.threshold_voltage)**2))

  def calcEnergyPerBit(self):
      self.operating_energy_per_link    = (self.nominal_energy_per_link * 
                                         (self.operating_voltage / self.nominal_voltage)**2)
      energy_per_bit                    = self.operating_energy_per_link
      return energy_per_bit

  def calcThroughput(self):
      throughput                      = (self.num_links * self.operating_freq)/8 
      return throughput
