import math
import numpy as np
import util
from topology_hack import Topology

kilo=1024.0
giga=1024.0*1024.0*1024.0

class System:
  def __init__(self, exp_config):
      self.num_nodes_per_wafer      = exp_config.system_hierarchy.num_nodes_per_wafer

class Base:  
  def __init__(self, exp_config):
      self.exp_config               = exp_config
      self.precision                = exp_config.sw_config.precision
      self.proc_chip_area_budget    = exp_config.area_breakdown.proc_chip_area_budget
      self.TDP                      = exp_config.power_breakdown.TDP
      self.throughput               = -1
      node_width                    = math.sqrt(self.proc_chip_area_budget)
      self.core_perimeter           = node_width * 4

  def calcThroughput(self):
      print("Each class should have its own calcThroughput")
  
  def getThroughput(self):
      assert(self.throughput != -1)
      return self.throughput

  def solve_poly(self, p0, p1, p2, p3):
    #solve p0.x^3 + p1.x^2 + p2.x + p3 = 0
    roots = np.roots([p0, p1, p2, p3]);
    real_roots = roots.real[abs(roots.imag)<1e-10] # where I chose 1-e10 as a threshold
    return real_roots[0]

class Memory(Base):
  def __init__(self, exp_config, level):
      super().__init__(exp_config)
      self.size                     = -1
      self.tile_dim                 = -1
      self.latency                  = -1
      self.core                     = Core(self.exp_config)
      self.level                    = level

  def getSize(self):
      assert(self.size != -1)
      return self.size

  def getTileDim(self):
      assert(self.tile_dim != -1)
      return self.tile_dim

  def getLatency(self):
      assert(self.latency != -1)
      return self.latency

  def getTileDims(self):
      return self.getPower2TileDims()
      #return self.getArbitraryTileDims(self):
  
  def getPower2TileDims(self):
      np.random.seed(1)
      tile_dim_candidates = set()
      num_candidates = 20
      M = self.size_per_bundle / self.precision
      max_power = int(math.floor(math.log2(M)))
      
      self.calcTileDim()
      square_tile = self.getTileDim()
      
      tile_dim_candidates.add((square_tile, square_tile, square_tile))
      tile_dim_candidates.add((square_tile//2, square_tile, square_tile*2))
      while len(tile_dim_candidates) < num_candidates:
          z = -1
          while(z < 0):
            s = [pow(2, i) for i  in np.random.randint(0, max_power, 2)]
            #store goes through cache at level 0 and 1 (register and shared memory)
            assert(self.level >= 0 and self.level <= 3)
            if self.level <= 1:
              z = math.floor((M - s[0] * s[1]) / (s[0] + s[1]))
            else:
            #store bypasses cache, directly goes to memory 
              z = math.floor((M - s[0] * s[1]) / s[1])
            
            if z <= 0:
              continue
            
            z = int(math.pow(2, math.floor(math.log2(z))))
            tile_dim = (s[0], s[1], z)
            tile_dim_candidates.add(tile_dim)

      #print(tile_dim_candidates)
      return list(tile_dim_candidates)


  def getArbitraryTileDims(self):
      np.random.seed(0)
      tile_dim_candidates = []
      self.calcTileDim()
      square_tile = self.getTileDim()
      mu, sigma = square_tile, square_tile
      M = self.size_per_bundle / self.precision
      tile_dim_candidates.append((square_tile, square_tile, square_tile))
      for i in range(0, 0):
          z = -1
          while(z < 0):
            s = [int(abs(i)) for i in np.random.normal(mu, sigma, 2)]      
            z = int(math.floor((M - s[0] * s[1]) / (s[0] + s[1])))
          tile_dim = (s[0], s[1], z)
          tile_dim_candidates.append(tile_dim)

      print(tile_dim_candidates)
      return tile_dim_candidates

  def calcTileDim(self):
      self.tile_dim = 0

      if (self.scope == 'global'):
        divisor = 1
      elif (self.scope == 'mcu-bundle'):
        divisor                       = self.core.num_bundle
      elif (self.scope == 'mcu'):
        divisor                       = self.core.num_mcu
      else: 
        NotImplemented()

      self.size_per_bundle            = 0 if (divisor == 0) else self.size / divisor
      
      if (self.size > 0):
          self.tile_dim = math.ceil(math.pow(2, math.floor(math.log(math.sqrt((self.size_per_bundle / self.precision) / 2), 2))))
          #self.tile_dim = math.floor(math.sqrt((self.size_per_bundle / self.precision) / 3))
 

class Core(Base):
  def __init__(self, exp_config):
      super().__init__(exp_config)
      self.tot_power                    = exp_config.power_breakdown.core * self.TDP
      self.tot_area                     = exp_config.area_breakdown.core * self.proc_chip_area_budget
      
      self.FMA_width                    = exp_config.tech_config.core.FMA_width
      self.dataflow                    = exp_config.tech_config.core.dataflow
      self.nominal_voltage              = exp_config.tech_config.core.nominal_voltage
      self.nominal_freq                 = exp_config.tech_config.core.nominal_frequency
      self.nominal_area_per_mcu         = exp_config.tech_config.core.nominal_area_per_mcu
      #TODO: Define it as a function of precision
      self.nominal_flop_rate_per_mcu    = exp_config.tech_config.core.nominal_flop_rate_per_mcu
      self.nominal_power_per_mcu        = exp_config.tech_config.core.nominal_power_per_mcu
      self.util                         = exp_config.tech_config.core.util
      

      #self.operating_voltage            = exp_config.tech_config.core.operating_voltage

      self.threshold_voltage            = exp_config.tech_config.core.threshold_voltage
      self.margin_voltage               = exp_config.tech_config.core.margin_voltage
      self.operating_area_per_mcu       = exp_config.tech_config.core.operating_area_per_mcu
      self.num_mcu_per_bundle           = exp_config.tech_config.core.num_mcu_per_bundle
      self.num_mcu                      = int(self.tot_area // self.operating_area_per_mcu)
      self.num_bundle                   = int(self.num_mcu // self.num_mcu_per_bundle)
      self.area_scaling                 = self.operating_area_per_mcu / self.nominal_area_per_mcu
      #Assumption: performance scales linearly with area
      self.operating_flop_rate_per_mcu  = self.nominal_flop_rate_per_mcu * self.area_scaling
      self.nominal_power                = self.nominal_power_per_mcu * self.num_mcu * self.area_scaling
      
      if self.tot_power > 0 and self.nominal_power > 0:
        self.calcOperatingVoltageFrequency()
      else:
        self.operating_freq = 0
      
      self.calcThroughput()


  def calcOperatingVoltageFrequency(self):
      #minimum voltage that meets power constraints
      self.operating_voltage            = self.solve_poly(p0 = 1, 
                                                          p1 = -2 * self.threshold_voltage, 
                                                          p2 = self.threshold_voltage**2, 
                                                          p3 = -1 * self.tot_power / self.nominal_power * self.nominal_voltage * (self.nominal_voltage - self.threshold_voltage)**2)
      #Calculate operating frequency at minimum voltage
      self.operating_freq               = self.nominal_freq * (((self.operating_voltage - self.threshold_voltage)**2 / (self.operating_voltage)) /
                                                              ((self.nominal_voltage - self.threshold_voltage)**2 / self.nominal_voltage)) 
      self.frequency_scaling_factor = 1 
      if self.operating_voltage < (self.threshold_voltage + self.margin_voltage):
          self.scaled_voltage           = self.threshold_voltage + self.margin_voltage
          self.frequency_scaling_factor = (self.operating_voltage / self.scaled_voltage)**2
          self.operating_voltage        = self.scaled_voltage
      
      self.operating_freq               = self.frequency_scaling_factor * self.operating_freq

      self.operating_power_per_mcu      = self.nominal_power_per_mcu * (self.operating_freq / self.nominal_freq) * (self.operating_voltage / self.nominal_voltage)**2

  def calcThroughput(self):
     self.operating_throughput         = self.operating_flop_rate_per_mcu * self.operating_freq * self.num_mcu
     self.throughput                   = self.operating_throughput * self.util

  def printStats(self, f):
     self.eff_power              = self.num_mcu * self.operating_power_per_mcu
     self.eff_area               = self.num_mcu * self.operating_area_per_mcu
     f.write("\n\n=============\n")
     f.write("Core\n")
     f.write("=============\n")
     f.write("operating_volatge: {0:.2f}, operating_freq: {1:.2f} (Ghz)\n".format(self.operating_voltage, self.operating_freq/1e9))
     f.write("voltage_lowerbound: {0:.2f}\n".format(self.threshold_voltage + self.margin_voltage))
     f.write("#mcu: {0:5d}, #bundles: {1:5d}\n".format(self.num_mcu, self.num_bundle))
     f.write("eff_area: {0:.2f} (mm2), tot_area: {1:.2f} (mm2), util: {2:.2f}%\n".format(self.eff_area, self.tot_area, self.eff_area/self.tot_area * 100 ))
     f.write("eff_power: {0:.2f} (watt), tot_power: {1:.2f} (watt), util: {2:.2f}%\n".format(self.eff_power, self.tot_power, self.eff_power/self.tot_power * 100 ))


class MemoryHierarchy(Base):
  def __init__(self, exp_config):
      super().__init__(exp_config)
      self.num_levels = exp_config.memory_hierarchy.num_levels
      self.memLayer = [None] * self.num_levels
      
      for level in range(0,self.num_levels):
        mem_config = exp_config.memory_hierarchy.mem_hr[level]

        if mem_config.type == 'DRAM':
          self.memLayer[level] = DRAM(exp_config, mem_config, level)
        elif mem_config.type == 'SRAM-R':
          self.memLayer[level] = SRAM(exp_config, exp_config.power_breakdown.reg_mem, exp_config.area_breakdown.reg_mem, exp_config.tech_config.SRAMR, mem_config, level)
        elif mem_config.type == 'SRAM-L1':
          self.memLayer[level] = SRAM(exp_config, exp_config.power_breakdown.L1, exp_config.area_breakdown.L1, exp_config.tech_config.SRAML1, mem_config, level)
        elif mem_config.type == 'SRAM-L2':
          self.memLayer[level] = SRAM(exp_config, exp_config.power_breakdown.L2, exp_config.area_breakdown.L2, exp_config.tech_config.SRAML2, mem_config, level)
        else:
          NotImplemented()
      

class DRAM(Memory):
  def __init__(self, exp_config, mem_config, level):
      super().__init__(exp_config, level)
      self.tot_power                  = exp_config.power_breakdown.DRAM * self.TDP
      self.tot_area                   = exp_config.area_breakdown.node_area_budget - self.proc_chip_area_budget
      self.tot_mem_ctrl_area          = self.proc_chip_area_budget * exp_config.area_breakdown.DRAM
      self.mem_ctrl_area              = exp_config.tech_config.DRAM.mem_ctrl_area
      self.dynamic_energy_per_bit     = exp_config.tech_config.DRAM.dynamic_energy_per_bit
      self.static_power_per_byte      = exp_config.tech_config.DRAM.static_power_per_bit * 8
      self.area_per_byte              = exp_config.tech_config.DRAM.area_per_bit * 8
      self.stack_capacity             = exp_config.tech_config.DRAM.stack_capacity
      self.area_per_stack             = exp_config.tech_config.DRAM.area_per_stack
      self.latency                    = exp_config.tech_config.DRAM.latency
      self.scope                      = mem_config.scope
      self.util                       = exp_config.tech_config.DRAM.util
      self.nominal_freq               = exp_config.tech_config.DRAM.nominal_frequency
      self.nominal_voltage            = exp_config.tech_config.DRAM.nominal_voltage
      self.threshold_voltage          = exp_config.tech_config.DRAM.threshold_voltage
      self.margin_voltage             = exp_config.tech_config.DRAM.margin_voltage
      self.max_voltage                = exp_config.tech_config.DRAM.max_voltage
      self.num_stacks                 = int(min(self.tot_area // self.area_per_stack, 
                                            self.tot_mem_ctrl_area // self.mem_ctrl_area))
      self.num_links_per_mm           = exp_config.tech_config.DRAM.num_links_per_mm
      self.num_links_per_stack        = exp_config.tech_config.DRAM.num_links_per_stack

      self.perimeter_bound            = int(self.core_perimeter * 
                                            exp_config.perimeter_breakdown.DRAM *
                                            self.num_links_per_mm)
      self.num_links                  = min(self.perimeter_bound, self.num_links_per_stack * self.num_stacks)
      self.calcSize()
      self.calcActiveEnergy()
      
      self.nominal_power            = self.dynamic_energy_per_bit * self.num_links * self.nominal_freq
      if (self.dynamic_power > 0 and self.nominal_power > 0):
          self.calcOperatingVoltageFrequency()
      else:
          self.operating_freq = 0

      self.calcThroughput()

      if self.dynamic_throughput <= 0:
        assert(self.dynamic_throughput == 0)
        self.num_stacks=0

      self.calcSize()
      self.calcTileDim()

  def calcOperatingVoltageFrequency(self):
      self.frequency_scaling_factor     = 1 
      self.operating_voltage            = self.solve_poly(p0 = 1, 
                                                          p1 = -2 * self.threshold_voltage, 
                                                          p2 = self.threshold_voltage**2, 
                                                          p3 = -1 * self.dynamic_power / self.nominal_power * self.nominal_voltage * (self.nominal_voltage - self.threshold_voltage)**2)
      #operating frequency at minimum voltage
      self.operating_freq               = self.nominal_freq * (((self.operating_voltage - self.threshold_voltage)**2 / (self.operating_voltage)) /
                                                              ((self.nominal_voltage - self.threshold_voltage)**2 / self.nominal_voltage)) 
      self.frequency_scaling_factor = 1 
      if self.operating_voltage < (self.threshold_voltage + self.margin_voltage):
          self.scaled_voltage           = self.threshold_voltage + self.margin_voltage
          self.frequency_scaling_factor = (self.operating_voltage / self.scaled_voltage)**2
          self.operating_freq           = self.frequency_scaling_factor * self.operating_freq
          self.operating_voltage        = self.scaled_voltage
    
      elif self.operating_voltage > self.max_voltage:
          self.max_freq           = self.operating_freq * (((self.max_voltage - self.threshold_voltage)**2 / (self.max_voltage)) /
                                                          ((self.operating_voltage - self.threshold_voltage)**2 / self.operating_voltage))
          self.operating_freq     = self.max_freq
          self.operating_voltage  = self.max_voltage
    

  def calcActiveEnergy(self):
      self.dynamic_power             = 0 if (self.tot_power < self.static_power_per_byte * self.size) else (self.tot_power - self.static_power_per_byte * self.size)

  def calcThroughput(self):
      self.dynamic_throughput         = 0 if (self.size == 0) else self.num_links * self.operating_freq / 8
      self.stack_bw                   = 0 if self.num_stacks == 0 else self.dynamic_throughput  / self.num_stacks
      self.throughput                 = self.dynamic_throughput * self.util
  
  def calcSize(self):
      #self.nominal_throughput         = self.tot_power / self.dynamic_energy_per_byte
      #self.size                       = min((self.nominal_throughput / self.stack_bw) * self.stack_capacity,
      #                                         self.cell_area / self.area_per_byte)
      self.size                       = self.num_stacks * self.stack_capacity


  def printStats(self, f):
      self.operating_dynamic_energy_per_bit  = self.dynamic_energy_per_bit * (self.operating_voltage / self.nominal_voltage)**2
      self.dynamic_power                     = self.num_links * self.operating_dynamic_energy_per_bit * self.operating_freq
      self.static_power                      = self.static_power_per_byte * self.size
      self.eff_power                         = self.dynamic_power + self.static_power
      self.eff_ctrl_area                     = self.num_stacks * self.mem_ctrl_area #_per_stack
      self.eff_stack_area                    = self.num_stacks * self.area_per_stack
      f.write("\n\n=============\n")
      f.write("DRAM\n")
      f.write("=============\n")
      f.write("operating_volatge: {0:6.2f}\t\t operating_freq: {1:9.2f} (Ghz)\n".format(self.operating_voltage, self.operating_freq/1e9))
      f.write("voltage_lowerbound: {0:5.2f}\t\t voltage_upperbound: {1:5.2f}\n".format(self.threshold_voltage + self.margin_voltage, self.max_voltage))
      f.write("num_stacks: {0:10d}\t\t\t node_area_limit: {1:5d}\t\t\t chip_area_limit: {2:5d}\n".format(self.num_stacks, 
                                                                                                         int(self.tot_area // self.area_per_stack),
                                                                                                         int(self.tot_mem_ctrl_area // self.mem_ctrl_area))) 
      f.write("num_links: {0:14d}\t\t stack_limit: {1:13d}\t\t perimeter_limit: {2:8d}\n".format(self.num_links,
                                                                                                 self.perimeter_bound,
                                                                                                 self.num_links_per_stack * self.num_stacks)) 
      f.write("stack_bandwidth: {0:9.2f} (GB/s)\t stack_capacity: {1:9.2f} (GB)\n".format(self.stack_bw/giga, self.stack_capacity/giga))
      f.write("eff_ctrl_area: {0:11.2f} (mm2)\t tot_ctrl_area: {1:11.2f} (mm2)\t\t\t\t\t\t\t\t\t\t util: {2:.2f}%\n".format(self.eff_ctrl_area, self.tot_mem_ctrl_area, self.eff_ctrl_area/self.tot_mem_ctrl_area * 100 ))
      f.write("eff_stack_area: {0:11.2f} (mm2)\t tot_stack_area: {1:11.2f} (mm2)\t\t\t\t\t\t\t\t\t\t util: {2:.2f}%\n".format(self.eff_stack_area, self.tot_area, self.eff_stack_area/self.tot_area * 100 ))
      f.write("dynamic_power: {0:11.2f}\t\t static_power: {1:11.2f}\t\t eff_power: {2:15.2f} (watt)\t tot_power: {3:.2f} (watt)\t\t util: {4:.2f}%\n".format(self.dynamic_power, self.static_power, self.eff_power, self.tot_power, self.eff_power/self.tot_power * 100 ))


class SRAM(Memory):
  def __init__(self, exp_config, power_config, area_config, tech_config, mem_hierarchy_config, level):
      super().__init__(exp_config, level)
      self.tot_power                  = power_config * self.TDP
      self.tot_area                   = area_config * self.proc_chip_area_budget
      self.dynamic_energy_per_bit     = tech_config.dynamic_energy_per_bit
      self.dynamic_energy_per_byte    = self.dynamic_energy_per_bit * 8
      self.static_power_per_byte      = tech_config.static_power_per_bit * 8
      self.area_per_byte              = tech_config.area_per_bit * 8
      self.bank_capacity              = tech_config.bank_capacity
      self.controller_area_per_link   = tech_config.controller_area_per_link
      self.latency                    = tech_config.latency
      self.overhead                   = tech_config.overhead #percetage of cells dedicated to cicuitry overhead for SRAM cells
      self.cell_percentage            = 1 - self.overhead
      self.util                       = tech_config.util
      self.scope                      = mem_hierarchy_config.scope
      self.type                       = mem_hierarchy_config.type
      self.bank_area                  = self.bank_capacity * self.area_per_byte
      self.num_banks                  = int(math.floor((self.cell_percentage * self.tot_area) // (self.bank_area + self.cell_percentage * self.core.num_bundle * self.controller_area_per_link)))

      self.core                       = Core(self.exp_config)
      
      self.calcArea()
      self.calcSize()
      self.calcActiveEnergy()
      self.calcThroughput()
      
      if self.dynamic_throughput <= 0:
        self.num_banks=0
      
      self.calcSize()
      self.calcTileDim()

  def calcArea(self):
      self.overhead_area              = self.num_banks * self.core.num_bundle * self.controller_area_per_link
      self.cell_area                  = (self.tot_area - self.overhead_area) * self.cell_percentage 
      if self.overhead_area > self.tot_area:
          self.cell_area              = 0
  
  def calcSize(self):
      self.size                       = self.num_banks * self.bank_capacity
  
  def calcActiveEnergy(self):
      self.static_power             = self.static_power_per_byte * self.size
      self.dynamic_power            = 0 if (self.tot_power < self.static_power) else (self.tot_power - self.static_power)

  def calcThroughput(self):
      self.dynamic_throughput         = 0 if (self.num_banks == 0) else self.dynamic_power / self.dynamic_energy_per_byte
      self.throughput                 = self.dynamic_throughput * self.util

      self.bank_bw                    = 0 if (self.num_banks == 0) else self.dynamic_throughput / self.num_banks
  
 
  def printStats(self, f):
      self.dynamic_power                     = self.dynamic_throughput * self.dynamic_energy_per_byte
      self.eff_power                         = self.dynamic_power + self.static_power
      self.ctrl_area                         = self.num_banks * self.core.num_bundle * self.controller_area_per_link
      self.tot_bank_area                     = self.num_banks * self.bank_area / self.cell_percentage
      f.write("\n\n=============\n")
      f.write("{}\n".format(self.type))
      f.write("=============\n")
      f.write("num_banks: {0:17d}\n".format(self.num_banks)) 
      f.write("bank_bandwidth: {0:13.2f} (GB/s)\t bank_capacity: {1:9.2f} (GB)\n".format(self.bank_bw/giga, self.bank_capacity/kilo))
      f.write("ctrl_area: {0:17.2f} (mm2)\t\t bank_area: {1:11.2f} (mm2)\t tot_area: {2:11.2f}(mm2)\t\t\t util: {3:.2f}%\n".format(self.ctrl_area, self.tot_bank_area, self.tot_area, (self.ctrl_area + self.tot_bank_area)/self.tot_area * 100 ))
      f.write("dynamic_power: {0:13.2f} (watt)\t\t static_power: {1:11.2f} (watt)\t\t eff_power: {2:15.2f} (watt)\t tot_power: {3:.2f} (watt)\t\t util: {4:.2f}%\n".format(self.dynamic_power, self.static_power, self.eff_power, self.tot_power, self.eff_power/self.tot_power * 100 ))


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

  def calcThroughput(self):
      inter_throughput = self.inter_network.calcThroughput() 
      intra_throughput = self.intra_network.calcThroughput()

      return intra_throughput, inter_throughput

  def calcLatency(self):
      inter_latency = self.inter_network.latency
      intra_latency = self.intra_network.latency

      return intra_latency, inter_latency

  def printStats(self, f):
      self.inter_network.printStats(f, "inter")
      self.intra_network.printStats(f, "intra")

class SubNetwork(Base):
  def __init__(self, exp_config, net_config, power_breakdown, area_breakdown, netLevel):
      super().__init__(exp_config)
      self.tot_power                  = power_breakdown * self.TDP
      self.tot_area                   = area_breakdown  * self.proc_chip_area_budget
      #TODO: Rename core_perimeter to proc_chip_perimeter
      self.latency                    = net_config.latency
      self.nominal_freq               = net_config.nominal_freq
      self.nominal_voltage            = net_config.nominal_voltage
      self.nominal_energy_per_link    = net_config.nominal_energy_per_link
      self.nominal_area_per_link      = net_config.nominal_area_per_link
      self.threshold_voltage          = net_config.threshold_voltage
      self.margin_voltage             = net_config.margin_voltage
      self.num_links_per_mm           = net_config.num_links_per_mm
      self.util                       = net_config.util
      self.inter                      = True if netLevel == 'inter' else False
      self.intra                      = True if netLevel == 'intra' else False

      inter_frac                      = exp_config.perimeter_breakdown.inter_node
      intra_frac                      = exp_config.perimeter_breakdown.intra_node
      perimeter_fraction              = inter_frac if self.inter else intra_frac
      self.tot_perimeter              = perimeter_fraction * self.core_perimeter 
      self.num_links                  = int(min(self.tot_area / 
                                                self.nominal_area_per_link, 
                                                perimeter_fraction * 
                                                self.core_perimeter * 
                                                self.num_links_per_mm))
     
      self.throughput                 = 0
      self.operating_freq             = 0
      self.operating_voltage          = 0
      if self.num_links > 0:
          self.calcOperatingVoltageFrequency()
          self.throughput                 = self.calcThroughput()
       
      #self.energy_per_bit             = self.calcEnergyPerBit()


  def calcOperatingVoltageFrequency(self):
      self.nominal_power              = self.nominal_energy_per_link * self.num_links * self.nominal_freq
      #minimum voltage to meet the power constraint
      self.operating_voltage          = self.solve_poly(p0 = 1, 
                                                        p1 = -2 * self.threshold_voltage, 
                                                        p2 = self.threshold_voltage**2, 
                                                        p3 = -1 * self.tot_power / self.nominal_power * self.nominal_voltage * (self.nominal_voltage - self.threshold_voltage)**2)
      #operating frequency at minimum voltage
      self.operating_freq             = self.nominal_freq * (((self.operating_voltage - self.threshold_voltage)**2 / (self.operating_voltage)) /
                                                              ((self.nominal_voltage - self.threshold_voltage)**2 / self.nominal_voltage)) 
      self.frequency_scaling_factor     = 1 
      if self.operating_voltage < (self.threshold_voltage + self.margin_voltage):
          self.scaled_voltage           = self.threshold_voltage + self.margin_voltage
          self.frequency_scaling_factor = (self.operating_voltage / self.scaled_voltage)**2

      self.operating_freq               = self.frequency_scaling_factor * self.operating_freq

  def calcEnergyPerBit(self):
      self.operating_energy_per_link    = (self.nominal_energy_per_link * 
                                         (self.operating_voltage / self.nominal_voltage)**2)
      energy_per_bit                    = self.operating_energy_per_link
      return energy_per_bit

  #Return P2P bw
  def calcThroughput(self):
      #TODO: update this to support other network topology
      #4 edges comes out of each node on wafer since we assume a mesh topology
      #1 edges for cross-wafer as each node is connected to only one node on the other wafer (mesh extension)
      degree = 4 if self.intra else 1
      throughput                      = (self.num_links * self.operating_freq * self.util)/(8 * degree)
      return throughput

  def printStats(self, f, name):
     self.calcEnergyPerBit()
     self.eff_power              = self.num_links * self.operating_freq * self.operating_energy_per_link
     self.eff_area               = self.num_links * self.nominal_area_per_link
     self.eff_perimeter          = self.num_links / self.num_links_per_mm
    
     if self.eff_power > 0  and self.eff_area > 0:
        f.write("\n\n=============\n")
        f.write("Network: {}\n".format(name))
        f.write("=============\n")
        f.write("operating_volatge: {0:.2f}, operating_freq: {1:.2f} (Ghz)\n".format(self.operating_voltage, self.operating_freq/1e9))
        f.write("voltage_lowerbound: {0:.2f}\n".format(self.threshold_voltage + self.margin_voltage))
        f.write("#links: {0:5d}\n".format(self.num_links))
        if self.tot_area !=0:
            f.write("eff_area: {0:.2f} (mm2), tot_area: {1:.2f} (mm2), util: {2:.2f}%\n".format(self.eff_area, self.tot_area, self.eff_area/self.tot_area * 100 ))
        if self.tot_power != 0:
            f.write("eff_power: {0:.2f} (watt), tot_power: {1:.2f} (watt), util: {2:.2f}%\n".format(self.eff_power, self.tot_power, self.eff_power/self.tot_power * 100 ))
        if self.tot_perimeter != 0:
            f.write("eff_perimeter: {0:.2f} (mm), tot_perimeter: {1:.2f} (mm), util: {2:.2f}%\n".format(self.eff_perimeter, self.tot_perimeter, self.eff_perimeter/self.tot_perimeter * 100 ))


