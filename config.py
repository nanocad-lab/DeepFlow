import ruamel as _ruamel
import ruamel.yaml as _yaml
import math
from collections import namedtuple as _namedtuple

class CoreConfig:
  def __init__(self, core_config_dict):
    self.nominal_power_per_mcu      = core_config_dict['nominal_power_per_mcu']
    self.nominal_flop_rate_per_mcu  = core_config_dict['nominal_flop_rate_per_mcu']
    self.nominal_area_per_mcu       = core_config_dict['nominal_area_per_mcu']
    self.nominal_frequency          = core_config_dict['nominal_frequency']
    self.nominal_voltage            = core_config_dict['nominal_voltage']
    self.threshold_voltage          = core_config_dict['threshold_voltage']
    self.margin_voltage             = core_config_dict['margin_voltage']
    self.operating_area_per_mcu     = core_config_dict['operating_area_per_mcu']
    self.num_mcu_per_bundle         = core_config_dict['num_mcu_per_bundle']
    self.FMA_width                  = core_config_dict['FMA_width']
    self.dataflow                   = core_config_dict['dataflow']
    self.util                       = core_config_dict['util']

class DRAMConfig:
  def __init__(self, dram_config_dict):
    self.dynamic_energy_per_bit   = dram_config_dict['dynamic_energy_per_bit']
    self.static_power_per_bit     = dram_config_dict['static_power_per_bit']
    self.area_per_bit             = dram_config_dict['area_per_bit']
    self.stack_capacity           = dram_config_dict['stack_capacity']
    self.area_per_stack           = dram_config_dict['area_per_stack']
    self.latency                  = dram_config_dict['latency']
    self.mem_ctrl_area            = dram_config_dict['mem_ctrl_area']
    self.nominal_frequency        = dram_config_dict['nominal_frequency']
    self.nominal_voltage          = dram_config_dict['nominal_voltage']
    self.threshold_voltage        = dram_config_dict['threshold_voltage']
    self.margin_voltage           = dram_config_dict['margin_voltage']
    self.num_links_per_mm         = dram_config_dict['num_links_per_mm']
    self.num_links_per_stack      = dram_config_dict['num_links_per_stack']
    self.max_voltage              = dram_config_dict['max_voltage']
    self.util                     = dram_config_dict['util']

class SRAMConfig:
  def __init__(self, sram_config_dict):
    self.dynamic_energy_per_bit   = sram_config_dict['dynamic_energy_per_bit']
    self.static_power_per_bit     = sram_config_dict['static_power_per_bit']
    self.area_per_bit             = sram_config_dict['area_per_bit']
    self.bank_capacity            = sram_config_dict['bank_capacity']
    self.controller_area_per_link = sram_config_dict['controller_area_per_link']
    self.latency                  = sram_config_dict['latency']
    self.overhead                 = sram_config_dict['overhead']
    self.util                     = sram_config_dict['util']


class NetworkConfig:
  def __init__(self, net_config_dict):
    self.intra_node               = SubNetworkConfig(net_config_dict['intra_node'])
    self.inter_node               = SubNetworkConfig(net_config_dict['inter_node'])

class SubNetworkConfig:
  def __init__(self, config_dict):
    self.latency                  = config_dict['latency']
    self.nominal_freq             = config_dict['nominal_frequency']
    self.nominal_voltage          = config_dict['nominal_voltage']
    self.nominal_energy_per_link  = config_dict['nominal_energy_per_link']
    self.nominal_area_per_link    = config_dict['nominal_area_per_link']
    self.threshold_voltage        = config_dict['threshold_voltage']
    self.margin_voltage           = config_dict['margin_voltage']
    self.num_links_per_mm         = config_dict['num_links_per_mm']
    self.util                     = config_dict['util']


class TechConfig:
  def __init__(self, tech_config_dict):
    self.core                   = CoreConfig(tech_config_dict['core'])
    self.DRAM                   = DRAMConfig(tech_config_dict['DRAM'])
    self.SRAML2                 = SRAMConfig(tech_config_dict['SRAM-L2'])
    self.SRAML1                 = SRAMConfig(tech_config_dict['SRAM-L1'])
    self.SRAMR                  = SRAMConfig(tech_config_dict['SRAM-R'])
    self.network                = NetworkConfig(tech_config_dict['network'])

class AreaBreakdownConfig:
  def __init__(self, config_dict):
    self.proc_chip_area_budget  = config_dict['proc_chip_area_budget']
    self.core                   = config_dict['core']
    self.DRAM                   = config_dict['DRAM']
    self.L2                     = config_dict['L2']
    self.L1                     = config_dict['L1']
    self.reg_mem                = config_dict['reg_mem']
    self.node_area_budget       = config_dict['device_area_budget']
    self.network                = NetworkAreaConfig(config_dict['network'])
    
    tot_sum                     = (self.core + self.DRAM + self.L2 + 
                                   self.L1 + self.reg_mem + 
                                   self.network.inter_node + 
                                   self.network.intra_node)
    #assert (tot_sum == 1), \
    #        "Error: Area fractions are not adding up to 1 (current sum = {})".format(tot_sum)

class PerimeterBreakdownConfig:
  def __init__(self, config_dict):
    self.DRAM                   = config_dict['DRAM']
    self.inter_node             = config_dict['inter_node']
    self.intra_node             = config_dict['intra_node']
    tot_sum                     = self.DRAM + self.inter_node + self.intra_node
    #assert (tot_sum == 1), \
    #        "Error: Perimeter fractions are not adding up to 1 (current sum = {})".format(tot_sum)

class NetworkAreaConfig:
  def __init__(self, config_dict):
    self.inter_node = config_dict['inter_node']
    self.intra_node = config_dict['intra_node']

class PowerBreakdownConfig:
  def __init__(self, config_dict):
    self.TDP                    = config_dict['TDP']
    self.core                   = config_dict['core']
    self.DRAM                   = config_dict['DRAM']
    self.L2                     = config_dict['L2']
    self.L1                     = config_dict['L1']
    self.reg_mem                = config_dict['reg_mem']
    self.network                = NetworkPowerConfig(config_dict['network'])

    tot_sum                     = (self.core + self.DRAM + self.L2 + 
                                   self.L1 + self.reg_mem + 
                                   self.network.inter_node + 
                                   self.network.intra_node)
    #assert (tot_sum == 1), \
    #        "Error: Power fractions are not adding up to 1 (current sum = {})".format(tot_sum)

class NetworkPowerConfig:
  def __init__(self, config_dict):
    self.inter_node            = config_dict['inter_node']
    self.intra_node            = config_dict['intra_node']

    self.intra_node            = config_dict['intra_node']

    self.intra_node            = config_dict['intra_node']

class SystemHierarchyConfig:
  def __init__(self, config_dict):
    #A device is an accelerator which can itself be composed of many single cores
    #This number does not say anything about number of cores within an accelerator.
    #It is the number of accelerators/devices per wafer.
    self.num_nodes_per_wafer  = config_dict['num_devices_per_node'] 
    #This is redundant but makes my life easier.
    self.num_wafers           = config_dict['num_nodes']
    self.num_workers          = int(self.num_wafers * self.num_nodes_per_wafer)
    #self.num_wafers           = int(math.ceil(self.num_workers / self.num_nodes_per_wafer))
    self.inter_derate         = config_dict['inter_derate']
    self.intra_derate         = config_dict['intra_derate']
    self.kp1_inter            = config_dict['kp1_inter']
    self.kp2_inter            = config_dict['kp2_inter']
    self.dp_inter             = config_dict['dp_inter']
    self.lp_inter             = config_dict['lp_inter']
    self.par2cross = {'kp1': self.kp1_inter, 'kp2': self.kp2_inter, 'dp': self.dp_inter, 'lp': self.lp_inter}

class TopologyConfig:
  def __init__(self, config_dict):
    self.topology = None
    if config_dict == 'hybrid':
      NotImplemented()
    else:
      self.topology = config_dict
      

class NetworkTopologyConfig:
  def __init__(self, config_dict):
    self.inter = TopologyConfig(config_dict['inter_node'])
    self.intra = TopologyConfig(config_dict['intra_node'])


class MemoryConfig:
  def __init__(self, config_dict):
    self.type = config_dict['type']
    self.scope = config_dict['scope']

class MemoryHierarchyConfig:
  def __init__(self, config_dict):
    self.num_levels = len(config_dict)
    self.mem_hr     = [None] * self.num_levels
    for level in range(0,self.num_levels):
      m = MemoryConfig(config_dict['l' + str(level)])
      self.mem_hr[level] =  m

ModelConfig = _namedtuple("model_param", ["batch_size", "vocab_size", 
                          "num_layers", "layer_size", "seq_len", "projection", 
                          "num_gates", "num_non_linear", "num_add", "data_scale"])

SWConfig = _namedtuple("sw_param", ["kernel_launch_overhead", 
                                        "precision"])

SchedulingConfig = _namedtuple("scheduling_param", ["auto",
                                                    "dp",
                                                    "lp", 
                                                    "kp_hidden_dim1", 
                                                    "kp_softmax_dim1", 
                                                    "kp_embedding_dim1", 
                                                    "kp_projection_dim1", 
                                                    "kp_hidden_dim2", 
                                                    "kp_softmax_dim2", 
                                                    "kp_embedding_dim2", 
                                                    "kp_projection_dim2", 
                                                    "kp_hidden_type",
                                                    "kp_softmax_type",
                                                    "kp_embedding_type",
                                                    "kp_projection_type"])

FullConfig = _namedtuple("FullConfig",["model_config", "sw_config",
                         "tech_config", "power_breakdown", "sch_config", 
                         "area_breakdown", "perimeter_breakdown", 
                         "system_config", "memory_hierarchy",
                         "network_topology"])

def convert(d):
  for key1, val1 in d.items():
    for key2, val2 in val1.items():
      if isinstance(val2,dict):
        for key3, val3 in val2.items():
          if isinstance(val3, str):
            digit = [int(s) for s in val3.split() if s.isdigit()]
            order = [str(s) for s in val3.split() if not s.isdigit()]
            if order and digit:
              assert(len(order) >= 1)
              assert(len(digit) >= 1)

              prefix = order[0][0]
              bit = order[0][1]
              mult = 1

              if (prefix == "K"):
                  mult = 1024
              elif (prefix == "M"):
                  mult = 1024*1024
              elif (prefix == "G"):
                  mult = 1024*1024*1024
              elif (prefix == "T"):
                  mult = 1024*1024*1024*1024
              else:
                  print("Unknown prefix: {} at {}: {}".format(prefix, key3, val3))
                  exit(0)

              if (bit == "b"):
                  mult = mult / 8 #Capacity is expected in Bytes 
              elif (bit =="B"):
                  mult = mult
              else: 
                  print("Unknown type: {} at {}: {}".format(bit, key3, val3))
                  exit(0)

              new_val = digit[0] * mult
              d[key1][key2][key3] = new_val


def parse_config(filename):
  """Parse a yaml configuration file for this experiment.
  Args:
          filename (str): Path to the configuration file
  Returns:
          FullConfig: Contains dataset, model, optimization, training and
          scheduling configurations
  """
  with open(filename, "r") as f:
    config_dict = _yaml.load(f, Loader=_ruamel.yaml.Loader)
    convert(config_dict)

  model_config      = ModelConfig(**config_dict['model_param'])
  sw_config         = SWConfig(**config_dict['sw_param'])
  sch_config        = SchedulingConfig(**config_dict['scheduling_param'])
  tech_config       = TechConfig(config_dict['tech_param'])
  power_config      = PowerBreakdownConfig(config_dict['power_breakdown'])
  area_config       = AreaBreakdownConfig(config_dict['area_breakdown'])
  perimeter_config  = PerimeterBreakdownConfig(config_dict['perimeter_breakdown'])
  system_config     = SystemHierarchyConfig(config_dict['system_hierarchy'])
  memory_hierarchy_config     = MemoryHierarchyConfig(config_dict['memory_hierarchy'])
  network_topology_config     = NetworkTopologyConfig(config_dict['network_topology'])

  return FullConfig(model_config=model_config, sw_config=sw_config, 
                    sch_config=sch_config, tech_config=tech_config, 
                    power_breakdown=power_config, area_breakdown=area_config,
                    perimeter_breakdown=perimeter_config,
                    system_config=system_config,
                    memory_hierarchy=memory_hierarchy_config,
                    network_topology=network_topology_config)
