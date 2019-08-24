import ruamel as _ruamel
import ruamel.yaml as _yaml
import math
from collections import namedtuple as _namedtuple


class CoreConfig:
  def __init__(self, core_config_dict):
    self.nominal_power_per_mcu = core_config_dict['nominal_power_per_mcu']
    self.nominal_flop_rate_per_mcu = core_config_dict['nominal_flop_rate_per_mcu']
    self.nominal_area_per_mcu = core_config_dict['nominal_area_per_mcu']
    self.nominal_frequency = core_config_dict['nominal_frequency']
    self.nominal_voltage = core_config_dict['nominal_voltage']
    self.threshold_voltage = core_config_dict['threshold_voltage']
    self.margin_voltage = core_config_dict['margin_voltage']
    #self.operating_frequency = core_config_dict['operating_frequency']
    #self.operating_voltage = core_config_dict['operating_voltage']
    self.operating_area_per_mcu = core_config_dict['operating_area_per_mcu']
    self.num_mcu_per_bundle = core_config_dict['num_mcu_per_bundle']

class DRAMConfig:
  def __init__(self, mem_config_dict):
    self.dynamic_energy_per_bit = mem_config_dict['dynamic_energy_per_bit']
    self.static_power_per_bit = mem_config_dict['static_power_per_bit']
    self.area_per_bit = mem_config_dict['area_per_bit']
    self.stack_capacity = mem_config_dict['stack_capacity']
    self.stack_bw = mem_config_dict['stack_bandwidth']
    self.area_per_stack = mem_config_dict['area_per_stack']
    self.latency = mem_config_dict['latency']
    self.mem_ctrl_area = mem_config_dict['mem_ctrl_area']
    self.nominal_frequency = mem_config_dict['nominal_frequency']
    self.nominal_voltage = mem_config_dict['nominal_voltage']
    self.threshold_voltage = mem_config_dict['threshold_voltage']
    self.margin_voltage = mem_config_dict['margin_voltage']

class L2Config:
  def __init__(self, l2_config_dict):
    self.dynamic_energy_per_bit = l2_config_dict['dynamic_energy_per_bit']
    self.static_power_per_bit = l2_config_dict['static_power_per_bit']
    self.area_per_bit = l2_config_dict['area_per_bit']
    self.bank_capacity = l2_config_dict['bank_capacity']
    self.bank_bw = l2_config_dict['bank_bandwidth']
    self.controller_area_per_link = l2_config_dict['controller_area_per_link']
    self.latency                  = l2_config_dict['latency']

class SMConfig:
  def __init__(self, sm_config_dict):
    self.dynamic_energy_per_bit   = sm_config_dict['dynamic_energy_per_bit']
    self.static_power_per_bit     = sm_config_dict['static_power_per_bit']
    self.area_per_bit             = sm_config_dict['area_per_bit']
    self.bank_capacity            = sm_config_dict['bank_capacity']
    self.bank_bw                  = sm_config_dict['bank_bandwidth']
    self.controller_area_per_link = sm_config_dict['controller_area_per_link']
    self.latency                  = sm_config_dict['latency']

class RegConfig:
  def __init__(self, reg_config_dict):
    self.dynamic_energy_per_bit   = reg_config_dict['dynamic_energy_per_bit']
    self.static_power_per_bit     = reg_config_dict['static_power_per_bit']
    self.area_per_bit             = reg_config_dict['area_per_bit']
    self.bank_capacity            = reg_config_dict['bank_capacity']
    self.bank_bw                  = reg_config_dict['bank_bandwidth']
    self.controller_area_per_link = reg_config_dict['controller_area_per_link']
    self.latency                  = reg_config_dict['latency']

class NetworkConfig:
  def __init__(self, net_config_dict):
    self.num_links_per_mm         = net_config_dict['num_links_per_mm']
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
    #self.operating_freq           = config_dict['operating_frequency']
    #self.operating_voltage        = config_dict['operating_voltage']
    #self.num_links_per_mm         = config_dict['num_links_per_mm']

    #self.parallelMap              = ParallelMap(str(config_dict['parallelMap']))

#class ParallelMap:
#  def __init__(self, par2network):
#    self.data     = False
#    self.kernel   = False
#    self.layer    = False
#    if "data" in par2network:
#      self.data   = True
#    if "kernel" in par2network:
#      self.kernel = True
#    if "layer" in par2network:
#      self.layer = True
#

class TechConfig:
  def __init__(self, tech_config_dict):
    self.core = CoreConfig(tech_config_dict['core'])
    self.DRAM = DRAMConfig(tech_config_dict['DRAM'])
    self.L2 = L2Config(tech_config_dict['L2'])
    self.shared_mem = SMConfig(tech_config_dict['shared_mem'])
    self.reg_mem = RegConfig(tech_config_dict['reg_mem'])
    self.network = NetworkConfig(tech_config_dict['network'])

class AreaBreakdownConfig:
  def __init__(self, config_dict):
    self.proc_chip_area_budget = config_dict['proc_chip_area_budget']
    self.core = config_dict['core']
    self.DRAM = config_dict['DRAM']
    self.L2 = config_dict['L2']
    self.shared_mem = config_dict['shared_mem']
    self.reg_mem = config_dict['reg_mem']
    self.node_area_budget = config_dict['node_area_budget']
    self.network = NetworkAreaConfig(config_dict['network'])

class PerimeterBreakdownConfig:
  def __init__(self, config_dict):
    self.DRAM = config_dict['DRAM']
    self.inter_node = config_dict['inter_node']
    self.intra_node = config_dict['intra_node']

class NetworkAreaConfig:
  def __init__(self, config_dict):
    self.inter_node = config_dict['inter_node']
    self.intra_node = config_dict['intra_node']

class PowerBreakdownConfig:
  def __init__(self, config_dict):
    self.TDP = config_dict['TDP']
    self.core = config_dict['core']
    self.DRAM = config_dict['DRAM']
    self.L2 = config_dict['L2']
    self.shared_mem = config_dict['shared_mem']
    self.reg_mem = config_dict['reg_mem']
    self.network = NetworkPowerConfig(config_dict['network'])


class NetworkPowerConfig:
  def __init__(self, config_dict):
    self.inter_node = config_dict['inter_node']
    self.intra_node = config_dict['intra_node']

    self.intra_node = config_dict['intra_node']

    self.intra_node = config_dict['intra_node']

class SystemHierarchyConfig:
  def __init__(self, config_dict):
    #A node is an accelerator which can itself be composed of many single cores
    #This number does not say anything about number of cores within an accelerator.
    #It is the number of accelerators per wafer.
    self.num_nodes_per_wafer = config_dict['num_nodes_per_wafer'] 
    #This is redundant but makes my life easier.
    self.tot_nodes           = config_dict['tot_nodes']
    self.num_wafers          = math.ceil(self.tot_nodes / self.num_nodes_per_wafer)
    self.device_placement    = ParallelMap(config_dict['device_placement'], 
                                           self.num_wafers, 
                                           self.num_nodes_per_wafer)

class ParallelMap:
  def __init__(self, config_dict, num_wafers, num_nodes_per_wafer):
    self.par2Dev = {}
    for i in range(0, num_wafers):
        for j in range(0, num_nodes_per_wafer):
            parMapStr = config_dict['w' + str(i)]['n' + str(j)]
            parMapList = [int(x) for x in parMapStr.split(',')]
            parMapId = tuple(i for i in parMapList)
            hwId = (i,j)
            if parMapId not in self.par2Dev:
                self.par2Dev[parMapId] = hwId
            else:
                print("Duplicate mapping:")
                print("parallelMapping: {} has been mapped to {} and {}".
                      format(parMapId, hwId, self.par2Dev[parMapId]))
                exit(0)
  def getPar2Dev():
      return self.par2Dev
      

ModelConfig = _namedtuple("model_param", ["batch_size", "vocab_size", 
                          "num_layers", "layer_size", "seq_len", "projection", 
                          "num_gates", "num_non_linear", "num_add"])

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
                         "area_breakdown", "perimeter_breakdown", "system_config"])

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
                  mult = mult * 8
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

  model_config = ModelConfig(**config_dict['model_param'])
  sw_config = SWConfig(**config_dict['sw_param'])
  sch_config = SchedulingConfig(**config_dict['scheduling_param'])
  tech_config = TechConfig(config_dict['tech_param'])
  power_config = PowerBreakdownConfig(config_dict['power_breakdown'])
  area_config = AreaBreakdownConfig(config_dict['area_breakdown'])
  perimeter_config = PerimeterBreakdownConfig(config_dict['perimeter_breakdown'])
  system_config = SystemHierarchyConfig(config_dict['system_hierarchy'])

  return FullConfig(model_config=model_config, sw_config=sw_config, 
                    sch_config=sch_config, tech_config=tech_config, 
                    power_breakdown=power_config, area_breakdown=area_config,
                    perimeter_breakdown=perimeter_config,
                    system_config=system_config)
