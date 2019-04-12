import ruamel as _ruamel
import ruamel.yaml as _yaml
from collections import namedtuple as _namedtuple


ModelConfig = _namedtuple("model_param", ["batch_size", "vocab_size", 
                          "num_layers", "layer_size", "seq_len", "projection", 
                          "num_gates", "num_non_linear", "num_add"])
ArchConfig = _namedtuple("arch_param", ["kernel_launch_overhead", 
                                        "precision"])
TechConfig = _namedtuple("tech_param", ["DRAM_energy_per_bit_trans",
                                        "L2_energy_per_bit",
                                        "shared_mem_energy_per_bit",
                                        "core_energy_per_flop",
                                        "internode_energy_per_bit",
                                        "HBM_stack_capacity",
                                        "HBM_stack_bw",
                                        "L2_bank_capacity",
                                        "L2_bank_bw",
                                        "shared_mem_bank_bw",
                                        "shared_mem_bank_capacity",
                                        "line_latency"])
PowerBreakConfig = _namedtuple("power_breakdown", ["TDP", "core", "DRAM",
                                                   "L2", "shared_mem", "IB"])
SchedulingConfig = _namedtuple("scheduling_param", ["dp",
                                                    "lp", 
                                                    "hlp", 
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

FullConfig = _namedtuple("FullConfig",["model_config", "arch_config",
                         "tech_config", "power_breakdown", "sch_config"])

def convert(d):
  for key1, val1 in d.items():
    for key2, val2 in val1.items():
      if isinstance(val2, str):

         digit = [int(s) for s in val2.split() if s.isdigit()]
         order = [str(s) for s in val2.split() if not s.isdigit()]
         if order:
           assert(len(order) == 1)
           assert(len(digit) == 1)

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
               print("Unknown prefix: {} at {}: {}".format(prefix, key2, val2))
               exit(0)

           if (bit == "b"):
               mult = mult * 8
           elif (bit =="B"):
               mult = mult
           else: 
               print("Unknown type: {} at {}: {}".format(bit, key2, val2))
               exit(0)

           new_val = digit[0] * mult
           d[key1][key2] = new_val


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
  arch_config = ArchConfig(**config_dict['arch_param'])
  sch_config = SchedulingConfig(**config_dict['scheduling_param'])
  tech_config = TechConfig(**config_dict['tech_param'])
  power_config = PowerBreakConfig(**config_dict['power_breakdown'])
  
  return FullConfig(model_config=model_config, arch_config=arch_config, 
                    sch_config=sch_config, tech_config=tech_config, power_breakdown=power_config)
