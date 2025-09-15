from dataclasses import dataclass, field
import ruamel as _ruamel
import ruamel.yaml as _yaml
import math
from collections import namedtuple as _namedtuple


@dataclass
class CoreConfig:
    nominal_power_per_mcu: float
    nominal_flop_rate_per_mcu: float
    nominal_voltage: float
    threshold_voltage: float
    margin_voltage: float
    operating_area_per_mcu: float
    num_mcu_per_bundle: int
    FMA_dims: tuple
    dataflow: str
    util: float
    num_bundles: int = None
    operating_frequency: float = None
    nominal_frequency: float = None
    nominal_area_per_mcu: float = None

    @classmethod
    def from_dict(cls, core_config_dict):
        return cls(
            nominal_power_per_mcu=core_config_dict["nominal_power_per_mcu"],
            nominal_flop_rate_per_mcu=core_config_dict["nominal_flop_rate_per_mcu"],
            nominal_voltage=core_config_dict["nominal_voltage"],
            threshold_voltage=core_config_dict["threshold_voltage"],
            margin_voltage=core_config_dict["margin_voltage"],
            operating_area_per_mcu=core_config_dict["operating_area_per_mcu"],
            num_mcu_per_bundle=core_config_dict["num_mcu_per_bundle"],
            FMA_dims=(core_config_dict["FMA_d1"], core_config_dict["FMA_d2"]),
            dataflow=core_config_dict["dataflow"],
            util=core_config_dict["util"],
            num_bundles=core_config_dict.get("num_bundles", None),
            operating_frequency=core_config_dict.get("operating_frequency", None),
            nominal_frequency=core_config_dict.get("nominal_frequency", None),
            nominal_area_per_mcu=core_config_dict.get("nominal_area_per_mcu", None),
        )


@dataclass
class DRAMConfig:
    dynamic_energy_per_bit: float
    static_power_per_bit: float
    area_per_bit: float
    stack_capacity: float
    area_per_stack: float
    latency: float
    mem_ctrl_area: float
    nominal_voltage: float
    threshold_voltage: float
    margin_voltage: float
    num_links_per_mm: int
    num_links_per_stack: int
    max_voltage: float
    util: float
    size: float = None
    bandwidth: float = None
    num_stacks: int = None
    operating_frequency: float = None
    nominal_frequency: float = None

    @classmethod
    def from_dict(cls, dram_config_dict):
        return cls(
            dynamic_energy_per_bit=dram_config_dict["dynamic_energy_per_bit"],
            static_power_per_bit=dram_config_dict["static_power_per_bit"],
            area_per_bit=dram_config_dict["area_per_bit"],
            stack_capacity=dram_config_dict["stack_capacity"],
            area_per_stack=dram_config_dict["area_per_stack"],
            latency=dram_config_dict["latency"],
            mem_ctrl_area=dram_config_dict["mem_ctrl_area"],
            nominal_voltage=dram_config_dict["nominal_voltage"],
            threshold_voltage=dram_config_dict["threshold_voltage"],
            margin_voltage=dram_config_dict["margin_voltage"],
            num_links_per_mm=dram_config_dict["num_links_per_mm"],
            num_links_per_stack=dram_config_dict["num_links_per_stack"],
            max_voltage=dram_config_dict["max_voltage"],
            util=dram_config_dict["util"],
            size=dram_config_dict.get("size", None),
            bandwidth=dram_config_dict.get("bandwidth", None),
            num_stacks=dram_config_dict.get("num_stacks", None),
            operating_frequency=dram_config_dict.get("operating_frequency", None),
            nominal_frequency=dram_config_dict.get("nominal_frequency", None),
        )


@dataclass
class SRAMConfig:
    dynamic_energy_per_bit: float
    static_power_per_bit: float
    area_per_bit: float
    bank_capacity: float
    controller_area_per_link: float
    latency: float
    overhead: float
    util: float
    size: float = None
    bandwidth: float = None

    @classmethod
    def from_dict(cls, sram_config_dict):
        return cls(
            dynamic_energy_per_bit=sram_config_dict["dynamic_energy_per_bit"],
            static_power_per_bit=sram_config_dict["static_power_per_bit"],
            area_per_bit=sram_config_dict["area_per_bit"],
            bank_capacity=sram_config_dict["bank_capacity"],
            controller_area_per_link=sram_config_dict["controller_area_per_link"],
            latency=sram_config_dict["latency"],
            overhead=sram_config_dict["overhead"],
            util=sram_config_dict["util"],
            size=sram_config_dict.get("size", None),
            bandwidth=sram_config_dict.get("bandwidth", None),
        )


@dataclass
class SubNetworkConfig:
    latency: float
    nominal_freq: float
    nominal_voltage: float
    nominal_energy_per_link: float
    nominal_area_per_link: float
    threshold_voltage: float
    margin_voltage: float
    num_links_per_mm: int
    util: float

    @classmethod
    def from_dict(cls, config_dict):
        return cls(
            latency=config_dict["latency"],
            nominal_freq=config_dict["nominal_frequency"],
            nominal_voltage=config_dict["nominal_voltage"],
            nominal_energy_per_link=config_dict["nominal_energy_per_link"],
            nominal_area_per_link=config_dict["nominal_area_per_link"],
            threshold_voltage=config_dict["threshold_voltage"],
            margin_voltage=config_dict["margin_voltage"],
            num_links_per_mm=config_dict["num_links_per_mm"],
            util=config_dict["util"],
        )


@dataclass
class NetworkConfig:
    intra_node: SubNetworkConfig
    inter_node: SubNetworkConfig

    @classmethod
    def from_dict(cls, d):
        return cls(
            intra_node=SubNetworkConfig.from_dict(d["intra_node"]),
            inter_node=SubNetworkConfig.from_dict(d["inter_node"]),
        )


@dataclass
class TechConfig:
    core: CoreConfig
    DRAM: DRAMConfig
    SRAML2: SRAMConfig
    SRAML1: SRAMConfig
    SRAMR: SRAMConfig
    network: NetworkConfig

    @classmethod
    def from_dict(cls, tech_config_dict):
        return cls(
            core=CoreConfig.from_dict(tech_config_dict["core"]),
            DRAM=DRAMConfig.from_dict(tech_config_dict["DRAM"]),
            SRAML2=SRAMConfig.from_dict(tech_config_dict["SRAM-L2"]),
            SRAML1=SRAMConfig.from_dict(tech_config_dict["SRAM-L1"]),
            SRAMR=SRAMConfig.from_dict(tech_config_dict["SRAM-R"]),
            network=NetworkConfig.from_dict(tech_config_dict["network"]),
        )


@dataclass
class AreaBreakdownConfig:
    proc_chip_area_budget: float
    core: float
    DRAM: float
    L2: float
    L1: float
    reg_mem: float
    node_area_budget: float
    network: "NetworkAreaConfig"

    @classmethod
    def from_dict(cls, area_config_dict):
        return cls(
            proc_chip_area_budget=area_config_dict["proc_chip_area_budget"],
            core=area_config_dict["core"],
            DRAM=area_config_dict["DRAM"],
            L2=area_config_dict["L2"],
            L1=area_config_dict["L1"],
            reg_mem=area_config_dict["reg_mem"],
            node_area_budget=area_config_dict["device_area_budget"],
            network=NetworkAreaConfig.from_dict(area_config_dict["network"]),
        )


@dataclass
class PerimeterBreakdownConfig:
    DRAM: float
    inter_node: float
    intra_node: float

    @classmethod
    def from_dict(cls, perimeter_config_dict):
        return cls(
            DRAM=perimeter_config_dict["DRAM"],
            inter_node=perimeter_config_dict["inter_node"],
            intra_node=perimeter_config_dict["intra_node"],
        )


@dataclass
class NetworkAreaConfig:
    inter_node: float
    intra_node: float

    @classmethod
    def from_dict(cls, network_config_dict):
        return cls(
            inter_node=network_config_dict["inter_node"],
            intra_node=network_config_dict["intra_node"],
        )


@dataclass
class PowerBreakdownConfig:
    TDP: float
    core: float
    DRAM: float
    L2: float
    L1: float
    reg_mem: float
    network: "NetworkPowerConfig"

    @classmethod
    def from_dict(cls, power_config_dict):
        return cls(
            TDP=power_config_dict["TDP"],
            core=power_config_dict["core"],
            DRAM=power_config_dict["DRAM"],
            L2=power_config_dict["L2"],
            L1=power_config_dict["L1"],
            reg_mem=power_config_dict["reg_mem"],
            network=NetworkPowerConfig.from_dict(power_config_dict["network"]),
        )


@dataclass
class NetworkPowerConfig:
    inter_node: float
    intra_node: float

    @classmethod
    def from_dict(cls, network_power_config_dict):
        return cls(
            inter_node=network_power_config_dict["inter_node"],
            intra_node=network_power_config_dict["intra_node"],
        )


@dataclass
class SystemHierarchyConfig:
    num_nodes_per_wafer: int
    num_wafers: int
    num_workers: int
    inter_derate: float
    intra_derate: float
    kp1_inter: float
    kp2_inter: float
    dp_inter: float
    lp_inter: float
    par2cross: dict

    @classmethod
    def from_dict(cls, system_config_dict):
        return cls(
            num_nodes_per_wafer=system_config_dict["num_devices_per_node"],
            num_wafers=system_config_dict["num_nodes"],
            num_workers=int(
                system_config_dict["num_nodes"]
                * system_config_dict["num_devices_per_node"]
            ),
            inter_derate=system_config_dict["inter_derate"],
            intra_derate=system_config_dict["intra_derate"],
            kp1_inter=system_config_dict["kp1_inter"],
            kp2_inter=system_config_dict["kp2_inter"],
            dp_inter=system_config_dict["dp_inter"],
            lp_inter=system_config_dict["lp_inter"],
            par2cross={
                "kp1": system_config_dict["kp1_inter"],
                "kp2": system_config_dict["kp2_inter"],
                "dp": system_config_dict["dp_inter"],
                "lp": system_config_dict["lp_inter"],
            },
        )


@dataclass
class TopologyConfig:
    topology: str = None

    @classmethod
    def from_dict(cls, d):
        if d == "hybrid":
            NotImplemented()
        else:
            return cls(topology=d)


@dataclass
class NetworkTopologyConfig:
    inter: TopologyConfig
    intra: TopologyConfig

    @classmethod
    def from_dict(cls, d):
        return cls(
            inter=TopologyConfig.from_dict(d["inter_node"]),
            intra=TopologyConfig.from_dict(d["intra_node"]),
        )


@dataclass
class MemoryConfig:
    type: str
    scope: str

    @classmethod
    def from_dict(cls, d):
        return cls(
            type=d["type"],
            scope=d["scope"],
        )


@dataclass
class MemoryHierarchyConfig:
    num_levels: int
    mem_hr: list

    @classmethod
    def from_dict(cls, d):
        num_levels = len(d)
        mem_hr = [None] * num_levels
        for level in range(num_levels):
            m = MemoryConfig.from_dict(d["l" + str(level)])
            mem_hr[level] = m
        return cls(
            num_levels=num_levels,
            mem_hr=mem_hr,
        )


ModelLSTMConfig = _namedtuple(
    "model_param",
    [
        "mode",
        "batch_size",
        "vocab_size",
        "num_layers",
        "layer_size",
        "seq_len",
        "projection",
        "num_gates",
        "num_non_linear",
        "num_add",
        "data_scale",
    
    ],
)
GEMMConfig = _namedtuple(
    "model_param",
    [
        "mode",
        "M",
        "K",
        "N",
        "backward",
    ],
)
LLMConfig = _namedtuple(
    "model_param",
    [
        "mode",
        "num_layers",
        "hidden_dim",
        "num_heads",
        "batch_size",
        "seq_len",
        "ffn_dim",
        "ffn_mult",
        "vocab_size",
        "n_tokens",
        "communication_time",
        "N_PP",
    ],
)
SWConfig = _namedtuple("sw_param", ["kernel_launch_overhead", "precision"])

SchedulingConfig = _namedtuple(
    "scheduling_param",
    [
        "auto",
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
        "kp_projection_type",
        "t",
        "kp1",
        "kp2",
    ],
)

FullConfig = _namedtuple(
    "FullConfig",
    [
        "model_config",
        "sw_config",
        "tech_config",
        "power_breakdown",
        "sch_config",
        "area_breakdown",
        "perimeter_breakdown",
        "system_config",
        "memory_hierarchy",
        "network_topology",
    ],
)

NetworkBackendAstraCollectives = _namedtuple(
    "NetworkBackendAstraCollectives",
    [
        "all_gather",
        "all_reduce",
        "reduce_scatter",
        "all_to_all",
    ],
)

NetworkBackendAstra = _namedtuple(
    "NetworkBackendAstra",
    [
        "backend",   # analytical | ns3 | garnet
        "mode",      # isolated | snapshot
        "collectives",
    ],
)

NetworkBackend = _namedtuple(
    "NetworkBackend",
    [
        "model",   # analytical | astra
        "astra",   # NetworkBackendAstra or None
    ],
)

HWConfig = _namedtuple(
    "HWConfig",
    [
        "sw_config",
        "tech_config",
        "power_breakdown",
        "sch_config",
        "area_breakdown",
        "perimeter_breakdown",
        "system_config",
        "memory_hierarchy",
        "network_topology",
        "network_backend",
    ],
)

MODELConfig = _namedtuple(
    "MODELConfig",
    [
        "model_config",
        
    ],
)


def convert(d):
    for key1, val1 in d.items():
        for key2, val2 in val1.items():
            if isinstance(val2, dict):
                for key3, val3 in val2.items():
                    if isinstance(val3, str):
                        digit = [int(s) for s in val3.split() if s.isdigit()]
                        order = [str(s) for s in val3.split() if not s.isdigit()]
                        if order and digit:
                            assert len(order) >= 1
                            assert len(digit) >= 1

                            prefix = order[0][0]
                            bit = order[0][1]
                            mult = 1

                            if prefix == "K":
                                mult = 1024
                            elif prefix == "M":
                                mult = 1024 * 1024
                            elif prefix == "G":
                                mult = 1024 * 1024 * 1024
                            elif prefix == "T":
                                mult = 1024 * 1024 * 1024 * 1024
                            else:
                                print(
                                    "Unknown prefix: {} at {}: {}".format(
                                        prefix, key3, val3
                                    )
                                )
                                exit(0)

                            if bit == "b":
                                mult = mult / 8  # Capacity is expected in Bytes
                            elif bit == "B":
                                mult = mult
                            else:
                                print(
                                    "Unknown type: {} at {}: {}".format(bit, key3, val3)
                                )
                                exit(0)

                            new_val = digit[0] * mult
                            d[key1][key2][key3] = new_val


def parse_config(filename, config_type):
    """Parse a yaml configuration file for this experiment.
    Args:
            filename (str): Path to the configuration file
    Returns:
            FullConfig: Contains dataset, model, optimization, training and
            scheduling configurations
    """
    with open(filename, "r") as f:
        config_dict = _yaml.load(f, Loader=_ruamel.yaml.Loader)
        # print(config_dict) 
        convert(config_dict)
    if config_type == "hardware":
        sw_config = SWConfig(**config_dict["sw_param"])
        sch_config = SchedulingConfig(**config_dict["scheduling_param"])
        tech_config = TechConfig.from_dict(config_dict["tech_param"])
        power_config = PowerBreakdownConfig.from_dict(config_dict["power_breakdown"])
        area_config = AreaBreakdownConfig.from_dict(config_dict["area_breakdown"])
        perimeter_config = PerimeterBreakdownConfig.from_dict(
            config_dict["perimeter_breakdown"]
        )
        system_config = SystemHierarchyConfig.from_dict(config_dict["system_hierarchy"])
        memory_hierarchy_config = MemoryHierarchyConfig.from_dict(
            config_dict["memory_hierarchy"]
        )
        network_topology_config = NetworkTopologyConfig.from_dict(
            config_dict["network_topology"]
        )
        # network backend (optional)
        nb_dict = config_dict.get("network_backend", {})
        nb_model = nb_dict.get("model", "analytical")
        astra_cfg = nb_dict.get("astra", {}) if nb_model == "astra" else None
        if astra_cfg is not None:
            coll = astra_cfg.get("collectives", {})
            coll_cfg = NetworkBackendAstraCollectives(
                all_gather=coll.get("all_gather", "auto"),
                all_reduce=coll.get("all_reduce", "auto"),
                reduce_scatter=coll.get("reduce_scatter", "auto"),
                all_to_all=coll.get("all_to_all", "auto"),
            )
            nb_astra = NetworkBackendAstra(
                backend=astra_cfg.get("backend", "analytical"),
                mode=astra_cfg.get("mode", "isolated"),
                collectives=coll_cfg,
            )
        else:
            nb_astra = None
        nb = NetworkBackend(model=nb_model, astra=nb_astra)

        config = HWConfig(
            sw_config=sw_config,
            tech_config=tech_config,
            power_breakdown=power_config,
            sch_config=sch_config,
            area_breakdown=area_config,
            perimeter_breakdown=perimeter_config,
            system_config=system_config,
            memory_hierarchy=memory_hierarchy_config,
            network_topology=network_topology_config,
            network_backend=nb,
        )
    elif config_type == "LSTM":
        model_config = ModelLSTMConfig(**config_dict["model_param"])
        config = MODELConfig(model_config=model_config)
    elif config_type == "GEMM":
        mp = dict(config_dict["model_param"])  # copy
        if "backward" not in mp:
            mp["backward"] = False
        model_config = GEMMConfig(**mp)
        config = MODELConfig(model_config=model_config)
    elif config_type == "LLM":
        model_config = LLMConfig(**config_dict["model_param"])
        config = MODELConfig(model_config=model_config)
    else:
        raise ValueError("Invalid config type: {}".format(config_type))
    
    # model_config = ModelConfig(**config_dict["model_param"])
    # sw_config = SWConfig(**config_dict["sw_param"])
    # sch_config = SchedulingConfig(**config_dict["scheduling_param"])
    # tech_config = TechConfig.from_dict(config_dict["tech_param"])
    # power_config = PowerBreakdownConfig.from_dict(config_dict["power_breakdown"])
    # area_config = AreaBreakdownConfig.from_dict(config_dict["area_breakdown"])
    # perimeter_config = PerimeterBreakdownConfig.from_dict(
    #     config_dict["perimeter_breakdown"]
    # )
    # system_config = SystemHierarchyConfig.from_dict(config_dict["system_hierarchy"])
    # memory_hierarchy_config = MemoryHierarchyConfig.from_dict(
    #     config_dict["memory_hierarchy"]
    # )
    # network_topology_config = NetworkTopologyConfig.from_dict(
    #     config_dict["network_topology"]
    # )

    return config
