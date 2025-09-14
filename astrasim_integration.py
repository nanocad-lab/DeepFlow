"""
Simple AstraSim config generator from DeepFlow hardware YAML.

Note: This file embeds minimal Chakra ET generator helpers adapted from
ASTRA-sim's examples (examples/workload/microbenchmarks/generator_scripts).
Those scripts are MIT-licensed; we inline the needed logic here for convenience
to generate .et microbenchmarks (comm-only) for collectives.

Scope (M0):
- Assumes everything is on the same package (uses intra-node IB/LL).
- Topology derived from DeepFlow YAML: network_topology.intra_node ∈ {fc, ring}
  maps to AstraSim {FullyConnected, Ring}.
- npus_count pulled from DeepFlow DP (scheduling_param.dp).
- Chooses collective algorithms per network_backend.astra.collectives with
  an 'auto' policy: FullyConnected -> direct for AG/A2A; ring for AR/RS.

No invocation of AstraSim here; just emit configs and comm-only ETs.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Any, Tuple, List, Optional

import ruamel.yaml as _yaml
from hw_component import Network  # to compute intra/inter throughput/latency

# Chakra ET dependencies (vendored in ASTRA-sim extern).
# Add their paths explicitly since they are inside 'astra-sim/extern/...'
import sys
BASE_DIR = os.path.dirname(__file__)
CHAKRA_PB_DIR = os.path.join(BASE_DIR, 'astra-sim', 'extern', 'graph_frontend', 'chakra', 'schema', 'protobuf')
CHAKRA_UTILS_DIR = os.path.join(BASE_DIR, 'astra-sim', 'extern', 'graph_frontend', 'chakra', 'src', 'third_party', 'utils')
sys.path.insert(0, CHAKRA_PB_DIR)
sys.path.insert(0, CHAKRA_UTILS_DIR)
import et_def_pb2 as pb  # type: ignore
from protolib import encodeMessage as chakra_encode  # type: ignore


def _save_yaml(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        _yaml.dump(data, f, Dumper=_yaml.RoundTripDumper)


def _save_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _gbps_from_bps(bps: float) -> float:
    """Convert bytes/second to GB/s (2^30 base)."""
    return float(bps) / float(1 << 30)


def _ns_from_s(sec: float) -> float:
    return float(sec) * 1e9


def _choose_collective(alg: str, topo: str, op: str) -> str:
    """
    Resolve 'auto' choice for collectives.
    - topo: 'FullyConnected' | 'Ring'
    - op: 'all-gather' | 'all-reduce' | 'reduce-scatter' | 'all-to-all'
    Policy: FullyConnected => direct for AG/A2A; ring for AR/RS. Ring => ring for all.
    """
    if alg != "auto":
        return alg
    if topo == "FullyConnected":
        if op in ("all-gather", "all-to-all"):
            return "direct"
        return "ring"
    # default for ring/other
    return "ring"


def compute_intra_inter_ib_ll_from_hw(hw_obj) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Same as above but accepts a pre-parsed HWConfig object."""
    net = Network(hw_obj)
    intra_throughput, inter_throughput = net.calcThroughput()
    intra_latency, inter_latency = net.calcLatency()
    return (intra_throughput, intra_latency), (inter_throughput, inter_latency)


def _derive_topology_from_yaml(cfg: Dict[str, Any]) -> str:
    """Map DeepFlow network_topology.intra_node string to AstraSim topology name."""
    topo_str = str(cfg.get("network_topology", {}).get("intra_node", "")).lower()
    if topo_str in ("fc", "fullyconnected", "fully_connected", "fully-connected"):
        return "FullyConnected"
    if topo_str in ("ring",):
        return "Ring"
    # Fallback: assume FullyConnected as M0 default
    return "FullyConnected"


def generate_astrasim_network_yaml(
    hw_cfg_or_obj,
    output_path: str,
    npus_count: int,
    astra_config_dir: str = "./astra_cache",
) -> Dict[str, Any]:
    """
    Emit Analytical network YAML using intra-node IB/LL and FullyConnected topo.
    npus_count is taken from scheduling_param.dp.
    """
    cfg = _load_yaml(hw_cfg_or_obj) if isinstance(hw_cfg_or_obj, str) else None
    (intra_ib_bps, intra_ll_s), _ = (
        compute_intra_inter_ib_ll(hw_cfg_or_obj)
        if isinstance(hw_cfg_or_obj, str)
        else compute_intra_inter_ib_ll_from_hw(hw_cfg_or_obj)
    )

    if not intra_ib_bps or intra_ib_bps <= 0:
        raise ValueError("Intra-node bandwidth computed as 0. Check hardware config/network model.")
    if not intra_ll_s or intra_ll_s <= 0:
        raise ValueError("Intra-node latency computed as 0. Check hardware config/network model.")

    ib_gbps = round(_gbps_from_bps(intra_ib_bps), 6)
    ll_ns = round(_ns_from_s(intra_ll_s), 3)

    topo = _derive_topology_from_yaml(cfg) if cfg is not None else (
        "FullyConnected" if getattr(hw_cfg_or_obj.network_topology.intra, "topology", "fc").lower() in ("fc", "fullyconnected") else "Ring"
    )

    # Write in the same flow style as AstraSim examples
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(f"topology: [ {topo} ]\n")
        f.write(f"npus_count: [ {int(npus_count)} ]\n")
        f.write(f"bandwidth: [ {ib_gbps} ]  # GB/s\n")
        f.write(f"latency: [ {ll_ns} ]   # ns\n")

    return {"topology": [topo], "npus_count": [int(npus_count)], "bandwidth": [ib_gbps], "latency": [ll_ns]}


def generate_astrasim_system_json(
    hw_obj,
    output_path: str,
    topo: str = "FullyConnected",
    astra_config_dir: str = "./astra_cache",
) -> Dict[str, Any]:
    """Emit system JSON selecting algorithms per network_backend.astra.collectives."""
    if isinstance(hw_obj, str):
        raise TypeError("generate_astrasim_system_json expects a parsed HWConfig object, not a path")
    nb = getattr(hw_obj, "network_backend", None)
    if nb and nb.astra:
        coll = nb.astra.collectives._asdict()
    else:
        coll = {}

    ag = _choose_collective(str(coll.get("all_gather", "auto")), topo, "all-gather")
    ar = _choose_collective(str(coll.get("all_reduce", "auto")), topo, "all-reduce")
    rs = _choose_collective(str(coll.get("reduce_scatter", "auto")), topo, "reduce-scatter")
    a2a = _choose_collective(str(coll.get("all_to_all", "auto")), topo, "all-to-all")

    # Minimal system JSON leveraging native collectives
    system = {
        "scheduling-policy": "LIFO",
        "endpoint-delay": 10,
        "active-chunks-per-dimension": 1,
        "preferred-dataset-splits": 4,
        "all-reduce-implementation": [ar],
        "all-gather-implementation": [ag],
        "reduce-scatter-implementation": [rs],
        "all-to-all-implementation": [a2a],
        "collective-optimization": "localBWAware",
        "local-mem-bw": 1600,
        "boost-mode": 0,
        "roofline-enabled": 0,
        "peak-perf": 900,
    }
    _save_json(output_path, system)
    return system


def generate_astrasim_configs(hw_obj, out_dir: str = "./astra_cache", npus_count: Optional[int] = None) -> Dict[str, str]:
    """
    Generate Analytical AstraSim configs under out_dir.
    Returns dict with file paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    net_yaml = os.path.join(out_dir, "network_analytical.yml")
    sys_json = os.path.join(out_dir, "system_native_collectives.json")

    if npus_count is None:
        raise ValueError("npus_count must be provided explicitly when generating AstraSim configs.")
    net = generate_astrasim_network_yaml(hw_obj, net_yaml, npus_count=npus_count, astra_config_dir=out_dir)
    topo = net["topology"][0]
    generate_astrasim_system_json(hw_obj, sys_json, topo=topo, astra_config_dir=out_dir)

    return {"network_yaml": net_yaml, "system_json": sys_json}

def generate_astrasim_configs_from_hw(hw_obj, out_dir: str = "./astra_cache", npus_count: Optional[int] = None) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    net_yaml = os.path.join(out_dir, "network_analytical.yml")
    sys_json = os.path.join(out_dir, "system_native_collectives.json")

    # Build topology from parsed object
    topo = "FullyConnected" if getattr(hw_obj.network_topology.intra, "topology", "fc").lower() in ("fc", "fullyconnected") else "Ring"
    if npus_count is None:
        raise ValueError("npus_count must be provided explicitly when generating AstraSim configs.")

    # IB/LL from parsed object
    (intra_ib_bps, intra_ll_s), _ = compute_intra_inter_ib_ll_from_hw(hw_obj)
    if not intra_ib_bps or intra_ib_bps <= 0:
        raise ValueError("Intra-node bandwidth computed as 0. Check hardware config/network model.")
    if not intra_ll_s or intra_ll_s <= 0:
        raise ValueError("Intra-node latency computed as 0. Check hardware config/network model.")
    ib_gbps = round(_gbps_from_bps(intra_ib_bps), 6)
    ll_ns = round(_ns_from_s(intra_ll_s), 3)

    # Write network YAML
    with open(net_yaml, "w") as f:
        f.write(f"topology: [ {topo} ]\n")
        f.write(f"npus_count: [ {int(npus_count)} ]\n")
        f.write(f"bandwidth: [ {ib_gbps} ]  # GB/s\n")
        f.write(f"latency: [ {ll_ns} ]   # ns\n")

    # System JSON collectives from parsed network_backend
    nb = getattr(hw_obj, "network_backend", None)
    if nb and nb.astra:
        coll = nb.astra.collectives
        ag = _choose_collective(coll.all_gather, topo, "all-gather")
        ar = _choose_collective(coll.all_reduce, topo, "all-reduce")
        rs = _choose_collective(coll.reduce_scatter, topo, "reduce-scatter")
        a2a = _choose_collective(coll.all_to_all, topo, "all-to-all")
    else:
        ag = _choose_collective("auto", topo, "all-gather")
        ar = _choose_collective("auto", topo, "all-reduce")
        rs = _choose_collective("auto", topo, "reduce-scatter")
        a2a = _choose_collective("auto", topo, "all-to-all")

    system = {
        "scheduling-policy": "LIFO",
        "endpoint-delay": 10,
        "active-chunks-per-dimension": 1,
        "preferred-dataset-splits": 4,
        "all-reduce-implementation": [ar],
        "all-gather-implementation": [ag],
        "reduce-scatter-implementation": [rs],
        "all-to-all-implementation": [a2a],
        "collective-optimization": "localBWAware",
        "local-mem-bw": 1600,
        "boost-mode": 0,
        "roofline-enabled": 0,
        "peak-perf": 900,
    }
    _save_json(sys_json, system)

    return {"network_yaml": net_yaml, "system_json": sys_json}


# -----------------------------
# Chakra ET helpers (comm-only)
# -----------------------------

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_et_node(fh, node: pb.Node) -> None:
    chakra_encode(fh, node)


def _new_comm_node(node_id: int, name: str, coll_type: int, size_bytes: int) -> pb.Node:
    n = pb.Node()
    n.id = node_id
    n.name = name
    n.type = pb.COMM_COLL_NODE
    n.attr.append(pb.AttributeProto(name="is_cpu_op", bool_val=False))
    n.attr.append(pb.AttributeProto(name="comm_type", int64_val=coll_type))
    n.attr.append(pb.AttributeProto(name="comm_size", int64_val=int(size_bytes)))
    return n


def _write_comm_microbenchmark(prefix_path: str, npus_count: int, coll_type: int, size_bytes: int) -> str:
    """Create per-rank ET files under prefix_path.{rank}.et and return prefix_path."""
    # Ensure parent dir exists
    _ensure_dir(os.path.dirname(prefix_path))
    node_id = 0
    for rank in range(npus_count):
        et_path = f"{prefix_path}.{rank}.et"
        with open(et_path, "wb") as fh:
            # Global metadata
            chakra_encode(fh, pb.GlobalMetadata(version="0.0.4"))
            # Single COMM_COLL node
            name = os.path.basename(prefix_path)
            node = _new_comm_node(node_id, name, coll_type, size_bytes)
            write_et_node(fh, node)
            node_id += 1
    return prefix_path


def generate_workload_et(comm: str, npus_count: int, size_bytes: int, astra_config_dir: str = "./astra_cache") -> str:
    """
    Generate Chakra ETs for a single collective microbenchmark.
    Returns the workload prefix path (no extension) to use with AstraSim.
    """
    comm = comm.lower()
    coll_map = {
        "all_gather": pb.ALL_GATHER,
        "allreduce": pb.ALL_REDUCE,
        "all_reduce": pb.ALL_REDUCE,
        "reduce_scatter": pb.REDUCE_SCATTER,
        "all_to_all": pb.ALL_TO_ALL,
        "alltoall": pb.ALL_TO_ALL,
    }
    if comm not in coll_map:
        raise ValueError(f"Unsupported comm type: {comm}")

    # Create a simple directory structure similar to examples
    size_mb = max(1, int(round(size_bytes / (1024 * 1024))))
    base_dir = os.path.join(astra_config_dir, "workload", comm, f"{npus_count}npus_{size_mb}MB")
    prefix = os.path.join(base_dir, comm)
    return _write_comm_microbenchmark(prefix, npus_count, coll_map[comm], size_bytes)


def get_remote_memory_path() -> str:
    """Return the relative path to the no_memory_expansion.json in repo."""
    return os.path.join(
        os.path.dirname(__file__),
        "astra-sim",
        "examples",
        "remote_memory",
        "analytical",
        "no_memory_expansion.json",
    )


# -----------------------------
# Cache + Execute wrapper
# -----------------------------

def _collectives_from_hw(hw_obj, topo: str) -> Dict[str, str]:
    nb = getattr(hw_obj, "network_backend", None)
    if nb and nb.astra:
        coll = nb.astra.collectives
        return {
            "all_gather": _choose_collective(coll.all_gather, topo, "all-gather"),
            "all_reduce": _choose_collective(coll.all_reduce, topo, "all-reduce"),
            "reduce_scatter": _choose_collective(coll.reduce_scatter, topo, "reduce-scatter"),
            "all_to_all": _choose_collective(coll.all_to_all, topo, "all-to-all"),
        }
    # Defaults
    return {
        "all_gather": _choose_collective("auto", topo, "all-gather"),
        "all_reduce": _choose_collective("auto", topo, "all-reduce"),
        "reduce_scatter": _choose_collective("auto", topo, "reduce-scatter"),
        "all_to_all": _choose_collective("auto", topo, "all-to-all"),
    }


def _canonical_sig(sig: Dict[str, Any]) -> str:
    """Return a canonical JSON string for use as a human-readable cache key."""
    return json.dumps(sig, sort_keys=True, separators=(",", ":"))


def _load_cache(cache_path: str) -> Dict[str, Any]:
    if not os.path.exists(cache_path):
        return {}
    try:
        with open(cache_path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_cache(cache_path: str, cache_obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    tmp_path = cache_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(cache_obj, f, indent=2)
    os.replace(tmp_path, cache_path)


def run_cache_astrasim(
    hw_obj,
    comm: str,
    npus_count: int,
    size_bytes: int,
    astra_config_dir: str = "./astra_cache",
    cache_path: str = "./astra_cache/cache.json",
) -> Tuple[List[float], float]:
    """
    Cached AstraSim run: generates configs (skips existing workload),
    checks cache, executes if needed, then stores results.
    Returns (per_node_sec, max_sec).
    """
    if isinstance(hw_obj, str):
        raise TypeError("run_cache_astrasim expects a parsed HWConfig object, not a path")

    # Compute network params and topology from HW
    (intra_ib_bps, intra_ll_s), _ = compute_intra_inter_ib_ll_from_hw(hw_obj)
    if not intra_ib_bps or intra_ib_bps <= 0 or not intra_ll_s or intra_ll_s <= 0:
        raise ValueError("Invalid intra-node IB/LL computed from HW config")
    ib_gbps = round(_gbps_from_bps(intra_ib_bps), 6)
    ll_ns = round(_ns_from_s(intra_ll_s), 3)
    topo = _derive_topology_from_hw(hw_obj)
    colls = _collectives_from_hw(hw_obj, topo)

    # Build signature and check cache
    sig = {
        "comm": comm.lower(),
        "npus": int(npus_count),
        "size_bytes": int(size_bytes),
        "topology": topo,
        "ib_gbps": ib_gbps,
        "ll_ns": ll_ns,
        "collectives": colls,
        "backend": "analytical",
    }
    key = _canonical_sig(sig)
    cache = _load_cache(cache_path)
    if key in cache:
        entry = cache[key]
        return entry.get("per_node_sec", []), float(entry.get("max_sec", 0.0))

    # Generate configs
    files = generate_astrasim_configs_from_hw(hw_obj, out_dir=astra_config_dir, npus_count=npus_count)

    # Ensure workload ET exists (skip writing if all ranks present)
    size_mb = max(1, int(round(size_bytes / (1024 * 1024))))
    base_dir = os.path.join(astra_config_dir, "workload", comm.lower(), f"{npus_count}npus_{size_mb}MB")
    prefix = os.path.join(base_dir, comm.lower())
    expected = [f"{prefix}.{r}.et" for r in range(npus_count)]
    if not all(os.path.exists(p) for p in expected):
        generate_workload_et(comm, npus_count, size_bytes, astra_config_dir=astra_config_dir)

    # Execute AstraSim
    per_node_sec, max_sec = run_astrasim_analytical(
        workload_prefix=prefix,
        system_json=files["system_json"],
        network_yaml=files["network_yaml"],
        remote_memory_json=get_remote_memory_path(),
    )

    # Update cache
    cache[key] = {
        "signature": sig,
        "per_node_sec": per_node_sec,
        "max_sec": max_sec,
        "workload_prefix": prefix,
        "system_json": files["system_json"],
        "network_yaml": files["network_yaml"],
    }
    _save_cache(cache_path, cache)

    return per_node_sec, max_sec

def generate_astrasim_run_files(
    hw_obj,
    comm: str,
    npus_count: int,
    size_bytes: int,
    astra_config_dir: str = "./astra_cache",
) -> Dict[str, str]:
    """
    Generate all 4 config paths needed by AstraSim Analytical backend:
    - workload prefix (.et files written next to it)
    - system JSON
    - network YAML
    - remote memory JSON (existing fixed relative path)
    Returns dict of paths: workload_prefix, system_json, network_yaml, remote_memory_json
    """
    if isinstance(hw_obj, str):
        raise TypeError("generate_astrasim_run_files expects a parsed HWConfig object, not a path")
    files = generate_astrasim_configs(hw_obj, out_dir=astra_config_dir, npus_count=npus_count)
    workload_prefix = generate_workload_et(comm, npus_count, size_bytes, astra_config_dir=astra_config_dir)
    files.update(
        {
            "workload_prefix": workload_prefix,
            "remote_memory_json": get_remote_memory_path(),
        }
    )
    return files


# -----------------------------
# AstraSim runner (analytical)
# -----------------------------

import subprocess
import re


def _astrasim_binary_path() -> str:
    return os.path.join(
        os.path.dirname(__file__),
        "astra-sim",
        "build",
        "astra_analytical",
        "build",
        "bin",
        "AstraSim_Analytical_Congestion_Aware",
    )


def run_astrasim_analytical(
    workload_prefix: str,
    system_json: str,
    network_yaml: str,
    remote_memory_json: str,
    binary_path: Optional[str] = None,
) -> Tuple[List[float], float]:
    """
    Execute AstraSim Analytical (congestion-aware) with the given configs.
    Returns (per_node_times_sec, max_time_sec).
    """
    bin_path = binary_path or _astrasim_binary_path()
    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"AstraSim binary not found at {bin_path}")

    cmd = [
        bin_path,
        f"--workload-configuration={workload_prefix}",
        f"--system-configuration={system_json}",
        f"--remote-memory-configuration={remote_memory_json}",
        f"--network-configuration={network_yaml}",
    ]

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    out = proc.stdout

    # Parse per-node wall times: lines like 'sys[0], Wall time: 41560'
    times_cycles = []
    for line in out.splitlines():
        m = re.search(r"sys\[(\d+)\],\s*Wall time:\s*(\d+)", line)
        if m:
            t_cycles = int(m.group(2))
            # 1 cycle = 1 ns
            times_cycles.append(t_cycles)

    if not times_cycles:
        # Fallback: look for 'finished, X cycles' if Wall time not found
        for line in out.splitlines():
            m = re.search(r"sys\[(\d+)\] finished,\s*(\d+) cycles", line)
            if m:
                t_cycles = int(m.group(2))
                times_cycles.append(t_cycles)

    per_node_sec = [t * 1e-9 for t in times_cycles]
    max_sec = max(per_node_sec) if per_node_sec else 0.0
    return per_node_sec, max_sec


if __name__ == "__main__":
    # Quick sample test: 1 MiB All-Gather across 4 nodes
    from config import parse_config
    hw_cfg_path = os.path.join("configs", "hardware-config", "a100_80GB.yaml")
    hw_obj = parse_config(hw_cfg_path, "hardware")
    out_dir = "./astra_cache"
    npus = 4
    size_bytes = 1 << 20
    comm = "all_gather"

    files = generate_astrasim_run_files(hw_obj, comm, npus, size_bytes, astra_config_dir=out_dir)
    per_node_sec, max_sec = run_astrasim_analytical(
        workload_prefix=files["workload_prefix"],
        system_json=files["system_json"],
        network_yaml=files["network_yaml"],
        remote_memory_json=files["remote_memory_json"],
    )
    print("AstraSim per-node times (s):", per_node_sec)
    print("AstraSim max time (s):", max_sec)
