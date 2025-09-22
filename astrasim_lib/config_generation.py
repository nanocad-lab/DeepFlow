"""Generate AstraSim configuration artifacts from DeepFlow hardware configs."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

from hw_component import Network

from .bootstrap import ensure_chakra_available

# Ensure Chakra dependencies are importable for downstream modules that rely on
# protobuf definitions. This module itself does not import them but provides the
# same setup entry point for consistency.
ensure_chakra_available()

ASTRA_DEBUG = False

_NET_YAML_CACHE: set[tuple[str, int, float, float, str]] = set()
_JSON_WRITTEN_BY_NPUS: set[object] = set()


def _save_json(path: str, data: Dict[str, Any], npus_key: Optional[int] = None) -> None:
    """Write ``data`` to ``path`` once per ``npus_key`` per process."""
    key = npus_key if npus_key is not None else path
    if key in _JSON_WRITTEN_BY_NPUS:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        import json as _json

        _json.dump(data, handle, indent=2)
    os.replace(tmp_path, path)
    _JSON_WRITTEN_BY_NPUS.add(key)


def _gbps_from_bps(bps: float) -> float:
    return float(bps) / float(1 << 30)


def _ns_from_s(sec: float) -> float:
    return float(sec) * 1e9


def choose_collective(alg: str, topo: str, op: str) -> str:
    """Resolve ``auto`` policies for collective algorithms."""
    if alg != "auto":
        return alg
    if topo == "FullyConnected":
        return "direct"
    if topo == "Switch":
        return "halvingDoubling"
    return "ring"


def compute_intra_inter_ib_ll_from_hw(hw_obj) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Return intra/inter bandwidth+latency tuples from a parsed DeepFlow config."""
    net = Network(hw_obj)
    intra_throughput, inter_throughput = net.calcThroughput()
    intra_latency, inter_latency = net.calcLatency()
    return (intra_throughput, intra_latency), (inter_throughput, inter_latency)


def derive_topology_from_hw(hw_obj) -> str:
    """Map DeepFlow network topology enums to AstraSim names."""
    try:
        topo = getattr(hw_obj.network_topology.intra, "topology", None)
        topo_str = (topo or "fc").lower()
        if topo_str in ("fc", "fullyconnected", "fully_connected", "fully-connected"):
            return "FullyConnected"
        if topo_str in ("ring",):
            return "Ring"
        if topo_str in ("switch",):
            return "Switch"
    except Exception:  # pragma: no cover - defensive
        pass
    return "FullyConnected"


def generate_astrasim_configs_from_hw(
    hw_obj,
    out_dir: str = "./astra_cache",
    npus_count: Optional[int] = None,
) -> Dict[str, str]:
    """Write AstraSim network/system configs derived from ``hw_obj``."""
    if npus_count is None:
        raise ValueError("npus_count must be provided explicitly when generating AstraSim configs.")

    net_yaml = os.path.join(out_dir, f"network_analytical_{int(npus_count)}.yml")
    sys_json = os.path.join(out_dir, "system_native_collectives.json")

    topo = derive_topology_from_hw(hw_obj)
    if topo == "Ring" and int(npus_count) <= 2:
        topo = "FullyConnected"

    (intra_ib_bps, intra_ll_s), _ = compute_intra_inter_ib_ll_from_hw(hw_obj)
    if not intra_ib_bps or intra_ib_bps <= 0:
        raise ValueError("Intra-node bandwidth computed as 0. Check hardware config/network model.")
    if not intra_ll_s or intra_ll_s <= 0:
        raise ValueError("Intra-node latency computed as 0. Check hardware config/network model.")
    ib_gbps = round(_gbps_from_bps(intra_ib_bps), 6)
    ll_ns = round(_ns_from_s(intra_ll_s), 3)

    net_content = (
        f"topology: [ {topo} ]\n"
        f"npus_count: [ {int(npus_count)} ]\n"
        f"bandwidth: [ {ib_gbps} ]  # GB/s\n"
        f"latency: [ {ll_ns} ]   # ns\n"
    )

    os.makedirs(os.path.dirname(net_yaml), exist_ok=True)
    cache_key = (net_yaml, int(npus_count), ib_gbps, ll_ns, topo)
    need_write = True
    if cache_key in _NET_YAML_CACHE:
        need_write = False
    elif os.path.exists(net_yaml):
        try:
            with open(net_yaml, "r", encoding="utf-8") as handle:
                existing = handle.read()
            if existing == net_content:
                need_write = False
        except Exception:  # pragma: no cover - defensive
            pass
    if need_write:
        tmp_path = net_yaml + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            handle.write(net_content)
        os.replace(tmp_path, net_yaml)
    _NET_YAML_CACHE.add(cache_key)

    exec_backend = getattr(hw_obj, "execution_backend", None)
    if exec_backend and exec_backend.astra:
        coll = exec_backend.astra.collectives
        sys_opts = getattr(exec_backend.astra, "sys_options", None)
        ag = choose_collective(coll.all_gather, topo, "all-gather")
        ar = choose_collective(coll.all_reduce, topo, "all-reduce")
        rs = choose_collective(coll.reduce_scatter, topo, "reduce-scatter")
        a2a = choose_collective(coll.all_to_all, topo, "all-to-all")
    else:
        ag = choose_collective("auto", topo, "all-gather")
        ar = choose_collective("auto", topo, "all-reduce")
        rs = choose_collective("auto", topo, "reduce-scatter")
        a2a = choose_collective("auto", topo, "all-to-all")
        sys_opts = None

    system = {
        "scheduling-policy": "LIFO",
        "endpoint-delay": 10,
        "active-chunks-per-dimension": 1,
        "preferred-dataset-splits": 1,
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
    if sys_opts is not None:
        if getattr(sys_opts, "endpoint_delay", None) is not None:
            system["endpoint-delay"] = sys_opts.endpoint_delay
        if getattr(sys_opts, "active_chunks_per_dimension", None) is not None:
            system["active-chunks-per-dimension"] = sys_opts.active_chunks_per_dimension
        if getattr(sys_opts, "preferred_dataset_splits", None) is not None:
            system["preferred-dataset-splits"] = sys_opts.preferred_dataset_splits

    _save_json(sys_json, system, npus_key=int(npus_count))

    return {"network_yaml": net_yaml, "system_json": sys_json}


__all__ = [
    "ASTRA_DEBUG",
    "choose_collective",
    "compute_intra_inter_ib_ll_from_hw",
    "derive_topology_from_hw",
    "generate_astrasim_configs_from_hw",
]
