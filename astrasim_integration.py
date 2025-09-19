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
- Chooses collective algorithms per execution_backend.astra.collectives with
  an 'auto' policy: FullyConnected -> direct for AG/A2A; ring for AR/RS.

No invocation of AstraSim here; just emit configs and comm-only ETs.
"""

import json
import os
from typing import Dict, Any, Tuple, List, Optional
import hashlib
import fcntl
import time
from contextlib import contextmanager

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
    - topo: 'FullyConnected' | 'Ring' | 'Switch'
    - op: 'all-gather' | 'all-reduce' | 'reduce-scatter' | 'all-to-all'
    Policy:
      * FullyConnected => direct for all collectives
      * Ring           => ring for all
      * Switch         => halvingDoubling for all
    """
    if alg != "auto":
        return alg
    if topo == "FullyConnected":
        return "direct"
    if topo == "Switch":
        return "halvingDoubling"
    # default for ring/other
    return "ring"


def compute_intra_inter_ib_ll_from_hw(hw_obj) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Same as above but accepts a pre-parsed HWConfig object."""
    net = Network(hw_obj)
    intra_throughput, inter_throughput = net.calcThroughput()
    intra_latency, inter_latency = net.calcLatency()
    return (intra_throughput, intra_latency), (inter_throughput, inter_latency)


def _derive_topology_from_hw(hw_obj) -> str:
    """Map DeepFlow parsed HWConfig to AstraSim topology name."""
    try:
        topo = getattr(hw_obj.network_topology.intra, "topology", None)
        topo_str = (topo or "fc").lower()
        if topo_str in ("fc", "fullyconnected", "fully_connected", "fully-connected"):
            return "FullyConnected"
        if topo_str in ("ring",):
            return "Ring"
        if topo_str in ("switch",):
            return "Switch"
    except Exception:
        pass
    return "FullyConnected"


    # Unused config-generation helpers removed; see generate_astrasim_configs_from_hw

def generate_astrasim_configs_from_hw(hw_obj, out_dir: str = "./astra_cache", npus_count: Optional[int] = None) -> Dict[str, str]:
    
    os.makedirs(out_dir, exist_ok=True)
    # Use per-npus network file to avoid cross-run clobbering
    net_yaml = os.path.join(out_dir, f"network_analytical_{int(npus_count)}.yml")
    sys_json = os.path.join(out_dir, "system_native_collectives.json")

    # Build topology from parsed object, supporting fc/ring/switch
    intratopo = getattr(hw_obj.network_topology.intra, "topology", "fc")
    topo_str = (intratopo or "fc").lower()
    if topo_str in ("fc", "fullyconnected", "fully_connected", "fully-connected"):
        topo = "FullyConnected"
    elif topo_str in ("ring",):
        topo = "Ring"
    elif topo_str in ("switch",):
        topo = "Switch"
    else:
        topo = "FullyConnected"
    # Special-case: AstraSim analytical Ring crashes for npus_count <= 2 → fall back to FC
    if topo == "Ring" and int(npus_count) <= 2:
        topo = "FullyConnected"
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

    # Write network YAML under a file lock for this specific path
    net_content = (
        f"topology: [ {topo} ]\n"
        f"npus_count: [ {int(npus_count)} ]\n"
        f"bandwidth: [ {ib_gbps} ]  # GB/s\n"
        f"latency: [ {ll_ns} ]   # ns\n"
    )
    os.makedirs(os.path.dirname(net_yaml), exist_ok=True)
    with _cache_file_lock(net_yaml, timeout_s=5.0, poll_s=0.05):
        tmp_path = net_yaml + ".tmp"
        with open(tmp_path, "w") as f:
            f.write(net_content)
        os.replace(tmp_path, net_yaml)

    # System JSON collectives from parsed execution backend
    exec_backend = getattr(hw_obj, "execution_backend", None)
    if exec_backend and exec_backend.astra:
        coll = exec_backend.astra.collectives
        sys_opts = getattr(exec_backend.astra, "sys_options", None)
        ag = _choose_collective(coll.all_gather, topo, "all-gather")
        ar = _choose_collective(coll.all_reduce, topo, "all-reduce")
        rs = _choose_collective(coll.reduce_scatter, topo, "reduce-scatter")
        a2a = _choose_collective(coll.all_to_all, topo, "all-to-all")
    else:
        ag = _choose_collective("auto", topo, "all-gather")
        ar = _choose_collective("auto", topo, "all-reduce")
        rs = _choose_collective("auto", topo, "reduce-scatter")
        a2a = _choose_collective("auto", topo, "all-to-all")
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
            system[
                "active-chunks-per-dimension"
            ] = sys_opts.active_chunks_per_dimension
        if getattr(sys_opts, "preferred_dataset_splits", None) is not None:
            system[
                "preferred-dataset-splits"
            ] = sys_opts.preferred_dataset_splits
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


def _new_send_node(node_id: int, name: str, size_bytes: int, dst_rank: int, tag: int) -> pb.Node:
    n = pb.Node()
    n.id = node_id
    n.name = name
    n.type = pb.COMM_SEND_NODE
    n.attr.append(pb.AttributeProto(name="comm_size", int64_val=int(size_bytes)))
    n.attr.append(pb.AttributeProto(name="comm_dst", int32_val=dst_rank))
    n.attr.append(pb.AttributeProto(name="comm_tag", int32_val=tag))
    return n


def _new_recv_node(node_id: int, name: str, size_bytes: int, src_rank: int, tag: int) -> pb.Node:
    n = pb.Node()
    n.id = node_id
    n.name = name
    n.type = pb.COMM_RECV_NODE
    n.attr.append(pb.AttributeProto(name="comm_size", int64_val=int(size_bytes)))
    n.attr.append(pb.AttributeProto(name="comm_src", int32_val=src_rank))
    n.attr.append(pb.AttributeProto(name="comm_tag", int32_val=tag))
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


def _write_point_to_point_microbenchmark(prefix_path: str, size_bytes: int) -> str:
    """Create point-to-point ET files: sender (rank 0) and receiver (rank 1)."""
    # Ensure parent dir exists
    _ensure_dir(os.path.dirname(prefix_path))

    # Sender (rank 0)
    et_path = f"{prefix_path}.0.et"
    with open(et_path, "wb") as fh:
        chakra_encode(fh, pb.GlobalMetadata(version="0.0.4"))
        node = _new_send_node(0, "pipeline_send", size_bytes, dst_rank=1, tag=0)
        write_et_node(fh, node)

    # Receiver (rank 1)
    et_path = f"{prefix_path}.1.et"
    with open(et_path, "wb") as fh:
        chakra_encode(fh, pb.GlobalMetadata(version="0.0.4"))
        node = _new_recv_node(0, "pipeline_recv", size_bytes, src_rank=0, tag=0)
        write_et_node(fh, node)

    return prefix_path


def _size_label(size_bytes: int) -> str:
    """Return a human label for size in KB..TB with 2 decimals (base 1024)."""
    units = [
        ("TB", 1024 ** 4),
        ("GB", 1024 ** 3),
        ("MB", 1024 ** 2),
        ("KB", 1024 ** 1),
    ]
    for suffix, div in units:
        if size_bytes >= div:
            val = size_bytes / div
            return f"{val:.2f}{suffix}"
    # Smaller than 1KB: express as fractional KB
    val = size_bytes / 1024.0
    return f"{val:.4f}KB"


def generate_workload_et(comm: str, npus_count: int, size_bytes: int, astra_config_dir: str = "./astra_cache") -> str:
    """
    Generate Chakra ETs for a single collective microbenchmark or point-to-point.
    Returns the workload prefix path (no extension) to use with AstraSim.
    """
    comm = comm.lower()

    # Directory and file prefix include human-readable size label with suffix
    label = _size_label(size_bytes)
    base_dir = os.path.join(astra_config_dir, "workload", comm, f"{npus_count}npus_{label}")
    prefix = os.path.join(base_dir, f"{comm}_{label}")

    # Handle point-to-point (pipeline)
    if comm == "pipeline":
        return _write_point_to_point_microbenchmark(prefix, size_bytes)

    # Handle collectives
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
    exec_backend = getattr(hw_obj, "execution_backend", None)
    if exec_backend and exec_backend.astra:
        coll = exec_backend.astra.collectives
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


def _hash_sig(canonical: str) -> str:
    """Return a sha256 hex digest for the canonical signature string."""
    h = hashlib.sha256()
    h.update(canonical.encode("utf-8"))
    return h.hexdigest()


def _hash_file_bundle(paths: List[str]) -> str:
    """Return a sha256 hex digest for the concatenated contents of the given files."""
    h = hashlib.sha256()
    for path in sorted(set(paths)):
        if not os.path.exists(path):
            continue
        with open(path, "rb") as fh:
            while True:
                chunk = fh.read(1024 * 1024)
                if not chunk:
                    break
                h.update(chunk)
    return h.hexdigest()


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


@contextmanager
def _cache_file_lock(cache_path: str, timeout_s: float = 30.0, poll_s: float = 0.1):
    """Best-effort process-wide lock using a sidecar .lock file (POSIX flock).
    Attempts to acquire an exclusive non-blocking lock, polling until timeout.
    Yields a boolean indicating whether the lock was acquired. On timeout,
    proceeds without a lock to avoid deadlock.
    """
    lock_path = cache_path + ".lock"
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    lf = open(lock_path, "a+")
    acquired = False
    start = time.time()
    try:
        while True:
            try:
                fcntl.flock(lf.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except BlockingIOError:
                if (time.time() - start) >= timeout_s:
                    acquired = False
                    break
                time.sleep(poll_s)
        yield acquired
    finally:
        if acquired:
            try:
                fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
        try:
            lf.close()
        except Exception:
            pass


def force_unlock_cache(cache_path: str = "./astra_cache/cache.json") -> None:
    """Best-effort to clear any stale lock artifacts for the cache.
    This removes the sidecar .lock file and attempts to unlock if possible.
    Safe to call when no lock is held.
    """
    lock_path = cache_path + ".lock"
    try:
        # Attempt unlock if we can open it
        if os.path.exists(lock_path):
            with open(lock_path, "a+") as lf:
                try:
                    fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass
    except Exception:
        pass
    try:
        if os.path.exists(lock_path):
            os.remove(lock_path)
    except Exception:
        pass


def ensure_cache_file_exists(cache_path: str = "./astra_cache/cache.json") -> None:
    """Create an empty JSON cache file if it does not exist."""
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        if not os.path.exists(cache_path):
            with open(cache_path, "w") as f:
                json.dump({}, f)
    except Exception:
        pass


def ensure_cache_unlocked_if_standalone(cache_path: str = "./astra_cache/cache.json") -> None:
    """If not running under astra_test harness, clear any stale cache lock.
    Checked via env var 'ASTRA_TEST'.
    """
    if not os.environ.get("ASTRA_TEST"):
        force_unlock_cache(cache_path)
        ensure_cache_file_exists(cache_path)


def run_cache_astrasim(
    hw_obj,
    comm: str,
    npus_count: int,
    size_bytes: int,
    astra_config_dir: str = "./astra_cache",
    cache_path: str = "./astra_cache/cache.json",
    bundle_paths: Optional[List[str]] = None,
) -> Tuple[List[float], float]:
    """
    Cached AstraSim run: generates configs (skips existing workload),
    checks cache, executes if needed, then stores results.
    Returns (per_node_sec, max_sec).
    """
    if isinstance(hw_obj, str):
        raise TypeError("run_cache_astrasim expects a parsed HWConfig object, not a path")

    # Allow per-run override of cache/config dir for concurrency (used by astra_test)
    override_dir = os.environ.get("ASTRA_CACHE_DIR")
    if override_dir:
        astra_config_dir = override_dir
        cache_path = os.path.join(override_dir, "cache.json")
        ensure_cache_file_exists(cache_path)

    debug_cache = bool(os.environ.get("ASTRA_CACHE_DEBUG"))
    def _cache_dbg(msg: str) -> None:
        if not debug_cache:
            return
        try:
            base_dir = os.path.dirname(cache_path)
            os.makedirs(base_dir, exist_ok=True)
            with open(os.path.join(base_dir, "cache_debug.log"), "a") as lf:
                lf.write(msg + "\n")
        except Exception:
            pass

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
        "physical_topology": topo,
        "ib_gbps": ib_gbps,
        "ll_ns": ll_ns,
        "collectives": colls,
        "backend": "analytical",
    }
    if bundle_paths:
        sig.update(
            {
                "multinode": True,
                "num_workers": getattr(getattr(hw_obj, "system_config", None), "num_workers", None),
                "dp": getattr(getattr(hw_obj, "sch_config", None), "dp", None),
                "lp": getattr(getattr(hw_obj, "sch_config", None), "lp", None),
                "mb": getattr(getattr(hw_obj, "sch_config", None), "mb", None),
            }
        )
    canonical = _canonical_sig(sig)

    # Generate configs
    files = generate_astrasim_configs_from_hw(hw_obj, out_dir=astra_config_dir, npus_count=npus_count)

    # Ensure workload ET exists (skip writing if all ranks present)
    label = _size_label(size_bytes)
    base_dir = os.path.join(astra_config_dir, "workload", comm.lower(), f"{npus_count}npus_{label}")
    prefix = os.path.join(base_dir, f"{comm.lower()}_{label}")
    expected = [f"{prefix}.{r}.et" for r in range(npus_count)]
    if not bundle_paths:
        if not all(os.path.exists(p) for p in expected):
            generate_workload_et(comm, npus_count, size_bytes, astra_config_dir=astra_config_dir)

    remote_mem_path = get_remote_memory_path()

    if bundle_paths:
        bundle_list = list(bundle_paths)
        for path in (files["system_json"], files["network_yaml"]):
            if path not in bundle_list:
                bundle_list.append(path)
        if remote_mem_path and os.path.exists(remote_mem_path) and remote_mem_path not in bundle_list:
            bundle_list.append(remote_mem_path)
        cache_key = _hash_file_bundle(bundle_list)
    else:
        cache_key = _hash_sig(canonical)

    # Locked read/check to avoid races with concurrent writers; on timeout, proceed best-effort
    with _cache_file_lock(cache_path, timeout_s=15.0, poll_s=0.05) as locked:
        cache = _load_cache(cache_path)
        if bundle_paths:
            entry = cache.get(cache_key) if cache else None
            if entry:
                cached_per = entry.get("per_node_sec", [])
                cached_max = float(entry.get("max_sec", 0.0))
                if cached_per and cached_max > 0 and len(cached_per) == int(npus_count) and all((t > 0 for t in cached_per)):
                    _cache_dbg(f"HIT bundle_hash={cache_key} comm={comm} npus={npus_count} size={size_bytes} max={cached_max}")
                    return cached_per, cached_max
                if locked:
                    fresh = _load_cache(cache_path)
                    if cache_key in fresh:
                        try:
                            fresh.pop(cache_key)
                            _save_cache(cache_path, fresh)
                        except Exception:
                            pass
                _cache_dbg(f"DROP_INVALID bundle_hash={cache_key} comm={comm} npus={npus_count} size={size_bytes}")
        else:
            entry = cache.get(cache_key) if cache else None
            if locked:
                # Migrate any legacy entry keyed by canonical JSON string
                if cache and canonical in cache and cache_key not in cache:
                    cache[cache_key] = cache.pop(canonical)
                    _save_cache(cache_path, cache)
            if entry:
                cached_per = entry.get("per_node_sec", [])
                cached_max = float(entry.get("max_sec", 0.0))
                if cached_per and cached_max > 0 and len(cached_per) == int(npus_count) and all((t > 0 for t in cached_per)):
                    _cache_dbg(f"HIT key={cache_key} comm={comm} npus={npus_count} size={size_bytes} max={cached_max}")
                    return cached_per, cached_max
                if locked:
                    fresh = _load_cache(cache_path)
                    if cache_key in fresh:
                        try:
                            fresh.pop(cache_key)
                            _save_cache(cache_path, fresh)
                        except Exception:
                            pass
                _cache_dbg(f"DROP_INVALID key={cache_key} comm={comm} npus={npus_count} size={size_bytes}")

    # Execute AstraSim with retry if zero time observed
    attempts = 0
    per_node_sec: List[float] = []
    max_sec: float = 0.0
    last_outcome_zero = False
    while attempts < 5:
        attempts += 1
        per_node_sec, max_sec = run_astrasim_analytical(
            workload_prefix=prefix,
            system_json=files["system_json"],
            network_yaml=files["network_yaml"],
            remote_memory_json=remote_mem_path,
        )
        # consider valid only if non-empty, positive, and matches npus length
        if per_node_sec and max_sec > 0 and len(per_node_sec) == int(npus_count) and all((t > 0 for t in per_node_sec)):
            last_outcome_zero = False
            break
        last_outcome_zero = True
        # brief backoff before retrying
        time.sleep(0.5)
        retry_key = cache_key
        _cache_dbg(f"RETRY attempt={attempts} key={retry_key} comm={comm} npus={npus_count} size={size_bytes}")
    if last_outcome_zero:
        raise RuntimeError(
            f"AstraSim returned zero time for {comm} size={size_bytes} npus={npus_count} after 5 retries"
        )

    # Update cache under lock to prevent lost updates; if lock cannot be acquired, skip caching
    cache_entry = {
        "signature": sig,
        "canonical": canonical,
        "per_node_sec": per_node_sec,
        "max_sec": max_sec,
        "workload_prefix": prefix,
        "system_json": files["system_json"],
        "network_yaml": files["network_yaml"],
    }
    if bundle_paths:
        cache_entry.update({
            "bundle_hash": cache_key,
            "bundle_paths": bundle_list,
        })

    with _cache_file_lock(cache_path, timeout_s=15.0, poll_s=0.05) as locked:
        if locked:
            cache = _load_cache(cache_path)
            cache[cache_key] = cache_entry
            _save_cache(cache_path, cache)
            _cache_dbg(f"MISS_WRITE key={cache_key} comm={comm} npus={npus_count} size={size_bytes} max={max_sec}")
        else:
            _cache_dbg(f"SKIP_WRITE_LOCK key={cache_key} comm={comm} npus={npus_count} size={size_bytes}")

    return per_node_sec, max_sec

    # Unused generate_astrasim_run_files removed; use run_cache_astrasim instead


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
    comm_group_json: Optional[str] = None,
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
    if comm_group_json and os.path.exists(comm_group_json):
        cmd.append(f"--comm-group-configuration={comm_group_json}")

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


def _new_comp_node(node_id: int, name: str, duration_micros: int) -> pb.Node:
    """Create a compute node for timing delays."""
    n = pb.Node()
    n.id = node_id
    n.name = name
    n.type = pb.COMP_NODE
    n.duration_micros = duration_micros
    n.attr.append(pb.AttributeProto(name="is_cpu_op", bool_val=False))
    return n


def generate_concurrent_collectives_et(
    npus_count: int,
    collectives: List[Tuple[str, int, int]],
    prefix_path: str
) -> str:
    """
    Generate Chakra ETs for multiple collectives with optional timing delays.

    Args:
        npus_count: Number of NPUs/ranks
        collectives: List of (collective_type, size_bytes, start_delay_ns) tuples
                    - collective_type: "all_gather", "all_reduce", etc.
                    - size_bytes: Communication size
                    - start_delay_ns: Delay before starting (0 = immediate)
        prefix_path: Path prefix for output files (without .rank.et extension)

    Returns:
        prefix_path for use with AstraSim
    """
    # Ensure parent dir exists
    _ensure_dir(os.path.dirname(prefix_path))

    # Map collective names to protobuf types
    coll_map = {
        "all_gather": pb.ALL_GATHER,
        "allreduce": pb.ALL_REDUCE,
        "all_reduce": pb.ALL_REDUCE,
        "reduce_scatter": pb.REDUCE_SCATTER,
        "all_to_all": pb.ALL_TO_ALL,
        "alltoall": pb.ALL_TO_ALL,
    }

    # Create ET files for each rank
    for rank in range(npus_count):
        et_path = f"{prefix_path}.{rank}.et"
        with open(et_path, "wb") as fh:
            # Global metadata
            chakra_encode(fh, pb.GlobalMetadata(version="0.0.4"))

            node_id = 0

            # Process each collective
            for i, (coll_name, size_bytes, delay_ns) in enumerate(collectives):
                if coll_name not in coll_map:
                    raise ValueError(f"Unsupported collective: {coll_name}")

                coll_type = coll_map[coll_name]

                if delay_ns > 0:
                    # Create delay compute node
                    delay_micros = delay_ns // 1000  # Convert ns to microseconds
                    delay_node = _new_comp_node(
                        node_id,
                        f"delay_{i}_{delay_micros}us",
                        delay_micros
                    )
                    write_et_node(fh, delay_node)
                    delay_node_id = node_id
                    node_id += 1

                    # Create collective node that depends on delay
                    comm_node = _new_comm_node(
                        node_id,
                        f"{coll_name}_{i}",
                        coll_type,
                        size_bytes
                    )
                    comm_node.ctrl_deps.append(delay_node_id)  # Wait for delay
                    write_et_node(fh, comm_node)
                    node_id += 1
                else:
                    # Create collective node with no dependencies (immediate start)
                    comm_node = _new_comm_node(
                        node_id,
                        f"{coll_name}_{i}",
                        coll_type,
                        size_bytes
                    )
                    write_et_node(fh, comm_node)
                    node_id += 1

    return prefix_path


if __name__ == "__main__":
    import tempfile
    import subprocess
    import re

    print("=== AstraSim Congestion Awareness Test ===")
    print("Testing: Sequential vs Concurrent AllGather operations")
    print()

    # Test parameters
    npus_count = 4
    size_bytes = 1024 * 1024  # 1MB
    delay_ns = 1000 * 1000    # 1ms delay

    # Create temporary directory for test files
    with tempfile.TemporaryDirectory(prefix="astra_congestion_test_") as tmpdir:
        print(f"Test directory: {tmpdir}")

        # Test 1: Concurrent AllGather (both start at t=0)
        concurrent_prefix = os.path.join(tmpdir, "concurrent_allgather")
        concurrent_collectives = [
            ("all_gather", size_bytes, 0),  # No delay
            ("all_reduce", size_bytes*2, 0),  # No delay
        ]
        generate_concurrent_collectives_et(npus_count, concurrent_collectives, concurrent_prefix)

        # Test 2: Sequential AllGather (second starts after delay)
        sequential_prefix = os.path.join(tmpdir, "sequential_allgather")
        sequential_collectives = [
            ("all_gather", size_bytes, 0),        # Immediate
            ("all_reduce", size_bytes*2, delay_ns), # After 1ms delay
        ]
        generate_concurrent_collectives_et(npus_count, sequential_collectives, sequential_prefix)

        # Network and system configs (Ring topology, 50GB/s, 4 NPUs)
        network_config = os.path.join(tmpdir, "Ring_4npus.yml")
        with open(network_config, "w") as f:
            f.write("topology: [ FullyConnected ]\n")
            f.write("npus_count: [ 4 ]\n")
            f.write("bandwidth: [ 50.0 ]  # GB/s\n")
            f.write("latency: [ 500.0 ]  # ns\n")

        system_config = os.path.join(tmpdir, "Ring_4chunks.json")
        with open(system_config, "w") as f:
            json.dump({
                "scheduling-policy": "LIFO",
                "endpoint-delay": 10,
                "active-chunks-per-dimension": 1,
                "preferred-dataset-splits": 1,
                "all-reduce-implementation": ["direct"],
                "all-gather-implementation": ["direct"],
                "reduce-scatter-implementation": ["direct"],
                "all-to-all-implementation": ["direct"],
                "collective-optimization": "localBWAware",
                "local-mem-bw": 1600,
                "boost-mode": 0,
                "roofline-enabled": 0,
                "peak-perf": 900
            }, f, indent=2)

        remote_memory_config = get_remote_memory_path()

        # Test both congestion models
        for congestion_model in ["congestion_aware", "congestion_unaware"]:
            print(f"\n--- Testing {congestion_model.upper()} model ---")

            binary_path = f"./astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_{congestion_model.title()}"

            results = {}
            for test_name, workload_prefix in [("concurrent", concurrent_prefix), ("sequential", sequential_prefix)]:
                print(f"Running {test_name} test...")

                cmd = [
                    binary_path,
                    f"--workload-configuration={workload_prefix}",
                    f"--system-configuration={system_config}",
                    f"--remote-memory-configuration={remote_memory_config}",
                    f"--network-configuration={network_config}"
                ]

                try:
                    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, check=False, timeout=60)
                    output = proc.stdout

                    # Parse wall times
                    times_cycles = []
                    for line in output.splitlines():
                        m = re.search(r"sys\[(\d+)\],\s*Wall time:\s*(\d+)", line)
                        if m:
                            times_cycles.append(int(m.group(2)))

                    if times_cycles:
                        max_cycles = max(times_cycles)
                        max_time_ns = max_cycles  # 1 cycle = 1 ns
                        results[test_name] = max_time_ns
                        print(f"  Max completion time: {max_time_ns:,} ns ({max_time_ns/1e6:.2f} ms)")
                    else:
                        print(f"  ERROR: Could not parse timing from output")
                        results[test_name] = None

                except subprocess.TimeoutExpired:
                    print(f"  ERROR: Test timed out")
                    results[test_name] = None
                except Exception as e:
                    print(f"  ERROR: {e}")
                    results[test_name] = None

            # Compare results
            if results["concurrent"] and results["sequential"]:
                conc_time = results["concurrent"]
                seq_time = results["sequential"] - delay_ns
                ratio = seq_time / conc_time
                overhead = ((conc_time - seq_time) / seq_time) * 100

                print(f"\nResults for {congestion_model}:")
                print(f"  Concurrent: {conc_time:,} ns ({conc_time/1e6:.2f} ms)")
                print(f"  Sequential: {seq_time:,} ns ({seq_time/1e6:.2f} ms)")
                print(f"  Ratio (seq/conc): {ratio:.3f}")
                print(f"  Congestion overhead: {overhead:+.1f}%")

                if congestion_model == "congestion_aware":
                    if overhead > 5:
                        print("  ✅ PASS: Congestion model shows realistic overhead")
                    else:
                        print("  ⚠️  MARGINAL: Low congestion overhead detected")
                else:
                    if abs(overhead) < 5:
                        print("  ✅ PASS: Congestion-unaware shows minimal overhead")
                    else:
                        print("  ⚠️  UNEXPECTED: Congestion-unaware shows significant overhead")
            else:
                print(f"\n❌ FAILED: Could not complete {congestion_model} tests")

    print("\n=== Test Complete ===")
