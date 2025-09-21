"""Performance-oriented GEMM tiling utilities.

This module models memory traffic for tiled GEMM on a multi-level memory system.
It is intentionally written to be very fast in pure Python because it is called
hundreds of thousands of times. Key design choices for performance and determinism:

- Avoid dynamic string processing in hot paths. Loop orders and dataflows are
  encoded as small integers, not strings. Conversion functions are provided for
  user-friendly inputs but are executed once up-front.
- Use integer-only arithmetic for loop counts and byte calculations to avoid
  floating-point rounding and to keep Python operations simple and predictable.
- Precompute frequently used quantities (e.g., clamped tile sizes, byte counts,
  integer factors) inside the `TiledGEMM` constructor so methods become very
  cheap property lookups and direct arithmetic.
- Memoize pure helper functions with `functools.lru_cache`.
- Keep formulas closed-form (no Python loops inside hot helpers) so the cost is
  dominated by a handful of integer ops and small conditionals.
"""

# This file uses the following trick repeatedly:
# In python, for positive integer A,B, the following is true:
# math.ceil(A/B) == -(-A//B)
# Dataflow/loop semantics in one place:
# Only the inner loop dimension affects stationarity and reload patterns.
# We encode inner loop as an int for speed and small cache keys:
#    0 → 'm' → WST (weights stationary)
#    1 → 'k' → OST (outputs stationary)
#    2 → 'n' → AST (activations stationary)
# "BEST" picks the largest reuse among M/FMA_x, N/FMA_x, K/FMA_y, then maps to WST/OST/AST accordingly.

from math import ceil
from functools import lru_cache
from enum import IntEnum
from typing import Tuple, Union


class Dataflow(IntEnum):
    """Dataflow policies as compact integer codes.

    Using integers here (instead of strings) keeps hot-path branching simple and
    cache keys small. Human-facing APIs may accept strings and convert once.
    """
    NONE = 0
    BEST = 1
    WST = 2
    AST = 3
    OST = 4


class InnerLoop(IntEnum):
    """Inner loop dimension code used for loop-order decisions.
    Also an IntEnum to keep comparisons and cache keys efficient while adding
    readability. 0→M, 1→K, 2→N.
    """
    M = 0
    K = 1
    N = 2


def as_dataflow_code(df: Union["Dataflow", int, str]) -> int:
    """Normalize user input into an integer dataflow code.

    Accepts enum, int, or common string shorthands (e.g., "wst"/"ws"). The
    returned value is a small int suitable for hot-path decisions and caching.
    """
    if isinstance(df, Dataflow):
        return int(df)
    if isinstance(df, int):
        return df
    if isinstance(df, str):
        s = df.lower()
        if s == "none":
            return int(Dataflow.NONE)
        if s == "best":
            return int(Dataflow.BEST)
        if s in ("wst", "ws"):
            return int(Dataflow.WST)
        if s in ("ast", "as"):
            return int(Dataflow.AST)
        if s in ("ost", "os"):
            return int(Dataflow.OST)
    raise ValueError(f"Unknown dataflow: {df}")
from enum import IntEnum


def inner_code_from_order(order_dims: str) -> InnerLoop:
    """Extract the inner loop dimension code from a 3-char order string.

    Returns: 0 for 'm', 1 for 'k', 2 for 'n'. Keeping this as an int lets us
    branch cheaply and form small cache keys later.
    """
    ch = order_dims[2]
    if ch == 'm':
        return InnerLoop.M
    if ch == 'k':
        return InnerLoop.K
    if ch == 'n':
        return InnerLoop.N
    raise ValueError(f"Unknown inner loop in order: {order_dims}")


def inner_code_to_char(code: Union[int, InnerLoop]) -> str:
    """Inverse of `inner_code_from_order` for diagnostics and display."""
    return 'm' if code == InnerLoop.M or code == 0 else (
        'k' if code == InnerLoop.K or code == 1 else 'n'
    )

# We only enumerate candidate *inner* loops, not full 3-permutations.
# Rationale: the stationary policy and all reload counts depend only on the
# innermost loop. The outer-loop order influences traversal, not totals, in our
# closed-form model. This reduces the search space without changing totals.
def generate_orders_inner_codes(dataflow_code: int, dim1: int, dim2: int, dim3: int) -> Tuple[InnerLoop, ...]:
    """Return preferred inner-loop codes for a given dataflow and problem size.

    We avoid building full 3-char permutations and reduce the space to the
    unique inner loops in the desired priority.
    """
    # Crucial insight: only inner code matters, so we can reduce the space to the unique inner codes in the desired priority.
    # TODO: Verify this is ok.
    if dataflow_code == Dataflow.BEST:
        if dim1 >= dim2 and dim1 >= dim3:
            dataflow_code = Dataflow.WST
        elif dim2 >= dim1 and dim2 >= dim3:
            dataflow_code = Dataflow.OST
        else:
            dataflow_code = Dataflow.AST

    if dataflow_code == Dataflow.WST:
        # was ["mnk", (optional) "mkn"] -> inner: k, n
        out = (InnerLoop.K, InnerLoop.N)
        return out if dim2 != dim3 else (InnerLoop.K,)
    if dataflow_code == Dataflow.AST:
        # was ["nmk", (optional) "nkm"] -> inner: k, m
        out = (InnerLoop.K, InnerLoop.M)
        return out if dim2 != dim1 else (InnerLoop.K,)
    if dataflow_code == Dataflow.OST:
        # was ["knm", (optional) "kmn"] -> inner: m, n
        out = (InnerLoop.M, InnerLoop.N)
        return out if dim1 != dim3 else (InnerLoop.M,)
    # NONE or BEST → consider all permutations; order influences search priority only.
    # Permutations of 'mnk': mnk, mkn, nmk, nkm, kmn, knm -> inner: k, n, k, m, n, m.
    # Reduce to unique inner codes in first-seen priority: k, n, m
    return (InnerLoop.K, InnerLoop.N, InnerLoop.M)


def generate_tile_space(memLayer, num_levels: int, dim1: int, dim2: int, dim3: int):
    """Enumerate tile shapes across memory levels.

    The memory objects define their permissible tile sizes. We build the space
    without nested Python loops where possible and keep the tuples compact.
    A small conditional on `original` is preserved for legacy behavior.

    Largely unchanged from original function as it is not perf-critical.
    """
    original = True
    tile_space = []
    tiles = [None] * num_levels
    for level in range(0, num_levels - 1):
        memory = memLayer[level]
        if not original and level == 2:
            tiles[level] = memory.getGEMMBasedTileDims(dim1, dim2, dim3)
        else:
            tiles[level] = memory.getTileDims()

    if num_levels == 1:
        tile_space = []
    elif num_levels == 2:
        tile_space = tiles[0]
    elif num_levels == 3:
        tile_space = [(x, y) for x in tiles[0] for y in tiles[1]]
    elif num_levels == 4:
        tile_space = []
        for x in tiles[2]:
            t1, t2, t3 = x
            if not original:
                tiles[1] = memLayer[1].getGEMMBasedTileDims(t1, t2, t3)
            tile_strategy = [
                (x, y, z) for y in tiles[1] for z in tiles[2]
            ]
            tile_space.extend(tile_strategy)
    else:
        raise NotImplementedError()

    return tile_space

@lru_cache(maxsize=65536)
def _sysarray_accesses_sig(M: int, N: int, K: int,
                           FMA_x: int, FMA_y: int,
                           dataflow_code: Union[int, Dataflow], dtype_size: int) -> float:
    """Estimate systolic-array traffic (bytes) for a GEMM tile.

    Hot-path, pure function. Memoized aggressively because many evaluations
    repeat the same `(M,N,K,FMA_x,FMA_y,dataflow_code,dtype_size)` tuples

    The closed-form expressions below avoid Python loops. Most math uses ints.
    """

    reuse = 1 # NONE
    if dataflow_code == Dataflow.BEST:  # BEST
        a = -(-M // FMA_x) # == math.ceil(M/FMA_x)
        b = -(-N // FMA_x)
        c = -(-K // FMA_y)
        reuse = max(a,b,c)
    elif dataflow_code == Dataflow.WST:  # WST
        reuse = -(-M // FMA_x)
    elif dataflow_code == Dataflow.AST:  # AST
        reuse = -(-N // FMA_x)
    elif dataflow_code == Dataflow.OST:  # OST
        reuse = -(-K // FMA_y)

    GEMM_flop = M * N * (2 * K - 1)
    load_bytes = GEMM_flop * 2.0 * FMA_y / (FMA_x * (2 * FMA_y - 1)) * dtype_size
    store_bytes = GEMM_flop / (reuse * (2 * FMA_y - 1)) * dtype_size
    return load_bytes + store_bytes


@lru_cache(maxsize=131072)
def _simulate_accesses_sig(inner_code: Union[int, InnerLoop], M: int, K: int, N: int,
                           l2_M: int, l2_K: int, l2_N: int,
                           l1_M: int, l1_K: int, l1_N: int,
                           dtype_size: int, FMA_x: int, FMA_y: int, dataflow_code: Union[int, Dataflow],
                           capacity: int, total_bytes: int) -> Tuple[float, int, int, int]:
    """Compute memory traffic at each level (sys, L1/shared, L2, DRAM).

    Parameters are integers. The function is pure and amenable to
    caching. It uses closed-form counts derived from tile reuse rather than
    iterating through actual tiles, which keeps evaluation fast.
    """
    sys_bytes = _sysarray_accesses_sig(M, N, K, FMA_x, FMA_y, dataflow_code, dtype_size)

    num_tiles_M = -(-M // l2_M) # == math.ceil(M/l2_M)
    num_tiles_K = -(-K // l2_K)
    num_tiles_N = -(-N // l2_N)
    max_reload = num_tiles_M * num_tiles_N * num_tiles_K

    reuse_M = -(-l2_M // l1_M)
    reuse_K = -(-l2_K // l1_K)
    reuse_N = -(-l2_N // l1_N)
    read_bytes = (l1_M * l1_K * dtype_size + l1_K * l1_N * dtype_size) * (reuse_M * reuse_N * reuse_K)
    write_bytes = (l1_M * l1_N * dtype_size) * (reuse_M * reuse_N)
    l1_shared_total = (read_bytes + write_bytes) * max_reload

    # L2 accesses
    factor_M = reuse_M
    factor_K = reuse_K
    factor_N = reuse_N
    if inner_code == InnerLoop.M:
        mk_load = max_reload
        kn_load = num_tiles_K * num_tiles_N
        mn_load = max_reload
        factor_M = 1
        dfk = Dataflow.WST
    elif inner_code == InnerLoop.K:
        mk_load = max_reload
        kn_load = max_reload
        mn_load = num_tiles_M * num_tiles_N
        factor_K = 1
        dfk = Dataflow.OST
    elif inner_code == InnerLoop.N:
        mk_load = num_tiles_M * num_tiles_K
        kn_load = max_reload
        mn_load = max_reload
        factor_N = 1
        dfk = Dataflow.AST

    l2_bytes = (
        mk_load * (l2_M * l2_K * dtype_size) * factor_N
        + kn_load * (l2_K * l2_N * dtype_size) * factor_M
        + 2 * mn_load * (l2_M * l2_N * dtype_size) * factor_K
    )

    l2f0 = M // l2_M
    l2f1 = K // l2_K
    l2f2 = N // l2_N
    reload_mk = 1
    reload_kn = 1
    reload_mn = 1
    if dfk == Dataflow.WST:
        reload_mk = l2f2
        reload_mn = l2f1
    elif dfk == Dataflow.OST:
        reload_mk = l2f2
        reload_kn = l2f0
    elif dfk == Dataflow.AST:
        reload_kn = l2f0
        reload_mn = l2f1

    dram_counts = reload_mk * (M * K * dtype_size) + reload_kn * (K * N * dtype_size)
    if total_bytes > capacity:
        dram_counts += reload_mn * (M * N * dtype_size)

    return (sys_bytes, l1_shared_total, l2_bytes, dram_counts)



class TiledGEMM:
    """Performance-focused container for a tiled GEMM instance.
    The constructor normalizes and precomputes everything that can be reused so
    later method calls are O(1) arithmetic with no string parsing and no loops.
    """

    def __init__(self, order_dims, tile_dims, core, memLayer, dtype_size=2):
        self.num_bundle = core.num_bundle
        tile_dims = list(tile_dims)
        # Problem and memory parameters
        self.memLayer = memLayer
        self.dtype_size = dtype_size
        self.M, self.K, self.N = tile_dims[3]
        self.capacity = memLayer[3].size_per_bundle
        self.order_dims = order_dims if isinstance(order_dims, str) else None
        # Precompute inner code once (accepts str or int code)
        self._inner_code = (
            inner_code_from_order(order_dims)
            if isinstance(order_dims, str)
            else int(order_dims)
        )
        self.FMA_x, self.FMA_y = core.FMA_dims
        # Normalize dataflow to enum code (small ints make cheap cache keys)
        self.dataflow = core.dataflow
        self._dataflow_code = as_dataflow_code(self.dataflow)


        # Precompute frequently used byte-size values once.
        # These are used across multiple methods and remain constant per instance.
        self.mk_bytes_ = self.M * self.K * self.dtype_size
        self.kn_bytes_ = self.K * self.N * self.dtype_size
        self.mn_bytes_ = self.M * self.N * self.dtype_size
        self.total_bytes = self.mk_bytes_ + self.kn_bytes_ + self.mn_bytes_

        # Compute clamped dims for L2/L1/L0 once.
        # L3 is the full problem (self.M, self.K, self.N).
        l2_src = tile_dims[2]
        l1_src = tile_dims[1]
        l0_src = tile_dims[0]
        self.l2_M = min(l2_src[0], self.M)
        self.l2_K = min(l2_src[1], self.K)
        self.l2_N = min(l2_src[2], self.N)
        self.l1_M = min(l1_src[0], self.l2_M)
        self.l1_K = min(l1_src[1], self.l2_K)
        self.l1_N = min(l1_src[2], self.l2_N)
        self.l0_M = min(l0_src[0], self.l1_M)
        self.l0_K = min(l0_src[1], self.l1_K)
        self.l0_N = min(l0_src[2], self.l1_N)

        self.mem_accesses = self.simulate_accesses()

    # def __repr__(self):
    #     return (
    #         f"  DRAM read: {formatBytes(self.mem_read[3])}, write: {formatBytes(self.mem_write[3])}\n"
    #         f"  L2 read: {formatBytes(self.mem_read[2])}, write: {formatBytes(self.mem_write[2])}\n"
    #         f"  Shared read: {formatBytes(self.mem_read[1])}, write: {formatBytes(self.mem_write[1])}\n"
    #         f"  Reg read: {formatBytes(self.mem_read[0])}, write: {formatBytes(self.mem_write[0])}\n"
    #         f"  loop order: {self.order_dims}\n"
    #         f"{self.tile.__repr__()}"
    #     )
    #     return [r + w for r, w in zip(self.mem_read, self.mem_write)]

    @property
    def GEMM_flop(self):
        """Theoretical FLOP count for the full GEMM of size MxKxN."""
        return self.M * self.N * (2 * self.K - 1)

    def sysarray_accesses(self):
        """Return systolic-array traffic (bytes) using cached pure function.

        Keeping method thin avoids duplicating logic and ensures cache hits.
        """
        return _sysarray_accesses_sig(self.M, self.N, self.K, self.FMA_x, self.FMA_y, self._dataflow_code, self.dtype_size)
    
    def simulate_accesses(self):
        """Compute all memory-level bytes (sys, L1, L2, DRAM) for this instance.
        Thin wrapper around the cached pure function."""
        return _simulate_accesses_sig(
            self._inner_code,
            self.M, self.K, self.N,
            self.l2_M, self.l2_K, self.l2_N,
            self.l1_M, self.l1_K, self.l1_N,
            self.dtype_size, self.FMA_x, self.FMA_y, self._dataflow_code,
            self.capacity, self.total_bytes,
        )

    @classmethod
    def enumerate_candidates(self, core, memLayer, dim1: int, dim2: int, dim3: int,
                              dtype_size: int = 2, original: bool = False):
        """Yield `TiledGEMM` instances across tile shapes and preferred inner codes.

        The generator form avoids materializing the full search space in memory.
        Inner loop choices are small ints for speed and cache-friendly keys.
        """
        df_code = as_dataflow_code(core.dataflow)
        inner_codes = generate_orders_inner_codes(df_code, dim1, dim2, dim3)
        space = generate_tile_space(memLayer, len(memLayer), dim1, dim2, dim3)
        for (l0, l1, l2) in space:
            tile_dims = (l0, l1, l2, (dim1, dim2, dim3))
            for code in inner_codes:
                yield self(code, tile_dims, core, memLayer, dtype_size)
    
    def per_layer_capacity(self):
        """Return utilization per memory level relative to its capacity.

        Computed from pre-stored dims to avoid recomputation and branches.
        """
        ds = self.dtype_size
        l2_total = (self.l2_M * self.l2_K * ds) + (self.l2_K * self.l2_N * ds) + (self.l2_M * self.l2_N * ds)
        l1_total = (self.l1_M * self.l1_K * ds) + (self.l1_K * self.l1_N * ds) + (self.l1_M * self.l1_N * ds)
        return [l2_total / self.memLayer[2].size_per_bundle, l1_total / self.memLayer[1].size_per_bundle]

    @property
    def mk_bytes(self):
        """Bytes of the MK matrix at the full problem scale."""
        return self.mk_bytes_

    @property
    def kn_bytes(self):
        """Bytes of the KN matrix at the full problem scale."""
        return self.kn_bytes_

    @property
    def mn_bytes(self):
        """Bytes of the MN matrix at the full problem scale."""
        return self.mn_bytes_


def formatBytes(size):
    """Format bytes into a human-readable string for diagnostics and logs."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TB"


if __name__ == "__main__":
    import os
    import time

    class _Core:
        def __init__(self, num_bundle, FMA_dims, dataflow):
            self.num_bundle = num_bundle
            self.FMA_dims = FMA_dims
            self.dataflow = dataflow

    class _Mem:
        def __init__(self, size_per_bundle):
            self.size_per_bundle = size_per_bundle

    # Simple micro-benchmark for local validation and perf sanity checking.
    # Repeats the same shapes to exercise the caches and measure steady-state.
    dataflows = ["none", "best", "wst", "ast", "ost"]
    orders = ["mkn", "kmn", "nmk", "mnk", "knm", "nkm"]
    cores = [_Core(num_bundle=4, FMA_dims=(8, 4), dataflow=df) for df in dataflows]

    # Memory layer capacities per level (0..3)
    memLayer = [
        _Mem(256 * 1024),  # L0/register-like
        _Mem(192 * 1024),  # L1/shared
        _Mem(40 * 1024 * 1024),  # L2
        _Mem(40 * 1024 * 1024 * 1024),  # DRAM
    ]

    # Problem sizes
    small = [64, 96, 128, 160]*2
    medium = [256, 384, 512, 768, 1024]*2
    large = [2048, 2560, 3072]*2
    sizes = small + medium + large

    # Tile shapes for (L0, L1, L2). L3 is the full problem tile (M,K,N)
    tile_shapes = [
        ((16, 16, 16), (64, 64, 64), (256, 256, 256)),
        ((16, 32, 16), (64, 128, 64), (256, 512, 256)),
        ((32, 16, 32), (128, 64, 128), (512, 256, 512)),
    ]


    build_count = 0
    checksum = 0.0
    t0 = time.perf_counter()
    repeat = 20
    for _ in range(repeat):
        for M in sizes:
            for N in sizes[:8]:
                for K in sizes[:8]:
                    for (l0, l1, l2) in tile_shapes:
                        l2M = min(l2[0], M); l2K = min(l2[1], K); l2N = min(l2[2], N)
                        l1M = min(l1[0], l2M); l1K = min(l1[1], l2K); l1N = min(l1[2], l2N)
                        l0M = min(l0[0], l1M); l0K = min(l0[1], l1K); l0N = min(l0[2], l1N)
                        tile_dims = [
                            (l0M, l0K, l0N),
                            (l1M, l1K, l1N),
                            (l2M, l2K, l2N),
                            (M, K, N),
                        ]
                        for order in orders:
                            for core in cores:
                                tg = TiledGEMM(order, tile_dims, core, memLayer, dtype_size=2)
                                ma0, ma1, ma2, ma3 = tg.mem_accesses
                                checksum += (ma0 + ma1 + ma2 + ma3) * 1e-18
                                for uc in tg.per_layer_capacity():
                                    checksum += uc * 1e-6
                                build_count += 1

    t1 = time.perf_counter()
    elapsed = (t1 - t0)/repeat
    print(
        f"Built {build_count} TiledGEMM instances in avg {elapsed:.3f}s (repeat={repeat}), checksum={checksum:.3f}"
    )
