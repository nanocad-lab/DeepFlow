from math import ceil
from functools import lru_cache
from enum import IntEnum
# from numba import njit
from typing import Tuple, Union


def my_njit(func):
    # return njit(cache=True, fastmath=True)(func)
    return func

class Dataflow(IntEnum):
    NONE = 0
    BEST = 1
    WST = 2
    AST = 3
    OST = 4


def as_dataflow_code(df: Union["Dataflow", int, str]) -> int:
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


def inner_code_from_order(order_dims: str) -> int:
    ch = order_dims[2]
    if ch == 'm':
        return 0
    if ch == 'k':
        return 1
    if ch == 'n':
        return 2
    raise ValueError(f"Unknown inner loop in order: {order_dims}")


def inner_code_to_char(code: int) -> str:
    return 'm' if code == 0 else ('k' if code == 1 else 'n')


def generate_orders_inner_codes(dataflow_code: int, dim1: int, dim2: int, dim3: int) -> Tuple[int, ...]:
    # Return preferred inner loop codes based on dataflow; eliminate strings
    # If BEST, narrow to a concrete policy based on largest dimension (legacy heuristic)
    if dataflow_code == int(Dataflow.BEST):
        if dim1 >= dim2 and dim1 >= dim3:
            dataflow_code = int(Dataflow.WST)
        elif dim2 >= dim1 and dim2 >= dim3:
            dataflow_code = int(Dataflow.OST)
        else:
            dataflow_code = int(Dataflow.AST)

    if dataflow_code == int(Dataflow.WST):
        # was ["mnk", (optional) "mkn"] -> inner: k, n
        out = (1, 2)
        return out if dim2 != dim3 else (1,)
    if dataflow_code == int(Dataflow.AST):
        # was ["nmk", (optional) "nkm"] -> inner: k, m
        out = (1, 0)
        return out if dim2 != dim1 else (1,)
    if dataflow_code == int(Dataflow.OST):
        # was ["knm", (optional) "kmn"] -> inner: m, n
        out = (0, 2)
        return out if dim1 != dim3 else (0,)
    # NONE or BEST → consider all permutations; order influences search priority only
    # permutations of 'mnk': mnk, mkn, nmk, nkm, kmn, knm -> inner: k, n, k, m, n, m
    # Reduce to unique inner codes in first-seen priority: k, n, m
    return (1, 2, 0)


def generate_tile_space(memLayer, num_levels: int, dim1: int, dim2: int, dim3: int):
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

@my_njit
def _sysarray_accesses_jit(M: int, N: int, K: int,
                           FMA_x: int, FMA_y: int,
                           dataflow_code: int, dtype_size: int) -> float:
    reuse = 1 # NONE 
    if dataflow_code == 1: # BEST
        a = (M + FMA_x - 1) // FMA_x
        b = (N + FMA_x - 1) // FMA_x
        c = (K + FMA_y - 1) // FMA_y
        reuse = a if a >= b and a >= c else (b if b >= a and b >= c else c)
    elif dataflow_code == 2: # WST
        reuse = (M + FMA_x - 1) // FMA_x
    elif dataflow_code == 3: # AST
        reuse = (N + FMA_x - 1) // FMA_x
    elif dataflow_code == 4: # OST
        reuse = (K + FMA_y - 1) // FMA_y

    GEMM_flop = M * N * (2 * K - 1)
    load_bytes = GEMM_flop * 2.0 * FMA_y / (FMA_x * (2 * FMA_y - 1)) * dtype_size
    store_bytes = GEMM_flop / (reuse * (2 * FMA_y - 1)) * dtype_size
    return load_bytes + store_bytes


@lru_cache(maxsize=65536)
def _sysarray_accesses_sig(M: int, N: int, K: int,
                           FMA_x: int, FMA_y: int,
                           dataflow_code: int, dtype_size: int) -> float:
    return _sysarray_accesses_jit(M, N, K, FMA_x, FMA_y, dataflow_code, dtype_size)


@my_njit
def _simulate_accesses_jit(inner_code: int, M: int, K: int, N: int,
                           l2_M: int, l2_K: int, l2_N: int,
                           l1_M: int, l1_K: int, l1_N: int,
                           dtype_size: int, FMA_x: int, FMA_y: int, dataflow_code: int,
                           capacity: int, total_bytes: int) -> Tuple[float, int, int, int]:
    sys_bytes = _sysarray_accesses_jit(M, N, K, FMA_x, FMA_y, dataflow_code, dtype_size)

    num_tiles_M = (M + l2_M - 1) // l2_M
    num_tiles_K = (K + l2_K - 1) // l2_K
    num_tiles_N = (N + l2_N - 1) // l2_N
    max_reload = num_tiles_M * num_tiles_N * num_tiles_K

    reuse_M = (l2_M + l1_M - 1) // l1_M
    reuse_K = (l2_K + l1_K - 1) // l1_K
    reuse_N = (l2_N + l1_N - 1) // l1_N
    read_bytes = (l1_M * l1_K * dtype_size + l1_K * l1_N * dtype_size) * (reuse_M * reuse_N * reuse_K)
    write_bytes = (l1_M * l1_N * dtype_size) * (reuse_M * reuse_N)
    l1_shared_total = (read_bytes + write_bytes) * max_reload

    # L2 accesses
    factor_M = reuse_M
    factor_K = reuse_K
    factor_N = reuse_N
    if inner_code == 0:  # M (ws)
        mk_load = max_reload
        kn_load = num_tiles_K * num_tiles_N
        mn_load = max_reload
        factor_M = 1
        dfk = 0
    elif inner_code == 1:  # K (os)
        mk_load = max_reload
        kn_load = max_reload
        mn_load = num_tiles_M * num_tiles_N
        factor_K = 1
        dfk = 1
    elif inner_code == 2:  # N (as)
        mk_load = num_tiles_M * num_tiles_K
        kn_load = max_reload
        mn_load = max_reload
        factor_N = 1
        dfk = 2

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
    if dfk == 0:
        reload_mk = l2f2
        reload_mn = l2f1
    elif dfk == 1:
        reload_mk = l2f2
        reload_kn = l2f0
    else:
        reload_kn = l2f0
        reload_mn = l2f1

    dram_counts = reload_mk * (M * K * dtype_size) + reload_kn * (K * N * dtype_size)
    if total_bytes > capacity:
        dram_counts += reload_mn * (M * N * dtype_size)

    return (sys_bytes, l1_shared_total, l2_bytes, dram_counts)


@lru_cache(maxsize=131072)
def _simulate_accesses_sig(inner_code: int, M: int, K: int, N: int,
                           l2_M: int, l2_K: int, l2_N: int,
                           l1_M: int, l1_K: int, l1_N: int,
                           dtype_size: int, FMA_x: int, FMA_y: int, dataflow_code: int,
                           capacity: int, total_bytes: int) -> Tuple[float, int, int, int]:
    return _simulate_accesses_jit(inner_code, M, K, N,
                                  l2_M, l2_K, l2_N,
                                  l1_M, l1_K, l1_N,
                                  dtype_size, FMA_x, FMA_y, dataflow_code,
                                  capacity, total_bytes)


def iceil_div(a: int, b: int) -> int:
    """Integer ceiling division using only integers."""
    return -(-a // b)

class TiledGEMM:
    # __slots__ = (
    #     'num_bundle', 'order_dims', 'FMA_x', 'FMA_y', 'dataflow', 'mem_accesses'
    # )
    """
    - order_dims: loop order for the GEMM computation, e.g. "mkn"
    - tile_dims: list of tuples (M, K, N) for each level of memory hierarchy
    - FMA_dims: tuple (FMA_x, FMA_y) representing the dimensions of the systolic array
    - dataflow: string representing the dataflow strategy, e.g. "none", "best", "wst", "ast", "ost"
    - dtype_size: size of one element in bytes (default is 2 bytes for fp16)
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
        # normalize dataflow to enum code
        self.dataflow = core.dataflow
        try:
            # Will be mapped when passed to cached funcs
            self._dataflow_code = as_dataflow_code(self.dataflow)
        except Exception:
            self._dataflow_code = 0

        # Precompute frequently used byte-size values once
        self.mk_bytes_ = self.M * self.K * self.dtype_size
        self.kn_bytes_ = self.K * self.N * self.dtype_size
        self.mn_bytes_ = self.M * self.N * self.dtype_size
        self.total_bytes = self.mk_bytes_ + self.kn_bytes_ + self.mn_bytes_

        # Compute clamped dims for L2/L1/L0 once
        # L3 is full problem (self.M, self.K, self.N)
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

        # Precompute factors using flattened dims
        self.num_tiles_M = iceil_div(self.M, self.l2_M)
        self.num_tiles_K = iceil_div(self.K, self.l2_K)
        self.num_tiles_N = iceil_div(self.N, self.l2_N)
        self.l2_factors = (self.M // self.l2_M, self.K // self.l2_K, self.N // self.l2_N)

        self.mem_accesses = self.simulate_accesses()

    def __repr__(self):
        return f"TiledGEMM {self.M}x{self.K}x{self.N}\n"

    @property
    def GEMM_flop(self):
        return self.M * self.N * (2 * self.K - 1)

    def sysarray_accesses(self):
        return _sysarray_accesses_sig(self.M, self.N, self.K, self.FMA_x, self.FMA_y, self._dataflow_code, self.dtype_size)
    
    def dram_accesses(self, df):
        l2_factors = self.l2_factors

        reload_mk, reload_kn, reload_mn = 1, 1, 1
        if df == "ws":
            reload_mk = l2_factors[2]
            reload_mn = l2_factors[1]
        elif df == "os":
            reload_mk = l2_factors[2]
            reload_kn = l2_factors[0]
        elif df == "as":
            reload_kn = l2_factors[0]
            reload_mn = l2_factors[1]

        dram_counts = reload_mk * self.mk_bytes + reload_kn * self.kn_bytes
        if self.total_bytes > self.capacity:
            dram_counts += reload_mn * self.mn_bytes

        return dram_counts

    def simulate_accesses(self):
        """Trace through tile accesses to each level of memory using precomputed inner code"""
        inner_code = self._inner_code
        return _simulate_accesses_sig(
            inner_code,
            self.M, self.K, self.N,
            self.l2_M, self.l2_K, self.l2_N,
            self.l1_M, self.l1_K, self.l1_N,
            self.dtype_size, self.FMA_x, self.FMA_y, self._dataflow_code,
            self.capacity, self.total_bytes,
        )

    @classmethod
    def enumerate_candidates(cls, core, memLayer, dim1: int, dim2: int, dim3: int,
                              dtype_size: int = 2, original: bool = False):
        """Yield TiledGEMM instances across all tile shapes and preferred inner codes.

        Accepts integer inner codes (0=m,1=k,2=n) to avoid string orders.
        """
        df_code = as_dataflow_code(core.dataflow)
        inner_codes = generate_orders_inner_codes(df_code, dim1, dim2, dim3)
        space = generate_tile_space(memLayer, len(memLayer), dim1, dim2, dim3)
        for (l0, l1, l2) in space:
            tile_dims = (l0, l1, l2, (dim1, dim2, dim3))
            for code in inner_codes:
                yield cls(code, tile_dims, core, memLayer, dtype_size)
    
    def per_layer_capacity(self):
        # Compute utilizations from stored dims
        ds = self.dtype_size
        l2_total = (self.l2_M * self.l2_K * ds) + (self.l2_K * self.l2_N * ds) + (self.l2_M * self.l2_N * ds)
        l1_total = (self.l1_M * self.l1_K * ds) + (self.l1_K * self.l1_N * ds) + (self.l1_M * self.l1_N * ds)
        return [l2_total / self.memLayer[2].size_per_bundle, l1_total / self.memLayer[1].size_per_bundle]

    @property
    def mk_bytes(self):
        return self.mk_bytes_

    @property
    def kn_bytes(self):
        return self.kn_bytes_

    @property
    def mn_bytes(self):
        return self.mn_bytes_


def formatBytes(size):
    """Format bytes into a human-readable string."""
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

    # Benchmark configuration
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
