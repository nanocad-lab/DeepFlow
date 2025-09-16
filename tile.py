from math import ceil

class Tile:
    def __init__(self, tile_dims, level, memLayer, dtype_size):
        """
        Base class for tile transfer in GEMM across memory hierarchy.

        Parameters:
        - M, N, K: Full GEMM problem dimensions (C = A x B, where A is MxK, B is KxN)
        - dtype_size: Size of one element in bytes (default is 2 bytes for fp16)
        """

        assert level > 0, "Invalid tile level"

        self.level = level
        self.M, self.K, self.N = tile_dims[level]
        self.dtype_size = dtype_size
        self.memLayer = memLayer
        self.capacity = memLayer[level].size_per_bundle

        tile_dims[level - 1] = tuple(
            [
                min(tile_dims[level - 1][i], tile_dims[level][i])
                for i in range(len(tile_dims[level]))
            ]
        )

        self.tile = self.get_tile(tile_dims)
        self.total_bytes = self.mk_bytes + self.kn_bytes + self.mn_bytes

    def __repr__(self):
        return (
            f"({self.level}) {self.__class__.__name__} {self.M}x{self.K}x{self.N}\n"
            + repr(self.tile)
            # f"  total bytes: {formatBytes(self.total_bytes)}\n"
            # f"      mk_bytes: {formatBytes(self.mk_bytes())}, kn_bytes: {formatBytes(self.kn_bytes())}, mn_bytes: {formatBytes(self.mn_bytes())}\n"
        )

    @property
    def mk_bytes(self):
        """Returns the size of A matrix in bytes (shape: M x K)"""
        return self.M * self.K * self.dtype_size

    @property
    def kn_bytes(self):
        """Returns the size of B matrix in bytes (shape: K x N)"""
        return self.K * self.N * self.dtype_size

    @property
    def mn_bytes(self):
        """Returns the size of C matrix in bytes (shape: M x N)"""
        return self.M * self.N * self.dtype_size
    
    @property
    def util_capacity(self):
        return self.total_bytes / self.capacity

    def get_tile(self, tile_dims):
        """Returns the lower level tile object"""
        return None


class TiledGEMM(Tile):
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
        super().__init__(tile_dims, 3, memLayer, dtype_size)
        self.order_dims = order_dims
        self.FMA_x, self.FMA_y = core.FMA_dims
        self.dataflow = core.dataflow
        self.mem_accesses = self.simulate_accesses(order_dims)

    def __repr__(self):
        return (
            super().__repr__()
            # + f"  DRAM read: {formatBytes(self.mem_read[3])}, write: {formatBytes(self.mem_write[3])}\n"
            # f"  L2 read: {formatBytes(self.mem_read[2])}, write: {formatBytes(self.mem_write[2])}\n"
            # f"  Shared read: {formatBytes(self.mem_read[1])}, write: {formatBytes(self.mem_write[1])}\n"
            # f"  Reg read: {formatBytes(self.mem_read[0])}, write: {formatBytes(self.mem_write[0])}\n"
            # f"  loop order: {self.order_dims}\n"
            # f"{self.tile.__repr__()}"
        )
        return [r + w for r, w in zip(self.mem_read, self.mem_write)]

    @property
    def GEMM_flop(self):
        return self.M * self.N * (2 * self.K - 1)

    def get_tile(self, tile_dims):
        return L2Tile(tile_dims, self.level - 1, self.num_bundle, self.memLayer, self.dtype_size)

    def sysarray_accesses(self):
        """
        assume systolic engine can support n x m x n GEMM  (e.g. 8 x 4 x 8 for A100 tensorcore), which is FLOPs_tile = n^2 * (m-1) FLOPs
        reuse is the number of n x m x n GEMMs that are performed before stationary values (weight, activations, or output) get swapped
        every n x n output tile:
          1. loads nxm activations and mxn weights -> 2 * reuse * n * m accesses
          2. performs reuse * FLOPs_tile computations
          3. writes back n^2 output elements
        """
        if self.dataflow == "none":
            reuse = 1
        elif self.dataflow == "best":
            reuse = max(
                ceil(self.M / self.FMA_x),
                ceil(self.N / self.FMA_x),
                ceil(self.K / self.FMA_y),
            )
        elif self.dataflow == "wst":  # weight stationary
            reuse = ceil(self.M / self.FMA_x)
        elif self.dataflow == "ast":  # activation stationary
            reuse = ceil(self.N / self.FMA_x)
        elif self.dataflow == "ost":  # output stationary
            reuse = ceil(self.K / self.FMA_y)
        else:
            raise NotImplementedError()

        load_bytes = (
            self.GEMM_flop
            * 2
            * self.FMA_y
            / (self.FMA_x * (2 * self.FMA_y - 1))
            * self.dtype_size
        )
        store_bytes = (
            self.GEMM_flop / (reuse * (2 * self.FMA_y - 1)) * self.dtype_size
        )
        return load_bytes + store_bytes
    
    def dram_accesses(self, df):
        l2_factors = (self.M // self.tile.M, self.K // self.tile.K, self.N // self.tile.N)

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

    def simulate_accesses(self, order_dims):
        """Trace through tile accesses to each level of memory for a given loop ordering"""
        mem_accesses = [0] * 4

        mem_accesses[0] = self.sysarray_accesses()

        num_tiles_M = ceil(self.M / self.tile.M)
        num_tiles_K = ceil(self.K / self.tile.K)
        num_tiles_N = ceil(self.N / self.tile.N)

        max_reload = num_tiles_M * num_tiles_N * num_tiles_K

        l2_shared = self.tile.shared_accesses()
        mem_accesses[1] = l2_shared * max_reload

        factor_M, factor_K, factor_N = self.tile.tile_factors()
        inner = order_dims[2]  # most inner loop
        if inner == "m": # ws
            mk_load = max_reload
            kn_load = num_tiles_K * num_tiles_N
            mn_load = max_reload
            factor_M = 1
            df = "ws"
        elif inner == "k": # os
            mk_load = max_reload
            kn_load = max_reload
            mn_load = num_tiles_M * num_tiles_N
            factor_K = 1
            df = "os"
        elif inner == "n": # as
            mk_load = num_tiles_M * num_tiles_K
            kn_load = max_reload
            mn_load = max_reload
            factor_N = 1
            df = "as"
        else:
            raise NotImplementedError()
        
        mem_accesses[2] = (
            mk_load * self.tile.mk_bytes * factor_N
            + kn_load * self.tile.kn_bytes * factor_M
            + 2 * mn_load * self.tile.mn_bytes * factor_K
        )

        mem_accesses[3] = self.dram_accesses(df)

        return mem_accesses
    
    def per_layer_capacity(self):
        tile = self.tile
        util = []
        while tile:
            util.append(tile.util_capacity)
            tile = tile.tile
        return util


class L2Tile(Tile):
    def __init__(self, tile_dims, level, num_bundle, memLayer, dtype_size):
        super().__init__(tile_dims, level, memLayer, dtype_size)
        self.num_bundle = num_bundle

    # def __repr__(self):
    #     return (
    #         super().__repr__() + f"  mk_read: {formatBytes(self.mk_read_bytes)}\n"
    #         f"  kn_read: {formatBytes(self.kn_read_bytes)}\n"
    #         f"  mn_read: {formatBytes(self.mn_read_bytes)}\n"
    #         f"  mn_write: {formatBytes(self.mn_write_bytes)}\n"
    #         f"{self.tile.__repr__()}"
    #     )

    def get_tile(self, tile_dims):
        return L1Tile(tile_dims, self.level - 1, self.memLayer, self.dtype_size)
    
    def tile_factors(self):
        num_tiles_M = ceil(self.M / self.tile.M)
        num_tiles_K = ceil(self.K / self.tile.K)
        num_tiles_N = ceil(self.N / self.tile.N)
        return num_tiles_M, num_tiles_K, num_tiles_N

    def shared_accesses(self):
        """Trace through tile accesses to L1 shared memory"""

        reuse_M = ceil(self.M / self.tile.M)
        reuse_K = ceil(self.K / self.tile.K)
        reuse_N = ceil(self.N / self.tile.N)

        # effective number of tiles that can be processed in parallel
        # eff_sm = min(self.num_bundle, reuse_M * reuse_K * reuse_N)

        # track bytes accessed from shared memory per sm
        read_bytes = (self.tile.mk_bytes + self.tile.kn_bytes) * (
            reuse_M * reuse_N * reuse_K
        )  # / eff_sm
        write_bytes = self.tile.mn_bytes * (reuse_M * reuse_N)  # / eff_sm

        return read_bytes + write_bytes


class L1Tile(Tile):
    def __init__(self, tile_dims, level, memLayer, dtype_size):
        super().__init__(tile_dims, level, memLayer, dtype_size)

    def __repr__(self):
        return f"({self.level}) {self.__class__.__name__} {self.M}x{self.K}x{self.N}\n"


def formatBytes(size):
    """Format bytes into a human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TB"
