from math import ceil
import numpy as np
import copy

class Tile:
    def __init__(self, tile_dims, level, dtype_size):
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
        self.tile = self.get_tile(tile_dims)
        self.total_bytes = self.mk_bytes() + self.kn_bytes() + self.mn_bytes()

    def __repr__(self):
        return (
            f"({self.level}) {self.__class__.__name__} {self.M}x{self.K}x{self.N}\n"
            f"  total bytes: {formatBytes(self.total_bytes)}\n"
            f"      mk_bytes: {formatBytes(self.mk_bytes())}, kn_bytes: {formatBytes(self.kn_bytes())}, mn_bytes: {formatBytes(self.mn_bytes())}\n"
        )

    def mk_bytes(self):
        """Returns the size of A matrix in bytes (shape: M x K)"""
        return self.M * self.K * self.dtype_size

    def kn_bytes(self):
        """Returns the size of B matrix in bytes (shape: K x N)"""
        return self.K * self.N * self.dtype_size

    def mn_bytes(self):
        """Returns the size of C matrix in bytes (shape: M x N)"""
        return self.M * self.N * self.dtype_size

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

    def __init__(self, order_dims, tile_dims, core, dtype_size=2):
        self.num_bundle = core.num_bundle
        super().__init__(tile_dims, 3, dtype_size)
        self.order_dims = order_dims
        self.FMA_x, self.FMA_y = core.FMA_dims
        self.dataflow = core.dataflow
        self.count = []
        self.mem_read, self.mem_write = self.simulate_accesses(order_dims)

    def __repr__(self):
        return (
            super().__repr__()
            + f"  DRAM read: {formatBytes(self.mem_read[3])}, write: {formatBytes(self.mem_write[3])}\n"
            f"  L2 read: {formatBytes(self.mem_read[2])}, write: {formatBytes(self.mem_write[2])}\n"
            f"  Shared read: {formatBytes(self.mem_read[1])}, write: {formatBytes(self.mem_write[1])}\n"
            f"  Reg read: {formatBytes(self.mem_read[0])}, write: {formatBytes(self.mem_write[0])}\n"
            f"  loop order: {self.order_dims}\n"
            f"{self.tile.__repr__()}"
        )

    def print_count(self):
        return f"read mk: {self.count[0]}, kn: {self.count[1]}, mn: {self.count[2]}, write mn: {self.count[3]}"

    def print_bytes(self):
        return f"{formatBytes(self.mem_read[3])}, {formatBytes(self.mem_write[3])}, {formatBytes(self.mem_read[2])}, {formatBytes(self.mem_write[2])}, {formatBytes(self.mem_read[1])}, {formatBytes(self.mem_write[1])}"

    def mem_accesses(self):
        return [r + w for r, w in zip(self.mem_read, self.mem_write)]

    def GEMM_flop(self):
        return self.M * self.N * (2 * self.K - 1)

    def get_tile(self, tile_dims):
        return L2Tile(tile_dims, self.level-1, self.num_bundle, self.dtype_size)
    
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
            self.GEMM_flop()
            * 2
            * self.FMA_y
            / (self.FMA_x * (2 * self.FMA_y - 1))
            * self.dtype_size
        )
        store_bytes = (
            self.GEMM_flop() / (reuse * (2 * self.FMA_y - 1)) * self.dtype_size
        )

        return load_bytes, store_bytes

    def simulate_accesses(self, order_dims):
        """Trace through tile accesses to each level of memory for a given loop ordering"""
        read_accesses = [0] * 4
        write_accesses = [0] * 4

        read_accesses[0], write_accesses[0] = self.sysarray_accesses()

        num_tiles_M = ceil(self.M / self.tile.M)
        num_tiles_N = ceil(self.N / self.tile.N)
        num_tiles_K = ceil(self.K / self.tile.K)

        max_reload = num_tiles_M * num_tiles_N * num_tiles_K

        l2_shared = self.tile.shared_accesses()
        read_accesses[1] = l2_shared[0] * max_reload
        write_accesses[1] = l2_shared[1] * max_reload

        inner = order_dims[2]  # most inner loop
        if inner == "m":
            mk_load = max_reload
            kn_load = num_tiles_K * num_tiles_N
            mn_load = max_reload
        elif inner == "k":
            mk_load = max_reload
            kn_load = max_reload
            mn_load = num_tiles_M * num_tiles_N
        elif inner == "n":
            mk_load = num_tiles_M * num_tiles_K
            kn_load = max_reload
            mn_load = max_reload
        read_accesses[2] = (
            mk_load * self.tile.mk_read_bytes
            + kn_load * self.tile.kn_read_bytes
            + mn_load * self.tile.mn_read_bytes
        )
        write_accesses[2] = mn_load * self.tile.mn_write_bytes

        # TODO: model DRAM accesses
        read_accesses[3] = self.mk_bytes() + self.kn_bytes() # input matrix compulsory cache misses only
        # write_accesses[3] = self.kn_bytes() # output matrix cache misses

        self.count = [mk_load, kn_load, mn_load, mn_load]

        return read_accesses, write_accesses


class L2Tile(Tile):
    def __init__(self, tile_dims, level, num_bundle, dtype_size):
        super().__init__(tile_dims, level, dtype_size)
        self.num_bundle = num_bundle
        self.mk_read_bytes = self.mk_bytes()
        self.kn_read_bytes = self.kn_bytes()
        self.mn_read_bytes = self.mn_bytes()
        self.mn_write_bytes = self.mn_bytes()


    def __repr__(self):
        return (
            super().__repr__() + f"  mk_read: {formatBytes(self.mk_read_bytes)}\n"
            f"  kn_read: {formatBytes(self.kn_read_bytes)}\n"
            f"  mn_read: {formatBytes(self.mn_read_bytes)}\n"
            f"  mn_write: {formatBytes(self.mn_write_bytes)}\n"
            f"{self.tile.__repr__()}"
        )

    def get_tile(self, tile_dims):
        return L1Tile(tile_dims, self.level - 1, self.dtype_size)

    def shared_accesses(self):
        """Trace through tile accesses to L1 shared memory"""

        reuse_M = ceil(self.M / self.tile.M)
        reuse_K = ceil(self.K / self.tile.K)
        reuse_N = ceil(self.N / self.tile.N)
        
        # effective number of tiles that can be processed in parallel
        eff_sm = min(self.num_bundle, reuse_M * reuse_K * reuse_N)

        # track bytes accessed from shared memory per sm
        read_bytes = (self.tile.mk_bytes() + self.tile.kn_bytes()) * (reuse_M * reuse_N * reuse_K) / eff_sm
        write_bytes = self.tile.mn_bytes() * (reuse_M * reuse_N) / eff_sm

        return read_bytes, write_bytes
    
class L1Tile(Tile):
    def __init__(self, tile_dims, level, dtype_size):
        super().__init__(tile_dims, level, dtype_size)


def formatBytes(size):
    """Format bytes into a human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TB"
