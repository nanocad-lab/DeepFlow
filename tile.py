from math import ceil
import numpy as np
import copy

class Mapping:
    def __init__(self, M, N, K, l2_M, l2_N, l2_K, l1_M, l1_N, l1_K, FMA_M, FMA_N, FMA_K, l2_loop_order, l1_loop_order):
        """
        Represents a hierarchical mapping of GEMM tiling.

        Parameters:
        - M, N, K: Global GEMM dimensions
        - l2_M, l2_N, l2_K: Tile sizes at L2 level
        - l1_M, l1_N, l1_K: Tile sizes at L1 level
        - FMA_M, FMA_N, FMA_K: GEMM sizes performed by systolic array
        - l2_loop_order, l1_loop_order
        """
        self.M = M
        self.N = N
        self.K = K
        self.l2_M = l2_M
        self.l2_N = l2_N
        self.l2_K = l2_K
        self.l1_M = l1_M
        self.l1_N = l1_N
        self.l1_K = l1_K
        self.FMA_M = FMA_M
        self.FMA_N = FMA_N
        self.FMA_K = FMA_K

        self.l2_loop_order = l2_loop_order
        self.l1_loop_order = l1_loop_order

    def __repr__(self):
        return (f"Mapping(GEMM=({self.M}, {self.N}, {self.K}), "
                f"L2=({self.l2_M}, {self.l2_N}, {self.l2_K}), "
                f"L1=({self.l1_M}, {self.l1_N}, {self.l1_K}))")

class Tile:
    def __init__(self, tile_dims, level, dtype_size):
        """
        Base class for tile transfer in GEMM across memory hierarchy.

        Parameters:
        - M, N, K: Full GEMM problem dimensions (C = A x B, where A is MxK, B is KxN)
        - tile_M, tile_N, tile_K: Tile sizes along M, N, K dimensions
        - dtype_size: Size of one element in bytes (default is 2 bytes for fp16)
        """

        assert level > 0, "Invalid tile level"

        self.level = level
        self.M, self.K, self.N = tile_dims[level]
        self.tile_M, self.tile_K, self.tile_N = tile_dims[level-1]
        self.dtype_size = dtype_size
        self.tile = self.get_tile(tile_dims)

    def __repr__(self):
        return (
            f"({self.level}) {self.__class__.__name__} {self.M}x{self.K}x{self.N}\n"
            f"  total bytes: {formatBytes(self.mk_bytes() + self.kn_bytes() + self.mn_bytes())}\n"
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
        # assume partial tiles are padded to be full?
        if self.level == 3:
            tileClass = L2Tile
        elif self.level == 2:
            tileClass = L1Tile
        else:
            return None
            
        return tileClass(tile_dims, self.level-1, self.dtype_size)

    def generate_tile_loops(self, loop_M: int, loop_N: int, loop_K: int, loop_order: str):
        assert loop_order in ["mkn", "mnk", "nkm", "nmk", "knm", "kmn"]
        if loop_order == "mnk":
            for m in range(loop_M):
                for n in range(loop_N):
                    for k in range(loop_K):
                        yield m, n, k
        elif loop_order == "mkn":
            for m in range(loop_M):
                for k in range(loop_K):
                    for n in range(loop_N):
                        yield m, n, k
        elif loop_order == "nmk":
            for n in range(loop_N):
                for m in range(loop_M):
                    for k in range(loop_K):
                        yield m, n, k
        elif loop_order == "nkm":
            for n in range(loop_N):
                for k in range(loop_K):
                    for m in range(loop_M):
                        yield m, n, k
        elif loop_order == "knm":
            for k in range(loop_K):
                for n in range(loop_N):
                    for m in range(loop_M):
                        yield m, n, k
        elif loop_order == "kmn":
            for k in range(loop_K):
                for m in range(loop_M):
                    for n in range(loop_N):
                        yield m, n, k

class TiledGEMM(Tile):
    def __init__(self, tile_dims, order_dims, dtype_size=2):
        super().__init__(tile_dims, 3, dtype_size)
        self.mem_read, self.mem_write = self.simulate_accesses(order_dims)
        self.order_dims = order_dims

    def __repr__(self):
        return (
            super().__repr__() +
            f"  L2 read: {formatBytes(self.mem_read[2])}, write: {formatBytes(self.mem_write[2])}\n"
            f"  Shared read: {formatBytes(self.mem_read[1])}, write: {formatBytes(self.mem_write[1])}\n"
            f"  loop order: {self.order_dims}\n"
            f"{self.tile.__repr__()}"
        )
    
    def simulate_accesses(self, order_dims):
        """Trace through tile accesses to each level of memory for a given loop ordering"""
        read_accesses = [0] * 4
        write_accesses = [0] * 4

        # total_read_bytes = self.tile_mk_bytes() + self.tile_kn_bytes()
        # total_write_bytes = 0

        # print(f"tile mk: {self.tile_mk_bytes()}, tile kn: {self.tile_kn_bytes()}, tile mn: {self.tile_mn_bytes()}")
        # print(f"l2 tile mk: {l2_tile.mk_read_bytes}, l2 tile kn: {l2_tile.kn_read_bytes}, l2 tile mn: {l2_tile.mn_read_bytes}, l2 tile mn write: {l2_tile.mn_write_bytes}")

        # read input tiles for first output tile
        read_accesses[2] += self.tile.mk_read_bytes + self.tile.kn_read_bytes

        prev_m, prev_n, prev_k = 0, 0, 0
        for m, n, k in self.generate_tile_loops(
            ceil(self.M / self.tile_M),
            ceil(self.N / self.tile_N),
            ceil(self.K / self.tile_K),
            order_dims,
        ):
            if m == 0 and n == 0 and k == 0:
                continue

            # read input tile if not already previously loaded
            if m == prev_m and k == prev_k:
                # total_read_bytes += self.tile_kn_bytes()
                read_accesses[2] += self.tile.kn_read_bytes
            elif k == prev_k and n == prev_n:
                # total_read_bytes += self.tile_mk_bytes()
                read_accesses[2] += self.tile.mk_read_bytes
            else:
                # total_read_bytes += self.tile_kn_bytes() + self.tile_mk_bytes()
                read_accesses[2] += self.tile.kn_read_bytes + self.tile.mk_read_bytes

            # replace previous output tile if not the same
            if not (m == prev_m and n == prev_n):
                # total_read_bytes += self.tile_mn_bytes()
                read_accesses[2] += self.tile.mn_read_bytes
                # total_write_bytes += self.tile_mn_bytes()
                write_accesses[2] += self.tile.mn_write_bytes

            l2_shared = self.tile.shared_accesses()
            read_accesses[1] += l2_shared[0]
            write_accesses[1] += l2_shared[1]

            prev_m, prev_n, prev_k = m, n, k

        # total_write_bytes += self.tile_mn_bytes()
        write_accesses[2] += self.tile.mn_write_bytes

        return read_accesses, write_accesses

class L2Tile(Tile):
    def __init__(self, tile_dims, level, dtype_size):
        super().__init__(tile_dims, level, dtype_size)
        self.num_mcu = 108 # A100

        self.mk_read_bytes, self.kn_read_bytes, self.mn_read_bytes, self.mn_write_bytes = self.simulate_accesses()
    
    def __repr__(self):
        return (
            super().__repr__() +
            f"{self.tile.__repr__()}"
        )

    def shared_accesses(self):
        """Trace through tile accesses to L1 shared memory"""

        reuse_M = ceil(self.M / self.tile_M)
        reuse_K = ceil(self.K / self.tile_K)
        reuse_N = ceil(self.N / self.tile_N)
        # assume perfect reuse within tile, only
        # track bytes accessed from shared memory
        read_bytes = (self.tile.mk_bytes() + self.tile.kn_bytes()) * (reuse_M * reuse_N * reuse_K)
        write_bytes = self.tile.mn_bytes() * (reuse_M * reuse_N)

        return read_bytes, write_bytes

    def simulate_accesses(self):
        """Trace through tile accesses to L2 for a given loop ordering"""
        # bytes read from each tile
        mk_read_bytes = 0
        kn_read_bytes = 0
        mn_read_bytes = 0
        mn_write_bytes = 0

        tile_mk_read = np.zeros(
            [ceil(self.M / self.tile_M), ceil(self.K / self.tile_K)], dtype=bool
        )
        tile_kn_read = np.zeros(
            [ceil(self.K / self.tile_K), ceil(self.N / self.tile_N)], dtype=bool
        )
        tile_mn_read = np.zeros(
            [ceil(self.M / self.tile_M), ceil(self.N / self.tile_N)], dtype=bool
        )
        tile_mn_write = np.zeros(
            [ceil(self.M / self.tile_M), ceil(self.N / self.tile_N)], dtype=bool
        )

        prev_mk_read = copy.deepcopy(tile_mk_read)
        prev_kn_read = copy.deepcopy(tile_kn_read)
        prev_mn_read = copy.deepcopy(tile_mn_read)
        prev_mn_write = copy.deepcopy(tile_mn_write)

        active_mcu = 0
        for m, n, k in self.generate_tile_loops(
            ceil(self.M / self.tile_M),
            ceil(self.N / self.tile_N),
            ceil(self.K / self.tile_K),
            "mkn",
        ):
            active_mcu += 1
            tile_mk_read[m, k] = ~prev_mk_read[m, k]
            tile_kn_read[k, n] = ~prev_kn_read[k, n]
            tile_mn_read[m, n] = 1
            tile_mn_write[m, n] = 1

            if active_mcu >= self.num_mcu or (
                m == ceil(self.M / self.tile_M) - 1 
                and n == ceil(self.N / self.tile_N) - 1 
                and k == ceil(self.K / self.tile_K) - 1
            ):
                mk_read_bytes += np.sum(tile_mk_read) * self.tile.mk_bytes()
                kn_read_bytes += np.sum(tile_kn_read) * self.tile.kn_bytes()
                mn_read_bytes += np.sum(tile_mn_read * ~(prev_mn_read[m, n] + prev_mn_write[m, n])) * self.tile.mn_bytes()

                mn_write_bytes += np.sum(prev_mn_write * (~tile_mn_read)) * self.tile.mn_bytes()
            
                active_mcu = 0

                prev_mk_read = copy.deepcopy(tile_mk_read)
                prev_kn_read = copy.deepcopy(tile_kn_read)
                prev_mn_read = copy.deepcopy(tile_mn_read)
                prev_mn_write = copy.deepcopy(tile_mn_write)

                tile_mk_read = np.zeros(
                    [ceil(self.M / self.tile_M), ceil(self.K / self.tile_K)], dtype=bool
                )
                tile_kn_read = np.zeros(
                    [ceil(self.K / self.tile_K), ceil(self.N / self.tile_N)], dtype=bool
                )
                tile_mn_read = np.zeros(
                    [ceil(self.M / self.tile_M), ceil(self.N / self.tile_N)], dtype=bool
                )
                tile_mn_write = np.zeros(
                    [ceil(self.M / self.tile_M), ceil(self.N / self.tile_N)], dtype=bool
                )
        mn_write_bytes += np.sum(prev_mn_write * (~tile_mn_read)) * self.tile.mn_bytes()

        return mk_read_bytes, kn_read_bytes, mn_read_bytes, mn_write_bytes

class L1Tile(Tile):
    def __init__(self, tile_dims, level, dtype_size):
        super().__init__(tile_dims, level, dtype_size)

def formatBytes(bytes):
    unit = ""
    if bytes < 1024:
        unit = "B"
    elif bytes < 1024 ** 2:
        unit = "KB"
        bytes /= 1024
    elif bytes < 1024 ** 3:
        unit = "MB"
        bytes /= 1024 ** 2
    else:
        unit = "GB"
        bytes /= 1024 ** 3
    return f'{bytes} {unit}'

def main():
    l2_M = [32, 32, 128, 128, 128, 128, 128, 128, 128]
    l2_N = [32, 32, 128, 256, 128, 128, 128, 128, 256]
    l2_K = [32, 32, 32, 32, 32, 32, 32, 32, 64]
    
    tile_dims = [(0,0,0)] * 4
    tile_dims[0] = (8, 4, 8)

    M = 8192
    for n in range(5,14):
        N = 2 ** n
        K = N
        tile_dims[3] = (M, K, N)
        tile_dims[2] = (l2_M[n-5], l2_K[n-5], l2_N[n-5])
        min_bytes_accessed = 2**63 - 1
        for l2_order in ["mkn", "mnk", "nkm", "nmk", "knm", "kmn"]:
            for l1_M in [16, 32, 64, 128, 256]:
                if l1_M > min(l2_M[n-5], l2_N[n-5]):
                    continue
                l1_N = l1_M
                for l1_K_tiling_factor in [1, 2, 4, 8, 16, 32]:
                    l1_K = ceil(l2_K[n-5] / l1_K_tiling_factor)
                    tile_dims[1] = (l1_M, l1_K, l1_N)
                    gemm = TiledGEMM(tile_dims, l2_order)
                    if min_bytes_accessed > gemm.mem_read[2] + gemm.mem_write[2]:
                        min_bytes_accessed = gemm.mem_read[2] + gemm.mem_write[2]
                        best_gemm = gemm
        print(repr(best_gemm) + "\n")
                
if __name__ == "__main__":
    main()