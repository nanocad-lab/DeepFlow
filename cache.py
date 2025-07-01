from math import ceil

word_size = 2  # in bytes
line_size = 8 * word_size
cache_size = 80 * 1024**2
num_blocks = cache_size // line_size
tile_size = (128, 32, 128)
M, K, N = 1024, 12288, 12288


"""
-- direct-mapped cache --
A(M, K)
    K/8 cache blocks per row
    element A[i][k] will map to block [(i*K/8) % num_blocks]
B(K, N)
    N/8 cache blocks per row
    element B[k][j] will map to block [(k*N/8) % num_blocks]
"""


def format_bytes(size):
    """Format bytes into a human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TB"


def main():
    accesses = [0] * 3  # A, B, C
    A_miss = (
        M * ceil(tile_size[1] * word_size / line_size) * (K // tile_size[1])
    )  # A only has cold misses
    B_miss = 0
    '''
    # Simulate tiled matrix multiplication cache misses
    for kk in range(0, K, tile_size[1]):
        for jj in range(0, N, tile_size[2]):
            B_miss += ceil(tile_size[2] * word_size / line_size) * min(
                tile_size[1], K - kk
            )  # assume B only as cold misses
            for i in range(M):
                for k in range(kk, min(kk + tile_size[1] - 1, K)):
                    accesses[0] += 1  # A access
                    for j in range(jj, min(jj + tile_size[2] - 1, N)):
                        accesses[1] += 1  # B access
                        accesses[2] += 1  # C access
    '''
    print(f"-- {M}x{K}x{N} ({tile_size[0]}x{tile_size[1]}x{tile_size[2]}) --")
    print(
        f"Matrix sizes: MK={format_bytes(M * K * word_size)}, "
        f"KN={format_bytes(K * N * word_size)}, "
        f"MN={format_bytes(M * N * word_size)}"
    )
    print(
        f"Tile sizes: mk={format_bytes(tile_size[0] * tile_size[1] * word_size)}, "
        f"kn={format_bytes(tile_size[1] * tile_size[2] * word_size)}, "
        f"mn={format_bytes(tile_size[0] * tile_size[2] * word_size)}\n"
    )

    print(f"A will cold miss every {line_size // word_size} accesses")
    print(f"A total misses: {A_miss}\n")
    print(f"B total misses: {B_miss}\n")

    print(f"total DRAM read: {format_bytes(A_miss * line_size + B_miss * line_size)}")


if __name__ == "__main__":
    main()
