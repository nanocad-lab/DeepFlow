from perf import TimeCalculation
from tile import formatBytes
import config


def main():
    exp_config = config.parse_config("configs/new-configs/a100_80GB.yaml")
    TC = TimeCalculation(exp_config)
    TC.debug = True
    '''
    with open("matmul_DF.csv", "w") as f:
        # M = 8192
        # for exp in range(5, 16):
        #     K = 2 ** exp
        #     N = K

        #     gemm_time,_,_,_, gemm = TC.getGEMMTime(M, K, N, "")
        #     tflops = 2 * M * N * K / gemm_time / 1e12

        #     f.write(f"{M}, {N}, {K}, {gemm_time*1e3:.4f}ms, {tflops:.4f}Tflops\n")

        l2_tiles = [
            (64, 64, 128),
            (128, 64, 256),
            (128, 64, 256),
            (128, 64, 256),
            (128, 32, 128),
            (256, 32, 128),
            (128, 32, 256),
            (256, 64, 128),
            (256, 64, 128),
            (256, 64, 128),
        ]

        N = 12288
        K = N
        for exp in range(10,11):
            M = 2**exp
            TC.l2_tile = l2_tiles[exp - 6]
            # TC.debug = True

            gemm_time, _, _, _, gemm = TC.getGEMMTime(M, K, N, f"{M}x{K}x{N}")
            tflops = 2 * M * N * K / gemm_time / 1e12

            TC.roofline(gemm.GEMM_flop(), gemm.mem_accesses(), name=f"{M}x{K}x{N}", info=True)
            
            f.write(
                f"{M}, {N}, {K}, {gemm_time * 1e3:.4f}ms, {tflops:.4f}Tflops\n"
            )
    '''
    M, K, N = 8192, 4096, 4096
    TC.l2_tile = (128, 32, 128)
    # TC.debug = True

    gemm_time, _, _, _, gemm = TC.getGEMMTime(M, K, N, f"{M}x{K}x{N}")
    tflops = 2 * M * N * K / gemm_time / 1e12

    TC.roofline(gemm.GEMM_flop(), gemm.mem_accesses(), name=f"{M}x{K}x{N}", info=True)
            
    print(
        f"{M}, {N}, {K}, {gemm_time * 1e3:.4f}ms, {tflops:.4f}Tflops\n"
    )

if __name__ == "__main__":
    main()
