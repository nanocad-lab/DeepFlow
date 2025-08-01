
import math
import os
import pandas as pd

def reshape_gemm_to_3d(arg,options=None):
    """
    Reshape a 4-dimensional GEMM [batch_size, M, K, N] into 3 dimensions [M, K, N].

    Parameters:
        arg (list or tuple): A list or tuple containing 4 dimensions [batch_size, M, K, N].
        options (str): If set to "multiply_batch_into_m", multiplies batch_size into M.
        "distribute_batch_cube_root": Takes the cube root of batch_size and distributes it across M, K, and N.

    Returns:
        tuple: A tuple (M, K, N) representing the reshaped GEMM dimensions.
    """
    
    if len(arg) != 4:
        raise ValueError("Input must contain exactly 4 dimensions [batch_size, M, K, N].")
    
    
    batch_size, M, K, N = arg
    if batch_size <= 0:
        raise ValueError("Batch size must be greater than 0.")
    if options == "multiply_batch_into_m":
        M *= batch_size  # Multiply batch_size into M
    elif options == "distribute_batch_cube_root":
        c = int(round(batch_size ** (1/3)))  # Start with the cube root of batch_size
        while batch_size % c != 0:  # Adjust c until batch_size is divisible by c
            c -= 1
        remaining = batch_size // c  # Remaining product after dividing by c
        
        b = int(round(remaining ** 0.5))  # Start with the square root of the remaining product
        while remaining % b != 0:  # Adjust b until remaining is divisible by b
            b -= 1
        a = remaining // b  # Calculate a
        # print(f"Distributing batch_size {batch_size} into M, K, N with factors: a={a}, b={b}, c={c}")
        # Distribute the factors across M, K, and N
        M *= a
        K *= b
        N *= c
    
        
    return M, K, N


def multihead_decoder_gemm(batch_size, seq_len, d_model, num_heads, ffn_dim):
    """
    Generate GEMM shapes [M, K, N] for a multi-head Transformer decoder block.

    Parameters:
        batch_size (int): batch size (B)
        seq_len (int): sequence length (S)
        d_model (int): hidden size (D)
        num_heads (int): number of attention heads (H)
        ffn_dim (int): first FFN layer output dimension (typically 4 * D)

    Returns:
        List of tuples: (level_name, GEMM spec), where GEMM spec is either
            - [M, K, N] for standard GEMM
            - ["batched", A_shape, B_shape] for batched GEMM
    """
    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
    head_dim = d_model // num_heads
    gemms = []
    levels = [
        "Q/K/V projection",
        "Q x K^T (attention scores)",
        "score x V (attention output)",
        "output projection",
        "FFN layer 1",
        "FFN layer 2"
    ]

    # 1. Input projection
    gemms.append([batch_size , seq_len, d_model, 3 * d_model])

    # 2. Attention score
    gemms.append([batch_size * num_heads, seq_len,  head_dim, seq_len])

    # 3. Attention output
    gemms.append([ batch_size * num_heads, seq_len, seq_len, head_dim])

    # 4. Output projection
    gemms.append([batch_size , seq_len, d_model, d_model])

    # 5. FFN layer 1
    gemms.append([batch_size , seq_len, d_model, ffn_dim])

    # 6. FFN layer 2
    gemms.append([batch_size , seq_len, ffn_dim, d_model])

    return list(zip(levels, gemms))


def process_gemm_shapes(batch_size, seq_len, d_model, num_heads, ffn_dim, output_file="mat_dims_llm.txt",option="multiply_batch_into_m"):
    """
    Process GEMM shapes, reshape them, and write both 4D and 3D GEMM shapes to a file.

    Parameters:
        batch_size (int): Batch size.
        seq_len (int): Sequence length.
        d_model (int): Hidden size.
        num_heads (int): Number of attention heads.
        ffn_dim (int): First FFN layer output dimension.
        output_file (str): File to write the GEMM shapes.
    """
    # Generate GEMM shapes in 4D
    gemm_shapes_4d = multihead_decoder_gemm(batch_size, seq_len, d_model, num_heads, ffn_dim)

    # Reshape GEMM shapes to 3D
    gemm_3d = [reshape_gemm_to_3d(shape, option) for _, shape in gemm_shapes_4d]

    # Write both 4D and 3D GEMM shapes to the same file
    with open(output_file, "w") as f:
        # Write 4D GEMM shapes
        for i, (level, shape) in enumerate(gemm_shapes_4d):
            f.write(f"Layer {i + 1} ({level}): 4D GEMM shape = {shape}\n")
            # print(f"Layer {i + 1} ({level}): 4D GEMM shape = {shape}")  # Print to console as well
        
        # Write 3D GEMM shapes
        f.write("\nGEMM shapes in 3D format:\n")
        for i, (level, shape) in enumerate(zip(gemm_shapes_4d, gemm_3d)):
            f.write(f"Layer {i + 1} ({level}): {shape}\n")
            # print(f"Layer {i + 1} ({level}): {shape}") 
    return gemm_shapes_4d, gemm_3d

def caltime(N_L, B, S, ntokens, comm_time, N_PP, directory, output_dir):

    
    # Directory containing the files
    # directory = "output/Trans/"
    
    # os.makedirs(output_dir, exist_ok=True)
     # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Define the output log file
    log_file = os.path.join(output_dir, "summary_LLM.txt")
    
    
    t_elapsed =0.0
    # print("Time spent in different GEMMs")
    
    
    with open(log_file, "w") as log:
        t_elapsed = 0.0
        log.write("Time spent in different GEMMs:\n")
        print("Time spent in different GEMMs")

        # Loop through files in the directory
        for filename in os.listdir(directory):
            if filename.endswith(".txt") and filename != "summary_LLM.txt":  # Exclude the log file
                file_path = os.path.join(directory, filename)

                data = pd.read_csv(file_path, sep=':', header=None, names=['Field', 'Value'], skipinitialspace=True)
                # Extract the time value
                time_row = data[data['Field'] == 'Time']
                if not time_row.empty:
                    time_value = float(time_row['Value'].iloc[0])
                    print(time_value)
                    log.write(f"{filename}: {time_value} seconds\n")
                else:
                    print("Time value not found in the file.")
                
                t_elapsed += time_value

        t_elapsed *= 3.0  # FW pass + BW pass (~2x FW pass)
        comp_time = t_elapsed
        nbatch = ntokens / (S * B)
        time = N_L * nbatch * t_elapsed / N_PP + comm_time

        # Log and print the results
        log.write(f"\nTotal computation time (FW + BW): {comp_time} seconds\n")
        log.write(f"Number of tokens: {ntokens}\n")
        log.write(f"Time to exhaust all tokens: {time} seconds ({time / 3600.0 / 24.0} days)\n")

    
    
    # Loop through files in the directory
    # for filename in os.listdir(directory):
    #     if filename.endswith(".txt"):  # You can adjust the file extension as needed
    #         file_path = os.path.join(directory, filename)

    #         data = pd.read_csv(file_path, sep=':', header=None, names=['Field', 'Value'], skipinitialspace=True)
    #         # Extract the time value
    #         time_row = data[data['Field'] == 'Time']
    #         if not time_row.empty:
    #             time_value = float(time_row['Value'].iloc[0])
    #             print(time_value)
    #         else:
    #             print("Time value not found in the file.")
            
    #         t_elapsed += time_value

    # t_elapsed = t_elapsed*3.0 #FW pass + BW pass (~ 2x FW pass)
    # comp_time = t_elapsed
    # #comm_time = 8.85 # (hours) comes from AMPED
    # nbatch = ntokens/(S*B)
    # time = N_L*nbatch*t_elapsed/N_PP + comm_time

    print("number of tokens:", ntokens, " | time to exhaust all tokens:", time, "(s)", " or ", time/3600.0/24.0, " days")
    print("Performance Results written to {}".format(log_file))

