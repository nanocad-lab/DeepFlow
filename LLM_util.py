
import math
import os
import pandas as pd
import config

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


def multihead_decoder_gemm(batch_size, seq_len, d_model, num_heads, ffn_dim, vocab_size):
    """
    Generate GEMM shapes [M, K, N] for a multi-head Transformer decoder block.

    Parameters:
        batch_size (int): batch size (B)
        seq_len (int): sequence length (S)
        d_model (int): hidden size (D)
        num_heads (int): number of attention heads (H)
        ffn_dim (int): first FFN layer output dimension (typically 4 * D)
        vocab_size (int): vocabulary size (V)


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
        "FFN layer 2",
        "linear layer"
    ]

    # 1. Input projection
    gemms.append([batch_size , seq_len, d_model, 3 * d_model])

    # 2. Attention score
    gemms.append([batch_size * num_heads, seq_len,  head_dim, seq_len])

    # 3. Attention output
    gemms.append([ batch_size * num_heads, seq_len, seq_len, head_dim])#todo: check if this is correct

    # 4. Output projection
    gemms.append([batch_size , seq_len, d_model, d_model])

    # 5. FFN layer 1
    gemms.append([batch_size , seq_len, d_model, ffn_dim])

    # 6. FFN layer 2
    gemms.append([batch_size , seq_len, ffn_dim, d_model])

    # 7. linear layer
    gemms.append([batch_size , seq_len, d_model, vocab_size])

    return list(zip(levels, gemms))


def process_gemm_shapes(batch_size, seq_len, d_model, num_heads, ffn_dim, vocab_size, option="multiply_batch_into_m"):
    """
    Process GEMM shapes, reshape them into 3d.

    Parameters:
        batch_size (int): Batch size.
        seq_len (int): Sequence length.
        d_model (int): Hidden size.
        num_heads (int): Number of attention heads.
        ffn_dim (int): First FFN layer output dimension.
        vocab_size (int): Vocabulary size.
    """
    # Generate GEMM shapes in 4D
    gemm_shapes_4d = multihead_decoder_gemm(batch_size, seq_len, d_model, num_heads, ffn_dim, vocab_size)

    # Reshape GEMM shapes to 3D
    gemm_3d = [reshape_gemm_to_3d(shape, option) for _, shape in gemm_shapes_4d]


    return  gemm_3d

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


def getTransformerMem_layer( d, t, batch_size, hidden_dim, seq_len, ffn_dim, n_heads, precision):#https://www.determined.ai/blog/act-mem-1.  https://arxiv.org/pdf/2205.05198. https://shjwudp.github.io/blog/2023/gpt-training-memory-estimation-nemo-training-practice/?utm_source=chatgpt.com
    #Activations refer to output activations that need to be stored
    alpha = 16 + ffn_dim/hidden_dim #parameter need to be changed accordingly
    beta = 3
    # d = 1# data parallelism degree
    # t = 1 #tensor parallelism degree
    attention_score_act = batch_size * n_heads * seq_len * seq_len
    act_memory_layer = (alpha * batch_size * hidden_dim * seq_len + beta * attention_score_act) * precision / 2  # assuming stored in a 16-bit floating point according to paper

    transformer_param_layer = (4 ) * hidden_dim * hidden_dim + ffn_dim * 2 * hidden_dim  # weights Wq,Wk,Wv,Wo,ffn1,ffn2
    optimizer_mem = 10 * transformer_param_layer * precision/ 2 / (t * d) 
    weight_memory_layer = 2 * transformer_param_layer * precision / t / 2  # assuming stored in a 16-bit floating point according to paper
    gradient_mem = 4 * transformer_param_layer * precision / t / 2  # assuming stored in a 16-bit floating point according to paper
    static_memory_layer = (6 + 10 / d) * transformer_param_layer / t * precision / 2  


    layer_mem = (act_memory_layer + weight_memory_layer)
    #cross entropy not included

    return layer_mem, act_memory_layer, static_memory_layer, gradient_mem, optimizer_mem, weight_memory_layer

def getlinearSoftmaxMem(batch_size, seq_len, hidden_dim, vocab_size, precision, t):
    # t = 1
    # weights = hidden_dim * vocab_size
    # softmax_act = batch_size * seq_len * vocab_size * precision
    # softmax_wt = (hidden_dim + 1) * vocab_size * precision
    # softmax_point = (2 * batch_size * seq_len * vocab_size + batch_size * seq_len) * precision
    # #NOTE: sigmoid and exp could have been combined
    # #1 sigmoids
    # #1 exp
    # #1 pointwise div
    # softmax_mem = (softmax_act + softmax_wt + softmax_point)
    mem = 4 * seq_len * batch_size * hidden_dim / t *(1+vocab_size/hidden_dim) * precision / 2 #from https://arxiv.org/pdf/2205.05198
    return mem


def getEmbeddingMem(batch_size, seq_len, hidden_dim, p, t, precision):
    mem = 4 * seq_len * batch_size * hidden_dim * p / t * precision / 2  # from https://arxiv.org/pdf/2205.05198

    return mem

def getTotMemReq(exp_hw_config, exp_model_config, **kwargs):
    # Model Params
    batch_size                   = int(kwargs.get('batch_size', exp_model_config.model_config.batch_size))
    hidden_dim                   = int(kwargs.get('hidden_dim', exp_model_config.model_config.hidden_dim))
    vocab_size                   = int(kwargs.get('vocab_size', exp_model_config.model_config.vocab_size))
    n_layers                   = int(kwargs.get('num_layer', exp_model_config.model_config.num_layers))
    n_heads                     = int(kwargs.get('num_heads', exp_model_config.model_config.num_heads))
    # projection          = exp_model_config.model_config.projection
    seq_len                   = int(kwargs.get('seq_len', exp_model_config.model_config.seq_len))
    ffn_dim                   = int(kwargs.get('ffn_mult', exp_model_config.model_config.ffn_mult)) * hidden_dim if kwargs.get('ffn_mult', exp_model_config.model_config.ffn_mult) !=None else int(kwargs.get('ffn_dim', exp_model_config.model_config.ffn_dim))
    # G                   = exp_model_config.model_config.num_gates
    precision           = exp_hw_config.sw_config.precision

    # MiniBatch
    dp                  = int(kwargs.get('dp', exp_hw_config.sch_config.dp))
    # print("Data Parallelism Degree:", dp)
    dp = 8 #for testing
    miniB               = math.ceil(batch_size / dp)

    transformer_mem_layer, transformer_act_layer, transformer_static_layer, gradient_mem_layer, optimizer_mem_layer, weight_memory_layer = (
        getTransformerMem_layer(
            d = dp,
            t = 1,
            batch_size=batch_size,
            hidden_dim=hidden_dim,
            seq_len=seq_len,
            ffn_dim=ffn_dim,
            n_heads=n_heads,
            precision=precision,
        )
    )
    # print(f"transformer_mem per layer: {transformer_mem_layer/1e9:.2f} GB, act: {transformer_act_layer/1e9:.2f} GB, static: {transformer_static_layer/1e9:.2f} GB")
    # print(f"gradient_mem per layer: {gradient_mem_layer/1e9:.2f} GB, optimizer_mem per layer: {optimizer_mem_layer/1e9:.2f} GB, weight_memory_layer per layer: {weight_memory_layer/1e9:.2f} GB")
    softmax_mem = getlinearSoftmaxMem(
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        precision=precision,
        t=1
    )



    embedding_mem = getEmbeddingMem(
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        p=1,
        t=1,
        precision=precision
    )

    tot_mem = transformer_mem_layer*n_layers + softmax_mem + embedding_mem

    # wt_mem = (transformer_wt + softmax_wt + projection_wt + embedding_wt)
    # act_mem = (transformer_act + softmax_act + projection_act + embedding_act)
    # point_mem = (transformer_point + softmax_point + projection_point + embedding_point)
    

    return tot_mem, embedding_mem, transformer_mem_layer*n_layers,transformer_act_layer*n_layers,transformer_static_layer*n_layers, gradient_mem_layer*n_layers, optimizer_mem_layer*n_layers, weight_memory_layer*n_layers, softmax_mem#, projection_mem, wt_mem, act_mem, point_mem

if __name__ == "__main__":

    
    exp_hw_config_path = "configs/hardware-config/a100_80GB.yaml"
    exp_model_config_path = "configs/model-config/LLM.yaml"
    exp_hw_path = os.path.expandvars(os.path.expanduser(exp_hw_config_path))
    exp_model_path = os.path.expandvars(os.path.expanduser(exp_model_config_path))
    exp_hw_config = config.parse_config(exp_hw_path, config_type="hardware")
    exp_model_config = config.parse_config(exp_model_path, config_type="LLM")
    mem, embedding_mem, transformer_mem, transformer_act_mem, transformer_static_mem, gradient_mem, optimizer_mem, weight_memory, softmax_mem = getTotMemReq(
                exp_hw_config,
                exp_model_config,

            )
    print(f"Total Memory Requirement: {mem/1e9:.2f} GB")
    print(f"Embedding Memory Requirement: {embedding_mem/1e9:.2f} GB")
    print(f"Transformer Memory Requirement: {transformer_mem/1e9:.2f} GB")
    print(f"Transformer Activation Memory Requirement: {transformer_act_mem/1e9:.2f} GB")
    print(f"Transformer Static Memory Requirement(grad+optim+weight): {transformer_static_mem/1e9:.2f} GB")
    print(f"Transformer Gradient Memory Requirement: {gradient_mem/1e9:.2f} GB")
    print(f"Transformer Optimizer Memory Requirement: {optimizer_mem/1e9:.2f} GB")
    print(f"Transformer Weight Memory Requirement: {weight_memory/1e9:.2f} GB")
    print(f"Softmax Memory Requirement: {softmax_mem/1e9:.2f} GB")
