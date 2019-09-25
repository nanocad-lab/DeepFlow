import math
import sys

core=0.7
DRAM=0.7
L2=1
shared_mem=1
reg_mem=1

def printError(message):
  sys.exit(message)

def getHiddenMem(L, Dim1, Dim2, Dim3, S, precision):
    #Activations refer to output activations that need to be stored
    hidden_act = Dim1 * Dim2 * S * L
    hidden_wt  = (Dim2 + 1) * Dim3 * L
    hidden_point = (Dim1 * Dim2 / 2) * 9 * L * S
    #3 sigmoids
    #2 tanh
    #3 pointwise multiply
    #1 addition
    hidden_mem = (hidden_act + hidden_wt + hidden_point) * precision

    return hidden_mem, hidden_act, hidden_wt, hidden_point

def getSoftmaxMem(B, S, P, V, precision):
     #activation output from each layer, assuming input ativation are taken 
    #into account in the previous layer
    softmax_act = B * S * V 
    softmax_wt = (P + 1) * V
    softmax_point = 2 * B * S * V + B * S
    #NOTE: sigmoid and exp could have been combined
    #1 sigmoids
    #1 exp
    #1 pointwise div
    softmax_mem = (softmax_act + softmax_wt + softmax_point) * precision

    return softmax_mem, softmax_act, softmax_wt, softmax_point

def getProjectionMem(B, S, P, D, precision):
    projection_act = B * S * P
    projection_wt = (D + 1) * P
    projection_point= B * S * P
    projection_mem = (projection_act + projection_wt + projection_point) * precision
  
    return projection_mem, projection_act, projection_wt, projection_point

def getEmbeddingMem(B, S, V, D, precision):
    embedding_act = B * S * D
    embedding_wt = V * D
    embedding_point = 0
    embedding_mem = (embedding_wt + embedding_act + embedding_point) * precision

    return embedding_mem, embedding_act, embedding_wt, embedding_point

def getTotMemReq(exp_config):
    #Model Params
    B          = exp_config.model_config.batch_size
    V          = exp_config.model_config.vocab_size
    L          = exp_config.model_config.num_layers
    D          = exp_config.model_config.layer_size
    projection = exp_config.model_config.projection
    S          = exp_config.model_config.seq_len
    G          = exp_config.model_config.num_gates
    precision  = exp_config.sw_config.precision
    
    #MiniBatch
    dp         = exp_config.sch_config.dp
    miniB      = math.ceil(B / dp)

    hidden_mem, hidden_act, hidden_wt, hidden_point =  getHiddenMem(L=L, 
                                                       Dim1 = miniB, 
                                                       Dim2 = 2 * D, 
                                                       Dim3 = G * D, 
                                                       S = S, 
                                                       precision = precision)
    softmax_mem, softmax_act, softmax_wt, softmax_point =  getSoftmaxMem(B=miniB,
                                                           S=S, 
                                                           P=projection, 
                                                           V=V, 
                                                           precision = precision)
    projection_mem, projection_act, projection_wt, projection_point =  getProjectionMem(B=miniB, 
                                                                       S=S, 
                                                                       P=projection, 
                                                                       D=D, 
                                                                       precision = precision)
    embedding_mem, embedding_act, embedding_wt, embedding_point =  getEmbeddingMem(B=miniB, 
                                                                   S=S, 
                                                                   V=V, 
                                                                   D=D, 
                                                                   precision = precision)
    
    tot_mem = hidden_mem + softmax_mem + embedding_mem + projection_mem
    
    wt_mem = (hidden_wt + softmax_wt + projection_wt + embedding_wt) * precision
    act_mem = (hidden_act + softmax_act + projection_act + embedding_act) * precision
    point_mem = (hidden_point + softmax_point + projection_point + embedding_point) * precision

    return tot_mem, embedding_mem, hidden_mem, softmax_mem, projection_mem, wt_mem, act_mem, point_mem

def getMemUsagePerCore(exp_config):
    #Model params
    B                   = exp_config.model_config.batch_size
    V                   = exp_config.model_config.vocab_size
    L                   = exp_config.model_config.num_layers
    D                   = exp_config.model_config.layer_size
    projection          = exp_config.model_config.projection
    S                   = exp_config.model_config.seq_len
    G                   =  exp_config.model_config.num_gates
    precision           = exp_config.sw_config.precision

    #Parallelism Params
    lp                  = exp_config.sch_config.lp
    kp_hidden_dim1      = exp_config.sch_config.kp_hidden_dim1
    kp_softmax_dim1     = exp_config.sch_config.kp_softmax_dim1
    kp_embedding_dim1   = exp_config.sch_config.kp_embedding_dim1
    kp_projection_dim1  = exp_config.sch_config.kp_projection_dim1
    kp_hidden_dim2      = exp_config.sch_config.kp_hidden_dim2
    kp_softmax_dim2     = exp_config.sch_config.kp_softmax_dim2
    kp_embedding_dim2   = exp_config.sch_config.kp_embedding_dim2
    kp_projection_dim2  = exp_config.sch_config.kp_projection_dim2
    dp                  = exp_config.sch_config.dp
    kp_hidden_type      = exp_config.sch_config.kp_hidden_type #1: CR, 2: RC
    kp_softmax_type     = exp_config.sch_config.kp_softmax_type #1: CR, 2: RC
    kp_embedding_type   = exp_config.sch_config.kp_embedding_type #1: CR, 2: RC
    kp_projection_type  = exp_config.sch_config.kp_projection_type #1: CR, 2: RC
    
    #miniBatch
    miniB               = math.ceil(B / dp)


    hidden_mem, hidden_act, hidden_wt, point_act =  getHiddenMem(L=L/lp, 
        Dim1 = miniB / (kp_hidden_dim1 if kp_hidden_type == 2 else  1), 
        Dim2 = 2 * D / (1 if kp_hidden_type == 2 else kp_hidden_dim1),  
        Dim3 = D * G / (kp_hidden_dim2 if kp_hidden_type == 2 else 1), 
        S = S, 
        precision = precision)

    #activation output from each layer, assuming input ativation are taken 
    #into account in the previous layer
    softmax_mem, softmax_act, softmax_wt, softmax_point =  getSoftmaxMem(B=miniB / (kp_softmax_dim1 if kp_softmax_type == 2 else  1), 
        S=S, 
        P=projection/ (1 if kp_softmax_type == 2 else kp_softmax_dim1 ), 
        V=V/(kp_softmax_dim2 if kp_softmax_type == 2 else 1), 
        precision = precision)

    projection_mem, projection_act, projection_wt, projection_point =  getProjectionMem(B=miniB/(kp_projection_dim1 if kp_projection_type == 2 else  1), 
        S=S, 
        D=D/(1 if kp_projection_type == 2 else kp_projection_dim1), 
        P=projection/(kp_projection_dim2 if kp_projection_type == 2 else 1), 
        precision = precision)
    
    #embedding_mem = miniB * S * D * precision + V * D / kp_embedding_dim1
    embedding_mem, embedding_act, embedding_wt, embedding_point =  getEmbeddingMem(B=miniB/(kp_embedding_dim1 if kp_embedding_type==1 else kp_embedding_dim1 * kp_embedding_dim2), 
        S=S, 
        V=V, 
        D=D, 
        precision = precision)

    tot_mem = hidden_mem + softmax_mem + projection_mem + embedding_mem
      
    return tot_mem, embedding_mem, hidden_mem, softmax_mem, projection_mem


def power2RoundUp(x):
  #TODO: This does not sound like an ideal option
  #Ideally we want to round up to a value which is a multiply of factor of 2 and a number
  #y = math.floor(math.pow(2,(math.ceil(math.log(x,2)))))
  log_power = math.ceil(math.log(x,2))
  power_2   = [2**p for p in range(0, log_power)]
  min_dist  = x
  min_val   = 1
  for i in power_2[::-1]: 
    a = math.ceil(x/i)
    dist = a * i - x
    if (dist < min_dist):
      min_val = a * i
      min_dist = dist
  return min_val
