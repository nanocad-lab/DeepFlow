import math
import sys
import os

import config

core=1
DRAM=1
L2=1
shared_mem=1
reg_mem=1
proj = False #Turn off the projection layer

def printError(message):
  sys.exit(message)

def getHiddenMem(L, Dim1, Dim2, Dim3, S, precision):
    #Activations refer to output activations that need to be stored
    hidden_act = Dim1 * Dim3 * S * L * precision
    hidden_wt  = (Dim2 + 1) * Dim3 * L * precision
    hidden_point = (Dim1 * Dim3 / 2) * 9 * L * S * precision
    #3 sigmoids
    #2 tanh
    #3 pointwise multiply
    #1 addition
    hidden_mem = (hidden_act + hidden_wt + hidden_point)

    return hidden_mem, hidden_act, hidden_wt, hidden_point

def getSoftmaxMem(B, S, P, V, precision):
     #activation output from each layer, assuming input ativation are taken 
    #into account in the previous layer
    softmax_act = B * S * V * precision 
    softmax_wt = (P + 1) * V * precision
    softmax_point = (2 * B * S * V + B * S) * precision
    #NOTE: sigmoid and exp could have been combined
    #1 sigmoids
    #1 exp
    #1 pointwise div
    softmax_mem = (softmax_act + softmax_wt + softmax_point)

    return softmax_mem, softmax_act, softmax_wt, softmax_point

def getProjectionMem(B, S, P, D, precision):
    projection_act = B * S * P * precision
    projection_wt = (D + 1) * P * precision
    projection_point= B * S * P * precision
    projection_mem = (projection_act + projection_wt + projection_point)
  
    return projection_mem, projection_act, projection_wt, projection_point

def getEmbeddingMem(B, S, V, D, precision):
    embedding_act = B * S * D * precision
    embedding_wt = V * D * precision
    embedding_point = 0
    embedding_mem = (embedding_wt + embedding_act + embedding_point)

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
                                                           P=(projection if proj else D), 
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


def getMemUsagePerCore(exp_config, **kwargs):
    #Model params
    B                   = int(kwargs.get('batch_size', exp_config.model_config.batch_size))
    D                   = int(kwargs.get('hidden_dim', exp_config.model_config.layer_size))
    V                   = exp_config.model_config.vocab_size
    L                   = exp_config.model_config.num_layers
    projection          = exp_config.model_config.projection 
    S                   = exp_config.model_config.seq_len
    G                   = exp_config.model_config.num_gates
    precision           = exp_config.sw_config.precision

    #Parallelism Params
    dp                  = int(kwargs.get('dp', exp_config.sch_config.dp))
    lp                  = int(kwargs.get('lp', exp_config.sch_config.lp))
    
    kp_hidden_dim1      = int(kwargs.get('kp1', exp_config.sch_config.kp_hidden_dim1))
    kp_softmax_dim1     = int(kwargs.get('kp1', exp_config.sch_config.kp_softmax_dim1))
    kp_embedding_dim1   = int(kwargs.get('kp1', exp_config.sch_config.kp_embedding_dim1))
    kp_projection_dim1  = int(kwargs.get('kp1', exp_config.sch_config.kp_projection_dim1))

    kp_hidden_dim2      = int(kwargs.get('kp2', exp_config.sch_config.kp_hidden_dim2))
    kp_softmax_dim2     = int(kwargs.get('kp2', exp_config.sch_config.kp_softmax_dim2))
    kp_embedding_dim2   = int(kwargs.get('kp2', exp_config.sch_config.kp_embedding_dim2))
    kp_projection_dim2  = int(kwargs.get('kp2', exp_config.sch_config.kp_projection_dim2))
    
    kp_hidden_type      = int(kwargs.get('kp_type', exp_config.sch_config.kp_hidden_type)) #1: CR, 2: RC
    kp_softmax_type     = int(kwargs.get('kp_type', exp_config.sch_config.kp_softmax_type)) #1: CR, 2: RC
    kp_embedding_type   = int(kwargs.get('kp_type', exp_config.sch_config.kp_embedding_type)) #1: CR, 2: RC
    kp_projection_type  = int(kwargs.get('kp_type', exp_config.sch_config.kp_projection_type)) #1: CR, 2: RC
    
    #miniBatch
    miniB               = math.ceil(B / dp)

    hlp = lp
    if lp > 2:
      hlp = hlp - 2
    hidden_mem, hidden_act, hidden_wt, point_act =  getHiddenMem(L=L/hlp, 
        Dim1 = math.ceil(miniB / (kp_hidden_dim1 if kp_hidden_type == 2 else  1)), 
        Dim2 = math.ceil(2 * D / (1 if kp_hidden_type == 2 else kp_hidden_dim1)),  
        Dim3 = math.ceil(D * G / (kp_hidden_dim2 if kp_hidden_type == 2 else 1)), 
        S = S, 
        precision = precision)

    #activation output from each layer, assuming input ativation are taken 
    #into account in the previous layer
    softmax_mem, softmax_act, softmax_wt, softmax_point =  getSoftmaxMem(
        B=math.ceil(miniB / (kp_softmax_dim1 if kp_softmax_type == 2 else  1)), 
        S=S, 
        P=math.ceil((projection if proj else D)/ (1 if kp_softmax_type == 2 else kp_softmax_dim1)), 
        V=math.ceil(V/(kp_softmax_dim2 if kp_softmax_type == 2 else 1)), 
        precision = precision)

    if proj:
        projection_mem, projection_act, projection_wt, projection_point =  getProjectionMem(
            B=math.ceil(miniB/(kp_projection_dim1 if kp_projection_type == 2 else  1)), 
            S=S, 
            D=math.ceil(D/(1 if kp_projection_type == 2 else kp_projection_dim1)), 
            P=math.ceil(projection/(kp_projection_dim2 if kp_projection_type == 2 else 1)), 
            precision = precision)
    else:
      projection_mem, projection_act, projection_wt, projection_point = 0, 0, 0 , 0
    #embedding_mem = miniB * S * D * precision + V * D / kp_embedding_dim1
    embedding_mem, embedding_act, embedding_wt, embedding_point =  getEmbeddingMem(
        B=math.ceil(miniB/(kp_embedding_dim1 if kp_embedding_type==2 else 1)), 
        S=S, 
        V=math.ceil(V/(1 if kp_embedding_type == 2 else kp_embedding_dim1)), 
        D=math.ceil(D/(kp_embedding_dim2 if kp_hidden_type == 2 else 1)), 
        precision = precision)

    tot_mem = 0

    if lp == 1:
      tot_mem = hidden_mem + softmax_mem + (projection_mem if proj else 0)+ embedding_mem
    elif lp >= 4: 
      tot_mem = max(hidden_mem, embedding_mem, (softmax_mem + projection_mem if proj else 0))
    else:
      NotImplemented

    return tot_mem, embedding_mem, hidden_mem, softmax_mem, projection_mem

def getChipArea(exp_config_path, **kwargs):

    exp_path = os.path.expandvars(os.path.expanduser(exp_config_path))
    exp_config = config.parse_config(exp_path)
    
    batch_size = int(kwargs.get('batch_size', exp_config.model_config.batch_size))
    hidden_dim = int(kwargs.get('hidden_dim', exp_config.model_config.layer_size))
    dp = int(kwargs.get('dp', exp_config.sch_config.dp))
    lp = int(kwargs.get('lp', exp_config.sch_config.lp))
    #type:-1 no kp
    #type: 1 col-row
    #type: 2 row-col
    kp_type = int(kwargs.get('kp_type', -1))
    kp1 = int(kwargs.get('kp1', 1))
    kp2 = int(kwargs.get('kp2', 1))
    tot_mem    = getMemUsagePerCore(exp_config, 
                                    batch_size=batch_size, 
                                    hidden_dim=hidden_dim,
                                    dp=dp, 
                                    lp=lp, 
                                    kp_type=kp_type,
                                    kp1=kp1,
                                    kp2=kp2)[0]
    stack_capacity = exp_config.tech_config.DRAM.stack_capacity 
    area_per_stack = exp_config.tech_config.DRAM.area_per_stack
    node_area_budget = exp_config.area_breakdown.node_area_budget 

    mem_area = math.ceil(tot_mem / stack_capacity) * area_per_stack
    #print("Node_Area: {}, Mem_area: {}".format(node_area_budget, mem_area))
    chip_area_budget = node_area_budget - mem_area

    return chip_area_budget

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
