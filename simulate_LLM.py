import math
from heapq import heappush, heappop
import sys
from graphviz import Digraph
import os
showing_breakdown = False
all_reduce =  "the end" #"every layer" define whether the all reduce happens after computation of every transformer layer or after the model is processed
debug = False
class Node:

    def __init__(self, name, op_id, hw_id, duration, fwd=True):
        self.name = name
        self.op_id = op_id
        self.hw_id = hw_id
        self.duration = duration
        self.done = False
        self.finish_time = -1
        self.parents = []
        self.children = []
        self.memory = 0  # memory usage
        self.fwd = fwd  # forward or backward
        self.scheduled = False

    def add_child(self, obj):
        self.children.append(obj)
        obj.parents.append(self)
        
        
        
class Data_batch:
    def __init__(self, name, batch_id, duration):
        self.name = name
        self.batch_id = batch_id
        self.duration = duration
        self.done = False
        self.finish_time = -1
        self.parents = []
        self.children = []
        self.scheduled = False
        # self.memory = 0  # memory usage

    def add_child(self, obj):
        self.children.append(obj)
        obj.parents.append(self)

class Edge:
  def __init__(self, name, op_id, link_id, duration, is_all_reduce=False, comm_size_bytes=0, comm_type=None, participants=1, comm_interconnect_type=None):
    self.name = name
    self.op_id = op_id
    self.link_id = link_id
    self.duration = duration
    self.done = False
    self.finish_time = -1
    self.parents = [] 
    self.children = []
    self.is_all_reduce = is_all_reduce
    # self.is_grad = is_grad
    self.scheduled = False
    self.comm_size_bytes = comm_size_bytes
    self.comm_type = comm_type
    self.participants = participants
    self.comm_interconnect_type = comm_interconnect_type

  def add_child(self, obj):
      self.children.append(obj)
      obj.parents.append(self)
      
class Gradient:
  def __init__(self, name, op_id, hw_id, duration):
    self.name = name
    self.op_id = op_id
    self.hw_id = hw_id
    self.duration = duration
  # "dp", "lp", "kp1", "kp2"
    self.done = False
    self.finish_time = -1
    self.parents = [] 
    self.children = []
    self.scheduled = False

  def add_child(self, obj):
      self.children.append(obj)
      obj.parents.append(self)

class Graph:

    def __init__(
        self,
        lp,     #pipeline depth or number of gpus
        num_layer,    #number of layers
        num_batch,     #number of micro-batches
        T_embedding_f,    #forward embedding time
        T_embedding_b,    #backward embedding time
        T_linear_softmax_f,    #forward linear softmax time
        T_linear_softmax_b,    #backward linear softmax time
        # Tf,    #forward transformer time
        # Tb,    #backward transformer time
        # T_reduction_transformer,  #reduction transformer time
        # T_grad_transformer,  #gradient transformer time
        # T_reduction_embedding,    #reduction embedding time
        # T_reduction_linear_softmax,    #reduction linear softmax time
        T_transformer_f,    #transformer forward time
        T_transformer_b,    #transformer backward time
        all_reduce,   #when the all_reduce happens in data parallelism
        comm_metadata,

    ):
        self.num_batch = num_batch
        self.num_layer = num_layer
        self.lp = lp
        self.T_embedding_f = T_embedding_f
        self.T_embedding_b = T_embedding_b
        self.T_linear_softmax_f = T_linear_softmax_f
        self.T_linear_softmax_b = T_linear_softmax_b
        self.T_transformer_f = T_transformer_f
        self.T_transformer_b = T_transformer_b
        self.all_reduce = all_reduce
        self.layer_per_device = math.ceil(num_layer / lp) #how many layers of transformer on one gpu
        # self.Tf = Tf
        # self.Tb = Tb
        # self.T_reduction_transformer = T_reduction_transformer
        # self.T_grad_transformer = T_grad_transformer
        # self.T_reduction_embedding = T_reduction_embedding
        # self.T_reduction_linear_softmax = T_reduction_linear_softmax
        self.comm_metadata = comm_metadata
    def get_link_id(self, hw_id1, hw_id2): #assuming ring topology
        if hw_id1 == hw_id2: #same gpu, no link
            return -1
        elif abs(hw_id1 - hw_id2) == 1: #adjacent gpus link id = first gpu
            return min(hw_id1, hw_id2)
        else: #between first gpu0 and last gpu(lp-1)
            return self.lp-1
    def create_comm_edge(self, name, op_id, hw_id, comm_key,is_all_reduce=False):
        """Create a communication edge with optional local computation node.

        Args:
            name: Edge name
            op_id: Operation ID
            hw_id: Hardware ID
            comm_key: Key into self.comm_metadata dict

        Returns:
            The communication edge (connection point for graph building)
        """
        comm_data = self.comm_metadata[comm_key]

        # Create communication edge with metadata
        comm_edge = Edge(
            name=name,
            op_id=op_id,
            link_id=hw_id,
            duration=0,  # Will be filled in second pass
            comm_size_bytes=comm_data['size'],
            comm_type=comm_data['type'],
            participants=comm_data['participants'],
            comm_interconnect_type=comm_data['interconnect_type'],
            is_all_reduce=is_all_reduce
        )

        # If there's local computation time, create local node after edge
        local_comp_time = comm_data.get('local_comp_time', 0)
        print(f"name: {name}")
        print(f"local_comp_time: {local_comp_time}")
        if local_comp_time > 0:
            local_node = Node(
                name=f"{name}_local_comp",
                op_id=op_id + 100000,  # Offset to avoid ID conflicts
                hw_id=hw_id,
                duration=local_comp_time
            )
            comm_edge.add_child(local_node)

        return comm_edge

    def convert_comm_sizes_to_times(self, roots, network_model, interconnect_params):
        """
        Args:
            network_model: NetworkModel instance for collective timing
            interconnect_params: Dict with bandwidth/latency for each type
                                {'dp': (ib, ll), 'lp': (ib, ll), 'kp1': (ib, ll), 'kp2': (ib, ll)}
        """
        def traverse_and_convert(node, visited=None):
            if visited is None:
                visited = set()
            if id(node) in visited:
                return
            visited.add(id(node))

            # Process children (edges and nodes)
            for child in node.children:
                # If it's an edge with communication size, convert to time
                if hasattr(child, 'comm_size_bytes') and child.comm_size_bytes > 0:
                    # Get the appropriate bandwidth/latency for this interconnect type
                    interconnect_type = child.comm_interconnect_type
                    if interconnect_type and interconnect_type in interconnect_params:
                        ib, ll = interconnect_params[interconnect_type]
                    else:
                        raise ValueError(f"Invalid interconnect type: {interconnect_type}") 

                    child.duration = network_model.collective(
                        kind=child.comm_type,
                        size_bytes=child.comm_size_bytes,
                        participants=child.participants,
                        ib=ib,
                        ll=ll,
                        local_bytes=0.0,
                        local_ops=0.0,
                        debug_label=f"{child.name}_conversion"
                    )
                    print(f"Converted {child.name} size {child.comm_size_bytes} bytes to duration {child.duration:.6f} sec using {interconnect_type} (ib={ib}, ll={ll})")

                # Recursively process this child
                traverse_and_convert(child, visited)

        traverse_and_convert(roots)
        return roots

    def construct_fwd_graph(self):
        embedding_node = []
        data_batch_node = []
        softmax_node = []
        transformer_nodes = [[] for _ in range(self.num_batch)]  # 

        op_id = 0  # operation ID, used to distinguish nodes and edges
        batch_id = 0  # batch ID, used to distinguish data batches
        
        data0 = Data_batch("data0", batch_id, 0)
        data_batch_node.append(data0)
        
        for i in range(1, self.num_batch):#create data batch node
            
            data_batch_node.append(Data_batch(f"data{i}", i, 0))
            data_batch_node[i-1].add_child(data_batch_node[i])

        for b in range(self.num_batch): #connect each data batch node with corresponding nodes
            linear_softmax = Node(f"linear_softmax{b}", op_id, self.lp-1, self.T_linear_softmax_f); op_id += 1
            softmax_node.append(linear_softmax)
            emb = Node(f"embeddding{b}", op_id, 0, self.T_embedding_f)      # hw_id = 0
            op_id += 1
            embedding_node.append(emb)
            data_batch_node[b].add_child(embedding_node[b])

            for l in range(self.num_layer):
                transformer_nodes[b].append([])

                hw_id = min(l // self.layer_per_device , self.lp - 1)#assign hw_id for transformer
                transformer_node = Node("transformer", op_id, hw_id, self.T_transformer_f)
                op_id += 1

                transformer_nodes[b][l]=transformer_node

            for l in range(1, self.num_layer):

                prev_node = transformer_nodes[b][l-1]        # previous layer
                curr_node  = transformer_nodes[b][l]            # current layer


                link_id = self.get_link_id(prev_node.hw_id, curr_node.hw_id)
                if link_id == -1:
                    edge = Edge("cross_layer", op_id, -1, 0)  # on same GPU
                else:
                    edge = Edge("cross_layer", op_id, link_id, self.Tf)  # on different GPU
                op_id += 1

                prev_node.add_child(edge); edge.add_child(curr_node)#connect previous layer and current layer

            first_node = transformer_nodes[b][0]   # first layer
            link_id = self.get_link_id(first_node.hw_id, embedding_node[b].hw_id)
            if link_id == -1:
                edge = Edge("Emb_node0", op_id, -1, 0)
            else:
                edge = Edge("cross_layer", op_id, -1, self.Tf)  
            op_id += 1
            embedding_node[b].add_child(edge) #connect embedding node and first transformer layer
            edge.add_child(first_node)


            last_node = transformer_nodes[b][-1]  # last layer
            link_id = self.get_link_id(last_node.hw_id, softmax_node[b].hw_id)
            if link_id == -1:
                node_Softmax = Edge("node_Softmax", op_id, -1, 0)  # same GPU
            else:
                node_Softmax = Edge("node_Softmax", op_id, link_id, self.Tf)
            op_id += 1
            last_node.add_child(node_Softmax) #connect last layer and softmax layer
            node_Softmax.add_child(softmax_node[b])
        
            #add dependency edges
        for b in range(self.num_batch - 1):
            gpu_index = 0
            last_transformer_layer = []
            first_transformer_layer = []
            first_transformer_layer.append(0)
            for l in range(self.num_layer-1):

                if transformer_nodes[b][l].hw_id != transformer_nodes[b][l+1].hw_id: # check if on different GPUs
                    last_transformer_layer.append(l) # record last layer on each GPU
                    first_transformer_layer.append(l+1) # record first layer on each GPU
                    gpu_index += 1
                    if transformer_nodes[b][l].hw_id == 0: #if first pipeline stage
                        transformer_nodes[b][l].add_child(embedding_node[b+1]) #add dependency edge between embedding of next batch and last transformer in stage 0
                    else:
                        transformer_nodes[b][l].add_child(transformer_nodes[b+1][first_transformer_layer[gpu_index-1]]) #add dependency edge between last transformer of current batch and first transformer of next batch

            softmax_node[b].add_child(transformer_nodes[b+1][first_transformer_layer[-1]]) #add dependency edge between softmax of current batch and transformer node of next batch

        return data_batch_node[0]

    def construct_fwd_bwd_graph(self):
        embedding_node = []
        data_batch_node = []
        softmax_node = []
        transformer_nodes = [[] for _ in range(self.num_batch)]  # 


        embedding_node_b = [[] for _ in range(self.num_batch)]
        softmax_node_b = [[] for _ in range(self.num_batch)]
        # data_batch_node = [[] for _ in range(self.num_batch)]

        R_edge = [[] for _ in range(self.num_batch)]
        G_edge = [[] for _ in range(self.num_batch)]
        
        #######
        transformer_nodes_b = [[] for _ in range(self.num_batch)]  # 
        for b in range(self.num_batch):
            transformer_nodes_b[b] = [[] for _ in range(self.num_layer)]




        op_id = 0  # operation ID, used to distinguish nodes and edges
        batch_id = 0  # batch ID, used to distinguish data batches
        
        data0 = Data_batch("data0", batch_id, 0)
        data_batch_node.append(data0)
        
        for i in range(1, self.num_batch):#create data batch node
            
            data_batch_node.append(Data_batch(f"data{i}", i, 0))
            data_batch_node[i-1].add_child(data_batch_node[i])

        for b in range(self.num_batch): #connect each data batch node with corresponding nodes
            linear_softmax = Node(f"linear_softmax{b}", op_id, self.lp-1, self.T_linear_softmax_f); op_id += 1
            softmax_node.append(linear_softmax)
            emb = Node(f"embeddding{b}", op_id, 0, self.T_embedding_f)      # hw_id = 0
            op_id += 1
            embedding_node.append(emb)
            data_batch_node[b].add_child(embedding_node[b])

            for l in range(self.num_layer):
                transformer_nodes[b].append([])

                hw_id = min(l // self.layer_per_device , self.lp - 1)#assign hw_id for transformer
                transformer_node = Node("transformer", op_id, hw_id, self.T_transformer_f)
                op_id += 1

                transformer_nodes[b][l]=transformer_node

            for l in range(1, self.num_layer):

                prev_node = transformer_nodes[b][l-1]        # previous layer
                curr_node  = transformer_nodes[b][l]            # current layer


                link_id = self.get_link_id(prev_node.hw_id, curr_node.hw_id)
                if link_id == -1:
                    edge = Edge("cross_layer", op_id, -1, 0)  # on same GPU
                else:
                    edge = self.create_comm_edge("cross_layer", op_id, link_id, 'cross_layer')  # on different GPU
                op_id += 1

                prev_node.add_child(edge); edge.add_child(curr_node)#connect previous layer and current layer

            first_node = transformer_nodes[b][0]   # first layer
            link_id = self.get_link_id(first_node.hw_id, embedding_node[b].hw_id)
            if link_id == -1:
                edge = Edge("Emb_node0", op_id, -1, 0)
            else:
                edge = self.create_comm_edge("Emb_node0", op_id, link_id, 'cross_layer')
                # edge = Edge("Emb_node0", op_id, link_id, self.Tf)
            op_id += 1
            embedding_node[b].add_child(edge) #connect embedding node and first transformer layer
            edge.add_child(first_node)


            last_node = transformer_nodes[b][-1]  # last layer
            link_id = self.get_link_id(last_node.hw_id, softmax_node[b].hw_id)
            if link_id == -1:
                node_Softmax = Edge("node_Softmax", op_id, -1, 0)  # same GPU
            else:
                node_Softmax = self.create_comm_edge("node_Softmax", op_id, link_id, 'cross_layer')
                # node_Softmax = Edge("node_Softmax", op_id, link_id, self.Tf)
            op_id += 1
            last_node.add_child(node_Softmax) #connect last layer and softmax layer
            node_Softmax.add_child(softmax_node[b])
        
            #add dependency edges
        for b in range(self.num_batch - 1):
            gpu_index = 0
            last_transformer_layer = []
            first_transformer_layer = []
            first_transformer_layer.append(0)
            for l in range(self.num_layer-1):

                if transformer_nodes[b][l].hw_id != transformer_nodes[b][l+1].hw_id: # check if on different GPUs
                    last_transformer_layer.append(l) # record last layer on each GPU
                    first_transformer_layer.append(l+1) # record first layer on each GPU
                    gpu_index += 1
                    if transformer_nodes[b][l].hw_id == 0: #if first pipeline stage
                        transformer_nodes[b][l].add_child(embedding_node[b+1]) #add dependency edge between embedding of next batch and last transformer in stage 0
                    else:
                        transformer_nodes[b][l].add_child(transformer_nodes[b+1][first_transformer_layer[gpu_index-1]]) #add dependency edge between last transformer of current batch and first transformer of next batch

            softmax_node[b].add_child(transformer_nodes[b+1][first_transformer_layer[-1]]) #add dependency edge between softmax of current batch and transformer node of next batch




######################backward


        for b in reversed(range(self.num_batch)): #connect each data batch node with corresponding nodes
            emb_b = Node("embeddding_b", op_id, 0, self.T_embedding_b, fwd=False)      # hw_id = 0
            op_id += 1
            embedding_node_b[b] = emb_b
            linear_softmax_b = Node("linear_softmax_b", op_id, self.lp-1, self.T_linear_softmax_b, fwd=False); op_id += 1
            softmax_node_b[b] = linear_softmax_b
            softmax_node[b].add_child(linear_softmax_b)


            for l in reversed(range(self.num_layer)):

                hw_id = min(l // self.layer_per_device , self.lp - 1)
                transformer_node_b = Node("transformer_b", op_id, hw_id, self.T_transformer_b, fwd=False); op_id += 1
                transformer_nodes_b[b][l] = transformer_node_b

            for l in reversed(range(1, self.num_layer)):
                curr_node = transformer_nodes_b[b][l]         # current layer's qkv_proj.
                next_ffn2  = transformer_nodes_b[b][l-1]            # next layer's layernorm.
                link_id =self.get_link_id(curr_node.hw_id, next_ffn2.hw_id)
                if link_id == -1:
                    edge = Edge("cross_layer", op_id, -1, 0)  
                else:
                    edge = self.create_comm_edge("cross_layer", op_id, link_id, 'cross_layer')
                    # edge = Edge("cross_layer", op_id, link_id, self.Tb)  
                op_id += 1

                curr_node.add_child(edge); edge.add_child(next_ffn2)

            qkv_0_b = transformer_nodes_b[b][0]     # first layer's qkv_proj
            link_id = self.get_link_id(qkv_0_b.hw_id, emb_b.hw_id)
            if link_id == -1:
                edge = Edge("Emb_node0", op_id, -1, 0)
            else:
                edge = self.create_comm_edge("Emb_node0", op_id, link_id, 'cross_layer')
                # edge = Edge("Emb_node0", op_id, link_id, self.Tb)
            op_id += 1
            qkv_0_b.add_child(edge)
            edge.add_child(emb_b)


            prev_layer_norm2 = transformer_nodes_b[b][self.num_layer-1] # last layer's layernorm2
            link_id = self.get_link_id(prev_layer_norm2.hw_id, softmax_node_b[b].hw_id)
            if link_id == -1:
                layernorm_Softmax = Edge("layernorm2_Softmax", op_id, -1, 0)  # same GPU
            else:
                layernorm_Softmax = self.create_comm_edge("layernorm2_Softmax", op_id, link_id, 'cross_layer')
                # layernorm_Softmax = Edge("layernorm2_Softmax", op_id, link_id, self.Tb)
            op_id += 1
            softmax_node_b[b].add_child(layernorm_Softmax)
            layernorm_Softmax.add_child(prev_layer_norm2)
                

            # all-reduce

            R_edge[b].append(self.create_comm_edge("Reduce_Embedding", op_id, transformer_nodes_b[b][-1].hw_id , "embedding", is_all_reduce=True))
            op_id += 1
            if self.all_reduce == "the end":
                R_edge[b].append(self.create_comm_edge("Reduce_transformer", op_id, transformer_nodes_b[b][-1].hw_id  , "transformer", is_all_reduce=True))
                op_id += 1

                R_edge[b].append(self.create_comm_edge("Reduce_Softmax", op_id,  transformer_nodes_b[b][0].hw_id, "softmax", is_all_reduce=True))
                op_id += 1
            # Attach All-Reduce Edges
                softmax_node_b[b].add_child(R_edge[b][-1])
                embedding_node_b[b].add_child(R_edge[b][0])
                transformer_nodes_b[b][0].add_child(R_edge[b][1])

            elif self.all_reduce == "every layer":
                for i in range(0, self.num_layer):
                    R_edge[b].append(self.create_comm_edge("Reduce_transformer", op_id, transformer_nodes_b[b][i].hw_id  , "transformer", is_all_reduce=True))
                    op_id += 1
                    # G_edge[b].append(Gradient("Grad_transformer", op_id, transformer_nodes_b[b][-1].hw_id, self.T_grad_transformer))
                    # op_id += 1
                    # R_edge[b][-1].add_child(G_edge[b][-1])

                R_edge[b].append(self.create_comm_edge("Reduce_Softmax", op_id,  transformer_nodes_b[b][0].hw_id, "softmax", is_all_reduce=True))
                op_id += 1
            # Attach All-Reduce Edges
                softmax_node_b[b].add_child(R_edge[b][-1])
                embedding_node_b[b].add_child(R_edge[b][0])
                for i in range(0, self.num_layer):
                    transformer_nodes_b[b][i].add_child(R_edge[b][i + 1])
            else:
                sys.exit("Invalid all_reduce option")
                    
                
                
        last_transformer_layer = [-1] * self.lp  # Initialize with -1 for all GPUs
        first_transformer_layer = [-1] * self.lp  # Initialize with -1 for all GPUs

        # first_transformer_layer.append(0)
        gpu_index = self.lp - 1
        for l in range(self.num_layer - 1, 0, -1):
            
            if transformer_nodes_b[0][l].hw_id != transformer_nodes_b[0][l-1].hw_id:  # Check if on different GPU
                # print("Layer ", l, " is on GPU ", transformer_nodes_b[0][l].hw_id)
                first_transformer_layer[gpu_index-1] = l-1  # Record first layer on each GPU
                last_transformer_layer[gpu_index] = l  # Record last layer on each GPU
                gpu_index -= 1
        # for id in range(self.lp):
            # print("GPU ", id, " first layer ", first_transformer_layer[id], " last layer ", last_transformer_layer[id])



        for b in range(self.num_batch-1, 0, -1):
            gpu_index = self.lp - 1
            

            for l in range(self.num_layer - 1, 0, -1):

                if transformer_nodes_b[b][l].hw_id != transformer_nodes_b[b][l-1].hw_id:  # Check if on different GPUs
                    # last_transformer_layer.append(l)  # Record last layer on each GPU
                    # first_transformer_layer.append(l-1)  # Record first layer on each GPU
                    
                    if transformer_nodes_b[b][l].hw_id == self.lp - 1:
                        transformer_nodes_b[b][l].add_child(softmax_node_b[b-1])  # Add dependency edge
                    else:
                        transformer_nodes_b[b][l].add_child(transformer_nodes_b[b-1][first_transformer_layer[gpu_index]])  # Add dependency edge
                    gpu_index -= 1
            # Ensure embedding_node_b[b] is connected to the correct transformer node
            # if first_transformer_layer:
            embedding_node_b[b].add_child(transformer_nodes_b[b-1][first_transformer_layer[0]])  # Add dependency edge






        return data_batch_node
    def construct_bwd_graph(self):
        embedding_node_b = [[] for _ in range(self.num_batch)]
        softmax_node_b = [[] for _ in range(self.num_batch)]
        data_batch_node = [[] for _ in range(self.num_batch)]

        #######
        transformer_nodes_b = [[] for _ in range(self.num_batch)]  # 
        for b in range(self.num_batch):
            transformer_nodes_b[b] = [[] for _ in range(self.num_layer)]

        R_edge = [[] for _ in range(self.num_batch)]
        G_edge = [[] for _ in range(self.num_batch)]

        op_id = 0  
        batch_id = 0  # batch ID, used to distinguish data batches
        data0 = Data_batch("data0", batch_id, 0)
        data_batch_node[0] = data0
        for i in range(1, self.num_batch):#create data batch node

            data_batch_node[i] = Data_batch(f"data{i}", i, 0)
            data_batch_node[i].add_child(data_batch_node[i-1])





        for b in reversed(range(self.num_batch)): #connect each data batch node with corresponding nodes
            emb_b = Node("embeddding_b", op_id, 0, self.T_embedding_b, fwd=False)      # hw_id = 0
            op_id += 1
            embedding_node_b[b] = emb_b
            linear_softmax_b = Node("linear_softmax_b", op_id, self.lp-1, self.T_linear_softmax_b, fwd=False); op_id += 1
            softmax_node_b[b] = linear_softmax_b
            data_batch_node[b].add_child(linear_softmax_b)


            for l in reversed(range(self.num_layer)):

                hw_id = min(l // self.layer_per_device , self.lp - 1)
                transformer_node_b = Node("transformer_b", op_id, hw_id, self.T_transformer_b, fwd=False); op_id += 1
                transformer_nodes_b[b][l] = transformer_node_b

            for l in reversed(range(1, self.num_layer)):
                curr_node = transformer_nodes_b[b][l]         # current layer's qkv_proj.
                next_ffn2  = transformer_nodes_b[b][l-1]            # next layer's layernorm.
                link_id =self.get_link_id(curr_node.hw_id, next_ffn2.hw_id)
                if link_id == -1:
                    edge = Edge("cross_layer", op_id, -1, 0)  
                else:
                    edge = Edge("cross_layer", op_id, link_id, self.Tb)  
                op_id += 1

                curr_node.add_child(edge); edge.add_child(next_ffn2)

            qkv_0_b = transformer_nodes_b[b][0]     # first layer's qkv_proj
            link_id = self.get_link_id(qkv_0_b.hw_id, emb_b.hw_id)
            if link_id == -1:
                edge = Edge("Emb_node0", op_id, -1, 0)
            else:
                edge = Edge("Emb_node0", op_id, link_id, self.Tb)
            op_id += 1
            qkv_0_b.add_child(edge)
            edge.add_child(emb_b)


            prev_layer_norm2 = transformer_nodes_b[b][self.num_layer-1] # last layer's layernorm2
            link_id = self.get_link_id(prev_layer_norm2.hw_id, softmax_node_b[b].hw_id)
            if link_id == -1:
                layernorm_Softmax = Edge("layernorm2_Softmax", op_id, -1, 0)  # same GPU
            else:
                layernorm_Softmax = Edge("layernorm2_Softmax", op_id, link_id, self.Tb)
            op_id += 1
            softmax_node_b[b].add_child(layernorm_Softmax)
            layernorm_Softmax.add_child(prev_layer_norm2)
                

            # all-reduce
        # R_edge.append(self.create_comm_edge("Reduce_Embedding", 0, self.lp - 1, "embedding"))

            R_edge[b].append(self.create_comm_edge("Reduce_Embedding", op_id, self.lp ,is_all_reduce=True))
            op_id += 1
            if self.all_reduce == "the end":
                R_edge[b].append(self.create_comm_edge("Reduce_transformer", op_id, self.lp  ,  is_all_reduce=True))
                op_id += 1
                G_edge[b].append(Gradient("Reduce_transformer", op_id, transformer_nodes_b[b][-1].hw_id, self.T_grad_transformer))
                op_id += 1

                R_edge[b].append(self.create_comm_edge("Reduce_Softmax", op_id,  self.lp, "softmax", is_all_reduce=True))
                op_id += 1
            # Attach All-Reduce Edges
                softmax_node_b[b].add_child(R_edge[b][-1])
                embedding_node_b[b].add_child(R_edge[b][0])
                transformer_nodes_b[b][0].add_child(R_edge[b][1])

            elif self.all_reduce == "every layer":
                for i in range(0, self.num_layer):
                    R_edge[b].append(self.create_comm_edge("Reduce_transformer", op_id, self.lp  ,  is_all_reduce=True))
                    op_id += 1
                    G_edge[b].append(Gradient("Gradient_transformer", op_id, transformer_nodes_b[b][l].hw_id, self.T_grad_transformer))
                    # print("Gradient event created")
                    op_id += 1
                    R_edge[b][-1].add_child(G_edge[b][-1])

                R_edge[b].append(self.create_comm_edge("Reduce_Softmax", op_id,   self.lp,  is_all_reduce=True))
                op_id += 1
            # Attach All-Reduce Edges
                softmax_node_b[b].add_child(R_edge[b][-1])
                embedding_node_b[b].add_child(R_edge[b][0])
                for i in range(0, self.num_layer):
                    transformer_nodes_b[b][i].add_child(R_edge[b][i + 1])
            else:
                sys.exit("Invalid all_reduce option")
                    
                
                
        last_transformer_layer = [-1] * self.lp  # Initialize with -1 for all GPUs
        first_transformer_layer = [-1] * self.lp  # Initialize with -1 for all GPUs

        # first_transformer_layer.append(0)
        gpu_index = self.lp - 1
        for l in range(self.num_layer - 1, 0, -1):
            
            if transformer_nodes_b[0][l].hw_id != transformer_nodes_b[0][l-1].hw_id:  # Check if on different GPU
                # print("Layer ", l, " is on GPU ", transformer_nodes_b[0][l].hw_id)
                first_transformer_layer[gpu_index-1] = l-1  # Record first layer on each GPU
                last_transformer_layer[gpu_index] = l  # Record last layer on each GPU
                # The code is decrementing the value of the variable `gpu_index` by 1.
                gpu_index -= 1
        # for id in range(self.lp):
            # print("GPU ", id, " first layer ", first_transformer_layer[id], " last layer ", last_transformer_layer[id])



        for b in range(self.num_batch-1, 0, -1):
            gpu_index = self.lp - 1
            

            for l in range(self.num_layer - 1, 0, -1):

                if transformer_nodes_b[b][l].hw_id != transformer_nodes_b[b][l-1].hw_id:  # Check if on different GPUs
                    # last_transformer_layer.append(l)  # Record last layer on each GPU
                    # first_transformer_layer.append(l-1)  # Record first layer on each GPU
                    
                    if transformer_nodes_b[b][l].hw_id == self.lp - 1:
                        transformer_nodes_b[b][l].add_child(softmax_node_b[b-1])  # Add dependency edge
                    else:
                        transformer_nodes_b[b][l].add_child(transformer_nodes_b[b-1][first_transformer_layer[gpu_index]])  # Add dependency edge
                    gpu_index -= 1
            # Ensure embedding_node_b[b] is connected to the correct transformer node
            # if first_transformer_layer:
            embedding_node_b[b].add_child(transformer_nodes_b[b-1][first_transformer_layer[0]])  # Add dependency edge


        return data_batch_node[self.num_batch - 1]

        

        # R_edge.append(self.create_comm_edge("Reduce_Embedding", 0, self.lp - 1, "embedding"))

        # for i in range(0, self.num_layer):
        #     R_edge.append(self.create_comm_edge("Reduce_transformer", i, (0 if self.lp == 1 else int(i // self.layer_per_device) + self.lp), "transformer"))

        # R_edge.append(self.create_comm_edge("Reduce_Softmax", 0, 2 * self.lp - 2, "softmax"))
        # # Attach All-Reduce Edges
        # softmax_node[0].add_child(R_edge[self.num_layer + 1])
        # embedding_node[0].add_child(R_edge[0])
        # for i in range(0, self.num_layer):
        #     xform_node[i][0][-1].add_child(R_edge[i + 1])

        # return softmax_node

    def simulate(self, root):
        time = 0
        counter = 0
        event_queue = []
        done_list = []
        ready_list = []

        # for r in roots:
        #  ready_list.append(r)
        ready_list.append(root)
        root.scheduled = True
        # print("Simulation started...")

        GPU_list = [True for i in range(0, self.lp)]
        link_list = [True for i in range(0, self.lp+1)]
        data_list = [False for i in range(0, self.num_batch)]

        heappush(event_queue, (root.duration, counter, root))
        if debug:
            print("{} enqueued at time {} batch id {}".format(root.name, 0, root.batch_id))
        ready_list.remove(root)
        counter = counter + 1
        # GPU_list[gid] = False

        # print("Start simulation...")
        # print("root: {}.{}".format(root.name, root.op_id))
        # for i in GPU_list:
        #    if i:
        #        print "_",
        #    else:
        #        print "A",
        #    print " ",
        # print " | ",
        # for i in link_list:
        #    if i:
        #        print "_",
        #    else:
        #        print "A",
        # print

        while len(event_queue) > 0:
            time, _, event = heappop(event_queue)
            event.done = True
            event.scheduled = False
            event.finish_time = time
            if debug:
                print("Event {} finished at time {}".format(event.name, time))

            for child in event.children:
                # if debug:
                #     print("child {}  ready at time {} ".format(child.name, time))
                is_ready = True
                max_time = -1
                for parent in child.parents:
                    if parent.done == False:
                        is_ready = False
                    else:
                        max_time = max(max_time, parent.finish_time)
                # if is_ready == True:
                if is_ready and (child not in ready_list) and (not child.done) and (not child.scheduled):
                    ready_list.append(child)
                    if debug:
                        print("child {}  ready at time {} ".format(child.name, time))

            if isinstance(event, Node):
                GPU_list[event.hw_id] = True
            elif isinstance(event, Edge):
                link_list[event.link_id] = True
            
            # if isinstance(event, Data_batch):
                



            for event in ready_list[:]:
                enqueued = False
                if isinstance(event, Data_batch):
                    # if GPU_list[event.hw_id] == True:
                    new_time = time + event.duration
                    heappush(event_queue, (new_time, counter, event))
                    event.scheduled = True
                    enqueued = True
                    if debug:
                        print("{} enqueued at time".format(event.name,  time))
                    counter = counter + 1
                    data_list[event.batch_id] = True #data batch sent to gpu
                    # GPU_list[event.hw_id] = False
                    ready_list.remove(event)

                elif isinstance(event, Node): 
                    # print("Node event")
                    # print("Node event hw id ", event.hw_id, "name ", event.name)
                    if GPU_list[event.hw_id] == True:
                        new_time = time + event.duration
                        heappush(event_queue, (new_time, counter, event))
                        event.scheduled = True
                        enqueued = True
                        if debug:
                            print("{}.{} enqueued at time {} at device {}".format(event.name, event.op_id, time, event.hw_id))
                        counter = counter + 1
                        GPU_list[event.hw_id] = False
                        ready_list.remove(event)
                elif isinstance(event, Edge): 
                    if event.link_id < 0:
                        new_time = time + event.duration
                        heappush(event_queue, (new_time, counter, event))
                        event.scheduled = True
                        if debug:
                            print("{}.{} enqueued at time {} at device {}".format(event.name, event.op_id, time, event.link_id))
                        enqueued = True
                        counter = counter + 1
                        ready_list.remove(event)
                        
                    elif link_list[event.link_id] == True: #todo:what if all reduce and cross layer transfer overlap
                        new_time = time + event.duration
                        heappush(event_queue, (new_time, counter, event))
                        event.scheduled = True
                        enqueued = True
                        if debug:
                            print("{}.{} enqueued at time {} at link {}".format(event.name, event.op_id, time, event.link_id))
                        counter = counter + 1
                        # if event.is_all_reduce == True: #when all reduce every link is busy
                        #     link_list[event.link_id] = False
                        #     for i in range(self.lp):
                        #         link_list[i] = False
                        # else: #single link between 2 gpus busy
                        link_list[event.link_id] = False
                        ready_list.remove(event)
                elif isinstance(event, Gradient):
                    # print("Gradient event")
                    new_time = time + event.duration
                    heappush(event_queue, (new_time, counter, event))
                    event.scheduled = True
                    enqueued = True
                    if debug:
                        print("{}.{} enqueued at time {} at device {}".format(event.name, event.op_id, time, event.hw_id))
                    counter = counter + 1
                    ready_list.remove(event)

        return time
    # def save_graph(self, output_folder = "output_graph/"):
    #     fw_roots = self.construct_fwd_graph()
    #     time_fw = self.simulate(fw_roots[0], 0)
    #     if not os.path.exists(output_folder):
    #         os.makedirs(output_folder)

    #     filename = "fwd_graph_s%s_l%s_lp%s" % (self.num_seq, self.num_layer, self.lp)
    #     filename_bwd = "bwd_graph_s%s_l%s_lp%s" % (self.num_seq, self.num_layer, self.lp)
    #     dot_fw = visualize_graph(fw_roots[0], filename=filename)
    #     dot_fw.render(output_folder + filename, format="png", cleanup=True)
    #     print("Forward graph saved to %s%s.png" % (output_folder , filename))
    #     print("Forward simulation time: {}".format(time_fw))

    #     bw_roots = self.construct_bwd_graph()

    #     time_bw = self.simulate(bw_roots[0], self.lp - 1)   
    #     dot_bw = visualize_graph(bw_roots[0], filename=filename + "_bwd")
    #     dot_bw.render(output_folder + filename_bwd , format="png", cleanup=True)
    #     print("Backward graph saved to %s%s.png" % (output_folder , filename_bwd))

    #     print("Backward simulation time: {}".format(time_bw))
    #     return time_fw , time_bw

    def save_graph(self, roots, output_folder = "output_graph/", filename="graph"):
        dot_fw = visualize_graph(roots, filename=output_folder + filename)
        dot_fw.render(output_folder + filename , format="png", cleanup=True)
        # dot_bw = visualize_graph(roots, filename=output_folder + filename + "_bwd")
        # dot_bw.render(output_folder + filename + "_bwd" , format="png", cleanup=True)
        # print("Forward graph saved to %s%s.png" % (output_folder , filename))
        print("graph saved to %s%s.png" % (output_folder , filename ))

# dedeepyo : 27-May-25 : Print DFS traversal of the graph.
def print_graph(root_nodes, visited=None):
    if visited is None:
        visited = set()

    for node in root_nodes:
        if node in visited:
            continue
        visited.add(node)

        node_type = "Node" if isinstance(node, Node) else "Edge" if isinstance(node, Edge) else "Data_batch" if isinstance(node, Data_batch) else "Gradient" if isinstance(node, Gradient) else "Unknown"
        print(f"{node_type}: {node.name}, op_id: {node.op_id}, hw_id: {node.hw_id}, duration: {node.duration}")

        for child in node.children:
            child_type = "Node" if isinstance(child, Node) else "Edge" if isinstance(child, Edge) else "Data_batch" if isinstance(child, Data_batch) else "Gradient" if isinstance(child, Gradient) else "Unknown"
            print(f"  --> {child_type}: {child.name}, op_id: {child.op_id}")

        # Recursively print the children
        print_graph(node.children, visited)

def total_duration(root, visited=None):
    if visited is None:
        visited = set()
    if root in visited:
        return 0
    visited.add(root)

    # Only add duration if it's a Node (not a Stage)
    duration = root.compute_time if isinstance(root, Node) else 0

    for child in root.children:
        duration += total_duration(child, visited)

    return duration

def visualize_graph(root, filename="graph", visited=None, dot=None):
    if visited is None:
        visited = set()
    if dot is None:
        dot = Digraph(comment='Computation Graph')

    if root in visited:
        return dot
    visited.add(root)
    if isinstance(root, Node):
        root_type = "Node"
        if root.fwd:
            color = "lightblue"
        else:
            color = "lightcoral"
    elif isinstance(root, Data_batch):
        root_type = "Data_batch"
        color = "gray"
    elif isinstance(root, Edge):
        root_type = "Edge"
        if root.is_all_reduce == True:
            color = "green"
        else:
            color = "yellow"
    elif isinstance(root, Gradient):
        root_type = "Gradient"
        color = "white"

    node_id = str(id(root))
    if isinstance(root, Data_batch):
            label = f"{root.name}\n( batch_id={root.batch_id}, dur={root.duration})"
    elif isinstance(root, Node):
            label = f"{root.name}\n(op_id={root.op_id}, hw_id={root.hw_id}, dur={root.duration})"
    elif isinstance(root, Edge):
            label = f"{root.name}\n(op_id={root.op_id}, link_id={root.link_id}, dur={root.duration})"
    elif isinstance(root, Gradient):
            label = f"{root.name}\n(op_id={root.op_id}, link_id={root.hw_id}, dur={root.duration})"
    # color = "lightblue" if isinstance(root, Node) else "gray" if isinstance(root, Data_batch) else "lightgreen"

    dot.node(node_id, label=label, style='filled', fillcolor=color, shape='box')

    for child in root.children:
        child_id = str(id(child))
        if isinstance(child, Data_batch):
            child_label = f"{child.name}\n( batch_id={child.batch_id}, dur={child.duration})"
        elif isinstance(child, Edge):
            child_label = f"{child.name}\n(op_id={child.op_id}, link_id={child.link_id}, dur={child.duration})"
        elif isinstance(child, Node):
            child_label = f"{child.name}\n(op_id={child.op_id}, hw_id={child.hw_id}, dur={child.duration})"
        elif isinstance(child, Gradient):
            # print("Gradient event in visualization")
            child_label = f"{child.name}\n(op_id={child.op_id}, link_id={child.hw_id}, dur={child.duration})"

        child_color = "lightblue" if isinstance(child, Node) and child.fwd else "lightcoral" if isinstance(child, Node) and not child.fwd else "yellow" if isinstance(child, Data_batch) else "green" if isinstance(child, Edge) and child.is_all_reduce else "white"

        dot.node(child_id, label=child_label, style='filled', fillcolor=child_color, shape='box')
        dot.edge(node_id, child_id)
        visualize_graph(child, filename, visited, dot)

    return dot


# dedeepyo : 27-May-25

def main():
    
    g = Graph(
        # num_seq=7,
        num_layer=2,
        num_batch=3,
        lp=2,
        all_reduce="every layer",  # "the end"  #"every layer"

        T_linear_softmax_f=0,
        T_linear_softmax_b=0,
        T_transformer_f=1,
        T_transformer_b=1,
        T_embedding_f=0,

        
        Tf=2,
        T_embedding_b=0,

        Tb=2,
        T_reduction_transformer=0,
        T_grad_transformer=0,
        T_reduction_embedding=0,
        T_reduction_linear_softmax=0,
        comm_metadata={
            'transformer': {'size': 1024, 'type': 'all_reduce', 'participants': 4, 'local_comp_time': 1, 'interconnect_type': 'dp'},
            'embedding': {'size': 512, 'type': 'all_reduce', 'participants': 4, 'local_comp_time': 1, 'interconnect_type': 'dp'},
            'softmax': {'size': 2048, 'type': 'all_reduce', 'participants': 4, 'local_comp_time': 1, 'interconnect_type': 'dp'},
            'cross_layer': {'size': 4096, 'type': 'point_to_point', 'participants': 2, 'local_comp_time': 0, 'interconnect_type': 'lp'},
        },
    )
    fw_roots = g.construct_fwd_bwd_graph()
    # fw_root = g.convert_comm_sizes_to_times(fw_roots[0], self.network_model, interconnect_params)
    # bw_root = g.convert_comm_sizes_to_times(bw_roots[0], self.network_model, interconnect_params)

    g.save_graph(fw_roots[0], output_folder="output_graph/", filename= "fw_bw")


if __name__ == "__main__":
    main()
