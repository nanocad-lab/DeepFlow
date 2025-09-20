import math
from heapq import heappush, heappop
import sys
from typing import Any, Dict
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

    def __repr__(self):
        return f"Node({self.name},op={self.op_id},hw={self.hw_id},{round(self.duration, 2)})"
    
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

    def remove_self_from_children(self):
        for child in self.children:
            child.parents.remove(self)
        self.children = []

class Edge:
  def __init__(self, name, op_id, duration, is_all_reduce=False, comm_size_bytes=0, comm_type=None, participants=1, comm_interconnect_type=None):
    self.name = name
    self.op_id = op_id
    self.duration = duration
    self.done = False
    self.finish_time = -1
    self.parents = [] 
    self.children = []
    self.is_all_reduce = is_all_reduce
    self.scheduled = False
    self.comm_size_bytes = comm_size_bytes
    self.comm_type = comm_type
    self.participants = participants
    self.comm_interconnect_type = comm_interconnect_type

  def add_child(self, obj):
      self.children.append(obj)
      obj.parents.append(self)

  def __repr__(self):
      return f"Edge({self.name},op={self.op_id},{round(self.duration, 2)})"
      
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
        mode: str,
        dp: int,
        lp: int,
        kp1: int,
        kp2: int,
        tp_mode: str,
        comp_times: Dict[str, Any],
        comm_metadata: Dict[str, Any],
        misc_metadata: Dict[str, Any],
    ) -> None:
        self.mode = mode
        self.dp = int(dp)
        self.lp = int(lp)
        self.kp1 = int(kp1) if kp1 else 1
        self.kp2 = int(kp2) if kp2 else 1
        self.tp_mode = tp_mode
        self.comp_times = comp_times or {}
        self.comm_metadata = comm_metadata or {}
        self.misc_metadata = misc_metadata or {}

        self.num_batch = self.misc_metadata.get("num_batch", 0)
        self.num_layer = self.misc_metadata.get("num_layer", 0)
        self.layer_per_device = max(1, math.ceil(self.num_layer / self.lp)) if self.lp else self.num_layer
        self.all_reduce = self.misc_metadata.get("all_reduce", "the end")

        self.transformer_cfg = self.comp_times.get("transformer", {})
        self.T_grad_transformer = self.comp_times.get("grad_transformer", 0.0)

    def _time(self, key: str, default: float = 0.0) -> float:
        value = self.comp_times.get(key)
        return float(value) if value is not None else default

    def create_comm_edge(self, name, op_id, comm_key, is_all_reduce=False, local_hw_id=None):
        """Create a communication edge with optional local computation node.

        Args:
            name: Edge name
            op_id: Operation ID
            comm_key: Key into self.comm_metadata dict

        Returns:
            The communication edge (connection point for graph building)
        """
        comm_data = self.comm_metadata[comm_key]

        # Create communication edge with metadata
        comm_edge = Edge(
            name=name,
            op_id=op_id,
            duration=0,  # Will be filled in second pass
            comm_size_bytes=comm_data['size'],
            comm_type=comm_data['type'],
            participants=comm_data['participants'],
            comm_interconnect_type=comm_data['interconnect_type'],
            is_all_reduce=is_all_reduce
        )

        # If there's local computation time, create local node after edge
        local_comp_time = comm_data.get('local_comp_time', 0)
        if local_comp_time > 0:
            if local_hw_id is None:
                raise ValueError(f"Local compute time requires a hardware id for edge '{name}'")
            local_node = Node(
                name=f"{name}_local_comp",
                op_id=op_id + 100000,  # Offset to avoid ID conflicts
                hw_id=local_hw_id,
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
                    # print(f"Converted {child.name} size {child.comm_size_bytes} bytes to duration {child.duration:.6f} sec using {interconnect_type} (ib={ib}, ll={ll})")

                # Recursively process this child
                traverse_and_convert(child, visited)

        traverse_and_convert(roots)
        return roots

    def construct_fwd_bwd_graph(self):
        embedding_node = []
        data_batch_node = []
        softmax_node = []
        transformer_nodes = [[] for _ in range(self.num_batch)]  # 

        embedding_b_time = self._time("embedding_b")
        linear_softmax_b_time = self._time("linear_softmax_b")
        transformer_b_time = self._time("transformer_b")
        linear_softmax_f_time = self._time("linear_softmax_f")
        linear_softmax_b_time = self._time("linear_softmax_b")
        transformer_f_time = self._time("transformer_f")
        transformer_b_time = self._time("transformer_b")
        embedding_f_time = self._time("embedding_f")
        embedding_b_time = self._time("embedding_b")
        cross_layer_time = self._time("cross_layer_f")


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
            linear_softmax = Node(f"linear_softmax{b}", op_id, self.lp-1, linear_softmax_f_time)
            op_id += 1
            softmax_node.append(linear_softmax)
            emb = Node(f"embeddding{b}", op_id, 0, embedding_f_time)      # hw_id = 0
            op_id += 1
            embedding_node.append(emb)
            data_batch_node[b].add_child(embedding_node[b])

            for l in range(self.num_layer):
                transformer_nodes[b].append([])

                hw_id = min(l // self.layer_per_device , self.lp - 1)#assign hw_id for transformer
                transformer_node = Node("transformer", op_id, hw_id, transformer_f_time)
                transformer_node.micro_batch_index = b
                transformer_node.layer_index = l
                transformer_node.direction = "forward"
                transformer_node.stage_id = hw_id
                op_id += 1

                transformer_nodes[b][l]=transformer_node

            for l in range(1, self.num_layer):

                prev_node = transformer_nodes[b][l-1]        # previous layer
                curr_node  = transformer_nodes[b][l]            # current layer


                if prev_node.hw_id == curr_node.hw_id:
                    edge = Edge("cross_layer", op_id, 0)  # on same GPU
                else:
                    edge = self.create_comm_edge('cross_layer', op_id, 'cross_layer')  # on different GPU
                op_id += 1

                prev_node.add_child(edge); edge.add_child(curr_node)#connect previous layer and current layer

            first_node = transformer_nodes[b][0]   # first layer
            if first_node.hw_id == embedding_node[b].hw_id:
                edge = Edge("Emb_node0", op_id, 0)
            else:
                edge = self.create_comm_edge('cross_layer', op_id, 'cross_layer')
            op_id += 1
            embedding_node[b].add_child(edge) #connect embedding node and first transformer layer
            edge.add_child(first_node)


            last_node = transformer_nodes[b][-1]  # last layer
            if last_node.hw_id == softmax_node[b].hw_id:
                node_Softmax = Edge("node_Softmax", op_id, 0)  # same GPU
            else:
                node_Softmax = self.create_comm_edge('cross_layer', op_id, 'cross_layer')
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

        for db_node in data_batch_node:
            db_node.remove_self_from_children()

        for b in reversed(range(self.num_batch)): #connect each data batch node with corresponding nodes
            emb_b = Node("embeddding_b", op_id, 0, embedding_b_time, fwd=False)      # hw_id = 0
            op_id += 1
            embedding_node_b[b] = emb_b
            linear_softmax_b = Node("linear_softmax_b", op_id, self.lp-1, linear_softmax_b_time, fwd=False)
            op_id += 1
            softmax_node_b[b] = linear_softmax_b
            softmax_node[b].add_child(linear_softmax_b)


            for l in reversed(range(self.num_layer)):

                hw_id = min(l // self.layer_per_device , self.lp - 1)
                transformer_node_b = Node("transformer_b", op_id, hw_id, transformer_b_time, fwd=False)
                transformer_node_b.micro_batch_index = b
                transformer_node_b.layer_index = l
                transformer_node_b.direction = "backward"
                transformer_node_b.stage_id = hw_id
                op_id += 1
                transformer_nodes_b[b][l] = transformer_node_b

            for l in reversed(range(1, self.num_layer)):
                curr_node = transformer_nodes_b[b][l]         # current layer's qkv_proj.
                next_ffn2  = transformer_nodes_b[b][l-1]            # next layer's layernorm.
                if curr_node.hw_id == next_ffn2.hw_id:
                    edge = Edge("cross_layer", op_id, 0)  
                else:
                    edge = self.create_comm_edge('cross_layer', op_id, 'cross_layer')
                op_id += 1

                curr_node.add_child(edge); edge.add_child(next_ffn2)

            qkv_0_b = transformer_nodes_b[b][0]     # first layer's qkv_proj
            if qkv_0_b.hw_id == emb_b.hw_id:
                edge = Edge("Emb_node0", op_id, 0)
            else:
                edge = self.create_comm_edge('cross_layer', op_id, 'cross_layer')
            op_id += 1
            qkv_0_b.add_child(edge)
            edge.add_child(emb_b)


            prev_layer_norm2 = transformer_nodes_b[b][self.num_layer-1] # last layer's layernorm2
            if prev_layer_norm2.hw_id == softmax_node_b[b].hw_id:
                layernorm_Softmax = Edge("layernorm2_Softmax", op_id, 0)  # same GPU
            else:
                layernorm_Softmax = self.create_comm_edge('cross_layer', op_id, 'cross_layer')
            op_id += 1
            softmax_node_b[b].add_child(layernorm_Softmax)
            layernorm_Softmax.add_child(prev_layer_norm2)
                

            # all-reduce
            if self.dp > 1:
                R_edge[b].append(self.create_comm_edge("embedding", op_id, "embedding", is_all_reduce=True))
                op_id += 1
                if self.all_reduce == "the end":
                    R_edge[b].append(
                        self.create_comm_edge(
                            "transformer",
                            op_id,
                            "transformer",
                            is_all_reduce=True,
                            local_hw_id=transformer_nodes_b[b][-1].hw_id,
                        )
                    )
                    op_id += 1

                    R_edge[b].append(self.create_comm_edge("softmax", op_id, "softmax", is_all_reduce=True))
                    op_id += 1
                # Attach All-Reduce Edges
                    softmax_node_b[b].add_child(R_edge[b][-1])
                    embedding_node_b[b].add_child(R_edge[b][0])
                    transformer_nodes_b[b][0].add_child(R_edge[b][1])

                elif self.all_reduce == "every layer":
                    for i in range(0, self.num_layer):
                        R_edge[b].append(
                            self.create_comm_edge(
                                "transformer",
                                op_id,
                                "transformer",
                                is_all_reduce=True,
                                local_hw_id=transformer_nodes_b[b][i].hw_id,
                            )
                        )
                        op_id += 1
                        # G_edge[b].append(Gradient("Grad_transformer", op_id, transformer_nodes_b[b][-1].hw_id, self.T_grad_transformer))
                        # op_id += 1
                        # R_edge[b][-1].add_child(G_edge[b][-1])

                    R_edge[b].append(self.create_comm_edge("softmax", op_id, "softmax", is_all_reduce=True))
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

        return embedding_node[0]

    def construct_transformer_graph(self, direction: str = "both"):
        transformer_cfg = self.transformer_cfg
        gemm_entries = transformer_cfg.get("gemms")
        if not gemm_entries:
            raise ValueError("Transformer GEMM times not provided")

        tp_degree = int(transformer_cfg.get("tp_degree", max(1, self.kp1 * self.kp2)))

        root = Data_batch("transformer_root", 0, 0)
        op_id = 0

        for rank in range(tp_degree):
            previous = root

            if direction in {"forward", "both"}:
                for idx, entry in enumerate(gemm_entries):
                    entry_name = entry.get("name", f"g{idx}")
                    forward_cfg = entry.get("forward", {})
                    fwd_duration = forward_cfg.get("duration")
                    if fwd_duration is None:
                        raise ValueError("Transformer GEMM entry missing forward duration")

                    node = Node(
                        name=f"{entry_name}_fwd_rank{rank}",
                        op_id=op_id,
                        hw_id=rank,
                        duration=fwd_duration,
                        fwd=True,
                    )
                    op_id += 1
                    previous.add_child(node)
                    previous = node

                    for comm_idx, comm_key in enumerate(forward_cfg.get("comm_keys", [])):
                        if comm_key not in self.comm_metadata:
                            raise KeyError(f"Missing transformer comm metadata for key '{comm_key}'")
                        comm_type = self.comm_metadata[comm_key]['type']
                        comm_edge = self.create_comm_edge(
                            name=comm_key,
                            op_id=op_id,
                            comm_key=comm_key,
                            is_all_reduce=(comm_type == 'all_reduce'),
                            local_hw_id=rank,
                        )
                        op_id += 1
                        previous.add_child(comm_edge)
                        previous = comm_edge

            if direction in {"backward", "both"}:
                for idx, entry in enumerate(reversed(gemm_entries)):
                    entry_name = entry.get("name", f"g{idx}")
                    backward_cfg = entry.get("backward", {})
                    bwd_duration = backward_cfg.get("duration")
                    if bwd_duration is None:
                        raise ValueError("Transformer GEMM entry missing backward duration")

                    node = Node(
                        name=f"{entry_name}_bwd_rank{rank}",
                        op_id=op_id,
                        hw_id=rank,
                        duration=bwd_duration,
                        fwd=False,
                    )
                    op_id += 1
                    previous.add_child(node)
                    previous = node

                    for comm_idx, comm_key in enumerate(backward_cfg.get("comm_keys", [])):
                        if comm_key not in self.comm_metadata:
                            raise KeyError(f"Missing transformer comm metadata for key '{comm_key}'")
                        comm_type = self.comm_metadata[comm_key]['type']
                        comm_edge = self.create_comm_edge(
                            name=comm_key,
                            op_id=op_id,
                            comm_key=comm_key,
                            is_all_reduce=(comm_type == 'all_reduce'),
                            local_hw_id=rank,
                        )
                        op_id += 1
                        previous.add_child(comm_edge)
                        previous = comm_edge

        return root
        
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
                    new_time = time + event.duration
                    heappush(event_queue, (new_time, counter, event))
                    event.scheduled = True
                    if debug:
                        print("{}.{} enqueued at time {}".format(event.name, event.op_id, time))
                    enqueued = True
                    counter = counter + 1
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
            label = f"{root.name}\n(op_id={root.op_id}, dur={root.duration})"
    elif isinstance(root, Gradient):
            label = f"{root.name}\n(op_id={root.op_id}, hw_id={root.hw_id}, dur={root.duration})"
    # color = "lightblue" if isinstance(root, Node) else "gray" if isinstance(root, Data_batch) else "lightgreen"

    dot.node(node_id, label=label, style='filled', fillcolor=color, shape='box')

    for child in root.children:
        child_id = str(id(child))
        if isinstance(child, Data_batch):
            child_label = f"{child.name}\n( batch_id={child.batch_id}, dur={child.duration})"
        elif isinstance(child, Edge):
            child_label = f"{child.name}\n(op_id={child.op_id}, dur={child.duration})"
        elif isinstance(child, Node):
            child_label = f"{child.name}\n(op_id={child.op_id}, hw_id={child.hw_id}, dur={child.duration})"
        elif isinstance(child, Gradient):
            child_label = f"{child.name}\n(op_id={child.op_id}, hw_id={child.hw_id}, dur={child.duration})"

        child_color = "lightblue" if isinstance(child, Node) and child.fwd else "lightcoral" if isinstance(child, Node) and not child.fwd else "yellow" if isinstance(child, Data_batch) else "green" if isinstance(child, Edge) and child.is_all_reduce else "white"

        dot.node(child_id, label=child_label, style='filled', fillcolor=child_color, shape='box')
        dot.edge(node_id, child_id)
        visualize_graph(child, filename, visited, dot)

    return dot
