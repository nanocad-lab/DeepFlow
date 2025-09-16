import math
from heapq import heappush, heappop
from graphviz import Digraph
import os
class Node:

    def __init__(self, name, op_id, hw_id, duration):
        self.name = name
        self.op_id = op_id
        self.hw_id = hw_id
        self.duration = duration
        self.done = False
        self.finish_time = -1
        self.parents = []
        self.children = []

    def add_child(self, obj):
        self.children.append(obj)
        obj.parents.append(self)


class Edge:
  def __init__(self, name, op_id, hw_id, duration, comm_size_bytes=0, comm_type=None, participants=1, comm_interconnect_type=None):
    self.name = name
    self.op_id = op_id
    self.hw_id = hw_id
    self.duration = duration
    self.comm_size_bytes = comm_size_bytes
    self.comm_type = comm_type
    self.participants = participants
    self.comm_interconnect_type = comm_interconnect_type  # "dp", "lp", "kp1", "kp2"
    self.done = False
    self.finish_time = -1
    self.parents = []
    self.children = []
  
  def add_child(self, obj):
      self.children.append(obj)

class Graph:

    def __init__(
        self,
        num_seq,
        num_layer,
        T_qkv_projection_f,
        T_qkv_projection_b,
        T_attention_score_f,
        T_attention_score_b,
        T_attention_scale_softmax_f,  # Add softmax timing
        T_attention_scale_softmax_b,  # Add softmax timing
        T_attention_output_f,
        T_attention_output_b,
        T_out_proj_f,
        T_out_proj_b,
        T_ffn1_f,
        T_ffn1_b,
        T_ffn2_f,
        T_ffn2_b,
        T_residual1_f,          # Add residual timing
        T_residual1_b,
        T_layer_norm1_f,        # Add layer norm timing
        T_layer_norm1_b,      # Add layer norm timing
        T_residual2_f,          # Add residual timing
        T_residual2_b,
        T_layer_norm2_f,        # Add layer norm timing
        T_layer_norm2_b,      # Add layer norm timing
        T_linear_softmax_f,     # Add linear softmax timing
        T_linear_softmax_b,     # Add linear softmax timing
        lp,
        T_embedding_f,
        # Cf,

        Tf,
        T_embedding_b,
        # Cb,

        Tb,
        comm_metadata,

    ):
        self.num_seq = num_seq
        self.num_layer = num_layer
        self.lp = lp
        self.T_qkv_f = T_qkv_projection_f
        self.T_qkv_b = T_qkv_projection_b
        self.T_attn_score_f= T_attention_score_f
        self.T_attn_score_b = T_attention_score_b
        self.T_attn_S_f = T_attention_scale_softmax_f
        self.T_attn_S_b = T_attention_scale_softmax_b
        self.T_attn_out_f = T_attention_output_f
        self.T_attn_out_b = T_attention_output_b
        self.T_out_f = T_out_proj_f
        self.T_out_b = T_out_proj_b
        self.T_ffn1_f = T_ffn1_f
        self.T_ffn1_b = T_ffn1_b
        self.T_ffn2_f = T_ffn2_f
        self.T_ffn2_b = T_ffn2_b
        self.T_residual1_f = T_residual1_f
        self.T_residual1_b = T_residual1_b
        self.T_layer_norm1_f = T_layer_norm1_f
        self.T_layer_norm1_b = T_layer_norm1_b
        self.T_residual2_f = T_residual2_f
        self.T_residual2_b = T_residual2_b
        self.T_layer_norm2_f = T_layer_norm2_f
        self.T_layer_norm2_b = T_layer_norm2_b
        self.T_linear_softmax_f = T_linear_softmax_f
        self.T_linear_softmax_b = T_linear_softmax_b
        hlp = lp
        if lp > 2:
            hlp = lp - 2
        self.layer_per_device = math.ceil(num_layer / hlp)
        self.T_embedding_f = T_embedding_f
        # self.Cf = Cf
        # self.Sf = Sf
        self.Tf = Tf
        self.T_embedding_b = T_embedding_b
        # self.Cb = Cb
        # self.Sb = Sb
        self.Tb = Tb
        self.comm_metadata = comm_metadata

    def create_comm_edge(self, name, op_id, hw_id, comm_key):
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
            hw_id=hw_id,
            duration=0,  # Will be filled in second pass
            comm_size_bytes=comm_data['size'],
            comm_type=comm_data['type'],
            participants=comm_data['participants'],
            comm_interconnect_type=comm_data['interconnect_type']
        )

        # If there's local computation time, create local node after edge
        local_comp_time = comm_data.get('local_comp_time', 0)
        if local_comp_time > 0:
            local_node = Node(
                name=f"{name}_local_comp",
                op_id=op_id + 100000,  # Offset to avoid ID conflicts
                hw_id=hw_id,
                duration=local_comp_time
            )
            comm_edge.add_child(local_node)

        return comm_edge

    def convert_comm_sizes_to_times(self, network_model, interconnect_params):
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

                # Recursively process this child
                traverse_and_convert(child, visited)

        # We need to traverse both forward and backward graphs
        # Start from all possible root nodes
        try:
            fw_roots = self.construct_fwd_graph()
            for root in fw_roots:
                traverse_and_convert(root)
        except:
            pass  # In case forward graph fails

        try:
            bw_roots = self.construct_bwd_graph()
            for root in bw_roots:
                traverse_and_convert(root)
        except:
            pass  # In case backward graph fails

    def construct_fwd_graph(self):
        embedding_node = []
        # softmax_node = []

        #######
        xform_node = []  # 
        edge_list = []

        op_id = 0  # operation ID, used to distinguish nodes and edges
        # ① Create ONE embedding node
        emb = Node("embeddding", op_id, 0, self.T_embedding_f)      # hw_id = 0
        op_id += 1
        embedding_node = [emb]      # 
        linear_softmax = Node("linear_softmax", op_id, self.lp-1, self.T_linear_softmax_f); op_id += 1

        # ② Create ONE softmax node (batch × sequence)
        # soft = Node("softmax", op_id, self.lp - 1, self.Sf)  # hw_id = lp-1
        # op_id += 1
        softmax_node = [linear_softmax]      
        for l in range(self.num_layer):
            xform_node.append([])

            hw_id = min(l // self.layer_per_device + 1, self.lp - 1)

            # for t in range(self.num_seq):
            # 6 gemm nodes
            qkv_proj  = Node("qkv_proj",  op_id, hw_id, self.T_qkv_f);  op_id += 1
            attn_score = Node("attn_score", op_id, hw_id, self.T_attn_score_f); op_id += 1
            attn_scale_softmax = Node("attn_scale_softmax", op_id, hw_id, self.T_attn_S_f); op_id += 1  # Add softmax node
            attn_out  = Node("attn_out", op_id, hw_id, self.T_attn_out_f); op_id += 1
            output_proj = Node("output_proj", op_id, hw_id, self.T_out_f); op_id += 1
            residual1 = Node("residual1", op_id, hw_id, self.T_residual1_f); op_id += 1  # Add residual node
            layer_norm1 = Node("layer_norm1", op_id, hw_id, self.T_layer_norm1_f); op_id += 1  # Add layer norm node

            ffn1 = Node("ffn1", op_id, hw_id, self.T_ffn1_f); op_id += 1
            ffn2 = Node("ffn2", op_id, hw_id, self.T_ffn2_f); op_id += 1
            residual2 = Node("residual2", op_id, hw_id, self.T_residual2_f); op_id += 1  # Add residual node
            layer_norm2 = Node("layer_norm2", op_id, hw_id, self.T_layer_norm2_f); op_id += 1

            qkv_proj.add_child(attn_score)  ; edge_list.append((qkv_proj, attn_score))
            attn_score.add_child(attn_scale_softmax) ; edge_list.append((attn_score, attn_scale_softmax))
            attn_scale_softmax.add_child(attn_out) ; edge_list.append((attn_scale_softmax, attn_out))
            attn_out.add_child(output_proj) ; edge_list.append((attn_out, output_proj))
            output_proj.add_child(residual1) ; edge_list.append((output_proj, residual1))
            residual1.add_child(layer_norm1) ; edge_list.append((residual1, layer_norm1))
            layer_norm1.add_child(ffn1) ; edge_list.append((layer_norm1, ffn1))
            ffn1.add_child(ffn2) ; edge_list.append((ffn1, ffn2))
            ffn2.add_child(residual2) ; edge_list.append((ffn2, residual2))
            residual2.add_child(layer_norm2) ; edge_list.append((residual2, layer_norm2))
            # layer_norm2.add_child(linear_softmax) ; edge_list.append((layer_norm2, linear_softmax))

            xform_node[l].append([qkv_proj, attn_score, attn_scale_softmax, attn_out, output_proj, residual1, layer_norm1, ffn1, ffn2, residual2, layer_norm2])

        for l in range(1, self.num_layer):
            # for t in range(self.num_seq):
            prev_ffn2 = xform_node[l-1][0][-1]        # previous layer's ffn2
            curr_qkv  = xform_node[l][0][0]            # current layer's qkv_proj

            if prev_ffn2.hw_id == curr_qkv.hw_id:
                edge = Edge("cross_layer", op_id, -1, 0)  
            else:
                edge = Edge("cross_layer", op_id, -1, self.Tf)  
            op_id += 1

            prev_ffn2.add_child(edge); edge.add_child(curr_qkv)

        qkv_0_t = xform_node[0][0][0]     # first layer's qkv_proj
        if qkv_0_t.hw_id == emb.hw_id:
            edge = Edge("Emb_qkv0", op_id, -1, 0)
        else:
            edge = Edge("Emb_qkv0", op_id, -1, self.Tf)
        op_id += 1
        emb.add_child(edge)
        edge.add_child(qkv_0_t)

        # for t in range(self.num_seq):

        prev_ffn2 = xform_node[-1][0][-1]  # last layer's ffn2
        if prev_ffn2.hw_id == softmax_node[0].hw_id:
            ffn2_Softmax = Edge("ffn2_Softmax", op_id, -1, 0)  # same GPU
        else:
            ffn2_Softmax = Edge("ffn2_Softmax", op_id, -1, self.Tf)
        op_id += 1
        prev_ffn2.add_child(ffn2_Softmax)
        ffn2_Softmax.add_child(softmax_node[0])

        return embedding_node
    def construct_bwd_graph(self):
        embedding_node = []
        softmax_node = []

        #######
        xform_node = []  # 
        edge_list = []

        op_id = 0  # operation ID, used to distinguish nodes and edges
        # ① Create ONE embedding node
        emb = Node("embeddding", op_id, 0, self.T_embedding_b)      # hw_id = 0
        op_id += 1
        embedding_node = [emb]      # 
        linear_softmax = Node("linear_softmax", op_id, self.lp-1, self.T_linear_softmax_b); op_id += 1

        # ② Create ONE softmax node (batch × sequence)
        # soft = Node("softmax", op_id, self.lp - 1, self.Sf)  # hw_id = lp-1
        # op_id += 1
        softmax_node = [linear_softmax]    
        while len(xform_node) <= self.num_layer :
            xform_node.append([])  
        for l in reversed(range(self.num_layer)):         
            # xform_node.append([])

            hw_id = min(l // self.layer_per_device + 1, self.lp - 1)

            # for t in range(self.num_seq):
            # 6 gemm nodes
            qkv_proj  = Node("qkv_proj",  op_id, hw_id, self.T_qkv_b);  op_id += 1
            attn_score = Node("attn_score", op_id, hw_id, self.T_attn_score_b); op_id += 1
            attn_scale_softmax = Node("attn_scale_softmax", op_id, hw_id, self.T_attn_S_b); op_id += 1  # Add softmax node
            attn_out  = Node("attn_out", op_id, hw_id, self.T_attn_out_b); op_id += 1
            output_proj = Node("output_proj", op_id, hw_id, self.T_out_b); op_id += 1
            residual1 = Node("residual1", op_id, hw_id, self.T_residual1_b); op_id += 1  # Add residual node
            layer_norm1 = Node("layer_norm1", op_id, hw_id, self.T_layer_norm1_b); op_id += 1  # Add layer norm node

            ffn1 = Node("ffn1", op_id, hw_id, self.T_ffn1_b); op_id += 1
            ffn2 = Node("ffn2", op_id, hw_id, self.T_ffn2_b); op_id += 1
            residual2 = Node("residual2", op_id, hw_id, self.T_residual2_b); op_id += 1  # Add residual node
            layer_norm2 = Node("layer_norm2", op_id, hw_id, self.T_layer_norm2_b); op_id += 1

            attn_score.add_child(qkv_proj)  ; edge_list.append((attn_score,qkv_proj ))
            attn_scale_softmax.add_child(attn_score) ; edge_list.append((attn_scale_softmax, attn_score))
            attn_out.add_child(attn_scale_softmax) ; edge_list.append((attn_out, attn_scale_softmax))
            output_proj.add_child(attn_out) ; edge_list.append((output_proj, attn_out))
            residual1.add_child(output_proj) ; edge_list.append((residual1, output_proj))
            layer_norm1.add_child(residual1) ; edge_list.append((layer_norm1, residual1))
            ffn1.add_child(layer_norm1) ; edge_list.append((ffn1, layer_norm1))
            ffn2.add_child(ffn1) ; edge_list.append((ffn2, ffn1))
            residual2.add_child(ffn2) ; edge_list.append((residual2, ffn2))
            layer_norm2.add_child(residual2) ; edge_list.append((layer_norm2, residual2))

            # qkv_proj.add_child(attn_score)  ; edge_list.append((qkv_proj, attn_score))
            # attn_score.add_child(attn_scale_softmax) ; edge_list.append((attn_score, attn_scale_softmax))
            # attn_scale_softmax.add_child(attn_out) ; edge_list.append((attn_scale_softmax, attn_out))
            # attn_out.add_child(output_proj) ; edge_list.append((attn_out, output_proj))
            # output_proj.add_child(residual1) ; edge_list.append((output_proj, residual1))
            # residual1.add_child(layer_norm1) ; edge_list.append((residual1, layer_norm1))
            # layer_norm1.add_child(ffn1) ; edge_list.append((layer_norm1, ffn1))
            # ffn1.add_child(ffn2) ; edge_list.append((ffn1, ffn2))
            # ffn2.add_child(residual2) ; edge_list.append((ffn2, residual2))
            # residual2.add_child(layer_norm2) ; edge_list.append((residual2, layer_norm2))
            # layer_norm2.add_child(linear_softmax) ; edge_list.append((layer_norm2, linear_softmax))

            xform_node[l].append([layer_norm2, residual2, ffn2, ffn1, layer_norm1, residual1, output_proj, attn_out, attn_scale_softmax, attn_score, qkv_proj     ])

        for l in reversed(range(1, self.num_layer)):
            # for t in range(self.num_seq):
            curr_qkv = xform_node[l][0][-1]         # current layer's qkv_proj.
            next_ffn2  = xform_node[l-1][0][0]            # next layer's layernorm.

            if curr_qkv.hw_id == next_ffn2.hw_id:
                edge = Edge("cross_layer", op_id, -1, 0)  
            else:
                edge = Edge("cross_layer", op_id, -1, self.Tf)  
            op_id += 1

            curr_qkv.add_child(edge); edge.add_child(next_ffn2)

        qkv_0_t = xform_node[0][0][-1]     # first layer's qkv_proj
        if qkv_0_t.hw_id == emb.hw_id:
            edge = Edge("Emb_qkv0", op_id, -1, 0)
        else:
            edge = Edge("Emb_qkv0", op_id, -1, self.Tf)
        op_id += 1
        qkv_0_t.add_child(edge)
        edge.add_child(emb)

        # for t in range(self.num_seq):

        prev_layer_norm2 = xform_node[self.num_layer-1][0][0]  # last layer's layernorm2
        if prev_layer_norm2.hw_id == softmax_node[0].hw_id:
            layernorm_Softmax = Edge("layernorm2_Softmax", op_id, -1, 0)  # same GPU
        else:
            layernorm_Softmax = Edge("layernorm2_Softmax", op_id, -1, self.Tf)
        op_id += 1
        softmax_node[0].add_child(layernorm_Softmax)
        layernorm_Softmax.add_child(prev_layer_norm2)
        # all-reduce


        R_edge = []

        R_edge.append(self.create_comm_edge("Reduce_Embedding", 0, self.lp - 1, "embedding"))

        for i in range(0, self.num_layer):
            R_edge.append(self.create_comm_edge("Reduce_transformer", i, (0 if self.lp == 1 else int(i // self.layer_per_device) + self.lp), "transformer"))

        R_edge.append(self.create_comm_edge("Reduce_Softmax", 0, 2 * self.lp - 2, "softmax"))
        # Attach All-Reduce Edges
        softmax_node[0].add_child(R_edge[self.num_layer + 1])
        embedding_node[0].add_child(R_edge[0])
        for i in range(0, self.num_layer):
            xform_node[i][0][-1].add_child(R_edge[i + 1])

        return softmax_node

    def simulate(self, root, gid):
        time = 0
        counter = 0
        event_queue = []
        done_list = []
        ready_list = []

        # for r in roots:
        #  ready_list.append(r)
        ready_list.append(root)

        GPU_list = [True for i in range(0, self.lp)]
        link_list = [True for i in range(0, 2 * self.lp - 1)]

        heappush(event_queue, (root.duration, counter, root))
        ready_list.remove(root)
        counter = counter + 1
        GPU_list[gid] = False

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
            event.finish_time = time

            # if event.name != "intra_edge":
            #  print("{}.{} finished at time {}".format(event.name, event.op_id, time))

            # Update ready_list
            for child in event.children:
                is_ready = True
                max_time = -1
                for parent in child.parents:
                    if parent.done == False:
                        is_ready = False
                    else:
                        max_time = max(max_time, parent.finish_time)
                if is_ready == True:
                    ready_list.append(child)

            if isinstance(event, Node):
                GPU_list[event.hw_id] = True
            else:
                link_list[event.hw_id] = True

            # Schedule work

            # print "READY NODES: ",
            # for event in ready_list:
            #    print event.name,
            #    print " ",
            #    print event.op_id,
            #    print " ",
            # print

            for event in ready_list[:]:
                enqueued = False
                if event.hw_id == -1:
                    new_time = time + event.duration
                    heappush(event_queue, (new_time, counter, event))
                    # print("{}.{} enqueued at time {} at device {}".format(event.name, event.op_id, time, event.hw_id))
                    enqueued = True
                    counter = counter + 1
                    ready_list.remove(event)
                elif isinstance(event, Node): 
                    if GPU_list[event.hw_id] == True:
                        new_time = time + event.duration
                        heappush(event_queue, (new_time, counter, event))
                        enqueued = True
                        # print("{}.{} enqueued at time {} at device {}".format(event.name, event.op_id, time, event.hw_id))
                        counter = counter + 1
                        GPU_list[event.hw_id] = False
                        ready_list.remove(event)
                elif isinstance(event, Edge): 
                    if link_list[event.hw_id] == True:
                        new_time = time + event.duration
                        heappush(event_queue, (new_time, counter, event))
                        enqueued = True
                        # print("{}.{} enqueued at time {} at device {}".format(event.name, event.op_id, time, event.hw_id))
                        counter = counter + 1
                        link_list[event.hw_id] = False
                        ready_list.remove(event)
                # if not enqueued:
                #  print "can't schedule: " + event.name + " " + str(event.op_id)

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
        return time
    def save_graph(self, output_folder = "output_graph/"):
        fw_roots = self.construct_fwd_graph()
        time_fw = self.simulate(fw_roots[0], 0)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        filename = "fwd_graph_s%s_l%s_lp%s" % (self.num_seq, self.num_layer, self.lp)
        filename_bwd = "bwd_graph_s%s_l%s_lp%s" % (self.num_seq, self.num_layer, self.lp)
        dot_fw = visualize_graph(fw_roots[0], filename=filename)
        dot_fw.render(output_folder + filename, format="png", cleanup=True)
        print("Forward graph saved to %s%s.png" % (output_folder , filename))
        print("Forward simulation time: {}".format(time_fw))

        bw_roots = self.construct_bwd_graph()

        time_bw = self.simulate(bw_roots[0], self.lp - 1)   
        dot_bw = visualize_graph(bw_roots[0], filename=filename + "_bwd")
        dot_bw.render(output_folder + filename_bwd , format="png", cleanup=True)
        print("Backward graph saved to %s%s.png" % (output_folder , filename_bwd))

        print("Backward simulation time: {}".format(time_bw))
        return time_fw , time_bw

# dedeepyo : 27-May-25 : Print DFS traversal of the graph.
def print_graph(root_nodes, visited=None):
    if visited is None:
        visited = set()

    for node in root_nodes:
        if node in visited:
            continue
        visited.add(node)

        node_type = "Node" if isinstance(node, Node) else "Edge"
        print(f"{node_type}: {node.name}, op_id: {node.op_id}, hw_id: {node.hw_id}, duration: {node.duration}")

        for child in node.children:
            child_type = "Node" if isinstance(child, Node) else "Edge"
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

    node_id = str(id(root))
    label = f"{root.name}\n(op_id={root.op_id}, hw_id={root.hw_id}, dur={root.duration})"
    color = "lightblue" if isinstance(root, Node) else "green"

    dot.node(node_id, label=label, style='filled', fillcolor=color, shape='box')

    for child in root.children:
        child_id = str(id(child))
        child_label = f"{child.name}\n(op_id={child.op_id}, hw_id={child.hw_id}, dur={child.duration})"
        child_color = "lightblue" if isinstance(child, Node) else "green"
        dot.node(child_id, label=child_label, style='filled', fillcolor=child_color, shape='box')
        dot.edge(node_id, child_id)
        visualize_graph(child, filename, visited, dot)

    return dot


# dedeepyo : 27-May-25

def main():
    g = Graph(
        num_seq=7,
        num_layer=3,
        lp=1,
        T_qkv_projection_f=1,
        T_qkv_projection_b=1,
        T_attention_score_f=1,
        T_attention_score_b=1,
        T_attention_scale_softmax_f=1,  # Add softmax timing
        T_attention_scale_softmax_b=1,  # Add softmax timing
        T_attention_output_f=1,
        T_attention_output_b=1,
        T_out_proj_f=1,
        T_out_proj_b=1,
        T_ffn1_f=1,
        T_ffn1_b=1,
        T_ffn2_f=1,
        T_ffn2_b=1,
        T_residual1_f=1,
        T_residual1_b=1,
        T_layer_norm1_f=1,
        T_layer_norm1_b=1,
        T_residual2_f=1,
        T_residual2_b=1,
        T_layer_norm2_f=1,
        T_layer_norm2_b=1,
        T_linear_softmax_f=1,
        T_linear_softmax_b=1,
        
        
        T_embedding_f=5,
        

        
        Tf=3,
        T_embedding_b=5,
        
        Tb=3,
        comm_metadata={
            'transformer': {'size': 1024, 'type': 'all_reduce', 'participants': 4, 'local_comp_time': 2, 'interconnect_type': 'dp'},
            'embedding': {'size': 512, 'type': 'all_reduce', 'participants': 4, 'local_comp_time': 2, 'interconnect_type': 'dp'},
            'softmax': {'size': 2048, 'type': 'all_reduce', 'participants': 4, 'local_comp_time': 2, 'interconnect_type': 'dp'}
        },
    )

    g.save_graph()


if __name__ == "__main__":
    main()
