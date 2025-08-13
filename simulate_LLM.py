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
        # Re,
        # Rc,
        # Rs,
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
        # self.Re = Re
        # self.Rc = Rc
        # self.Rs = Rs

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
        softmax_node = [linear_softmax]      # 
        # for t in range(self.num_seq):
        #     softmax_node.append(Node("softmax", op_id, self.lp - 1, self.Sf))
        #     op_id += 1

        # for t in range(self.num_seq):
        #   embedding_node.append(Node("embedding", t, 0,          self.Ef))
        #   softmax_node  .append(Node("softmax",   t, self.lp-1,  self.Sf))

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

        # for t in range(self.num_seq):
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
        """
    Build the backward DAG for a decoder-only Transformer whose forward path
    is: qkv_proj -> attn_score -> attn out -> output proj -> ffn1 -> ffn2.

    """

        op_id = 0

        # ---------- 0. Containers ----------
        qkv_projection_b   = [[None for _ in range(self.num_seq)] for _ in range(self.num_layer)]
        
        attention_score_b  = [[None for _ in range(self.num_seq)] for _ in range(self.num_layer)]
        attention_scale_softmax_b = [[None for _ in range(self.num_seq)] for _ in range(self.num_layer)]
        attention_output_b = [[None for _ in range(self.num_seq)] for _ in range(self.num_layer)]
        output_proj_b      = [[None for _ in range(self.num_seq)] for _ in range(self.num_layer)]
        residual1_b        = [[None for _ in range(self.num_seq)] for _ in range(self.num_layer)]
        layer_norm1_b      = [[None for _ in range(self.num_seq)] for _ in range(self.num_layer)]
        ffn1_b  = [[None for _ in range(self.num_seq)] for _ in range(self.num_layer)]
        ffn2_b  = [[None for _ in range(self.num_seq)] for _ in range(self.num_layer)]
        residual2_b        = [[None for _ in range(self.num_seq)] for _ in range(self.num_layer)]
        layer_norm2_b      = [[None for _ in range(self.num_seq)] for _ in range(self.num_layer)]
        # linear_softmax_b = [[None for _ in range(self.num_seq)] for _ in range(self.num_layer)]
        
        # reduce_ = [None for _ in range(self.num_layer)]              # one per layer

        linear_softmax_b = []     # return value

        # ---------- 1. Softmax-B roots ----------
        # for t in range(self.num_seq):
        n = Node("linear_softmax_b", op_id, self.lp - 1, self.T_linear_softmax_b)       # last GPU
        op_id += 1
        linear_softmax_b.append(n)

        # ---------- 2. Per-layer, per-token backward nodes ----------
        for l in reversed(range(self.num_layer)):                    # top → bottom
            # hw_id = l // self.layer_per_device + 1                  # same GPU as fwd
            hw_id = min(l // self.layer_per_device + 1, self.lp - 1)

            # add one All-Reduce node for this layer (sync grads)
            # reduce_node = Node("reduce_layer{}".format(l), op_id, hw_id, self.Rc)
            # op_id += 1
            # reduce_[l] = reduce_node

            # for t in range(self.num_seq):
                # ---------- 2-a. intra-layer forward order ----------
                # create nodes for this layer, this token
                # (note: these are in reverse order of forward path)
            layer_n_2_b = Node("layer_norm2_b", op_id, hw_id, self.T_layer_norm2_b); op_id += 1
            r2b = Node("residual2_b", op_id, hw_id, self.T_residual2_b); op_id += 1
            f2b = Node("ffn2_b", op_id, hw_id, self.T_ffn2_b); op_id += 1
            f1b = Node("ffn1_b", op_id, hw_id, self.T_ffn1_b); op_id += 1
            a_out_b = Node("attn_out_b", op_id, hw_id, self.T_attn_out_b); op_id += 1
            o_p_b  = Node("output_proj_b", op_id, hw_id, self.T_out_b); op_id += 1
            r1b = Node("residual1_b", op_id, hw_id, self.T_residual1_b); op_id += 1
            layer_n_1_b = Node("layer_norm1_b", op_id, hw_id, self.T_layer_norm1_b); op_id += 1
            attn_scale_soft_b = Node("attn_scale_softmax_b", op_id, hw_id, self.T_attn_S_b); op_id += 1
            attn_score_b  = Node("attn_score_b",  op_id, hw_id, self.T_attn_score_b); op_id += 1
            qb  = Node("qkv_b",   op_id, hw_id, self.T_qkv_b);  op_id += 1

            # store for later vertical edges
            
            ffn2_b[l][0] = f2b
            ffn1_b[l][0] = f1b
            attention_score_b[l][0] = attn_score_b
            qkv_projection_b[l][0] = qb
            attention_output_b[l][0] = a_out_b
            output_proj_b[l][0] = o_p_b
            residual2_b[l][0], layer_norm2_b[l][0] = r2b, layer_n_2_b
            residual1_b[l][0], layer_norm1_b[l][0] = r1b, layer_n_1_b
            attention_scale_softmax_b[l][0] = attn_scale_soft_b

            # ---------- 2-a. intra-layer backward order ----------
            layer_n_2_b.add_child(r2b)
            r2b.add_child(f2b)
            f2b.add_child(f1b)
            f1b.add_child(layer_n_1_b)
            layer_n_1_b.add_child(r1b)
            r1b.add_child(o_p_b)
            o_p_b.add_child(a_out_b)
            a_out_b.add_child(attn_scale_soft_b)
            attn_scale_soft_b.add_child(attn_score_b)
            attn_score_b.add_child(qb)
                
                # attn_out_b.add_child(output_proj_b)
                # output_proj_b.add_child(r1b)
                # # f1b.add_child(ob)
                # output_proj_b.add_child(attn_out_b)
                # # aob.add_child(asb)
                # attn_score_b.add_child(qb)

                # send parameter grads to All-Reduce
                # qb.add_child(reduce_node)

                # ---------- 2-b. connect to Softmax_b (only for top layer) ----------
                # if l == self.num_layer - 1:
                #     softmax_b_roots[t].add_child(f2b)   # gradient starts here

                # for t in range(self.num_seq):
                #     # Add backward softmax
                #     scale_softmax_b = Node("scale_softmax_b", op_id, hw_id, self.T_attn_S_b)
                #     op_id += 1
                #     attn_score_b.add_child(scale_softmax_b)
                #     scale_softmax_b.add_child(attn_out_b)

                #     # Add backward residual connection
                #     residual_1_b = Node("residual_1_b", op_id, hw_id, self.T_residual1_b)
                #     op_id += 1
                #     layer_n_1_b.add_child(residual_1_b)
                #     residual_1_b.add_child(output_proj_b)
                #     output_proj_b.add_child(attn_out_b)
                    
                #     residual2_b 

        # ---------- 3. vertical cross-layer edges ----------
        for l in reversed(range(1, self.num_layer)):                # (L-1) → 1
            prev_qkv_b = qkv_projection_b[l][0]     # any token is fine – same hw_id
            curr_layer_n_2_b = layer_norm2_b[l-1][0]  # previous layer's layer_norm2
            # curr_ffn2_b = ffn2_b[l-1][0]

            same_gpu = prev_qkv_b.hw_id == curr_layer_n_2_b.hw_id
            edge = Edge("cross_layer_b", op_id,
                    -1 if same_gpu else 0,
                    0  if same_gpu else self.Tb)
            op_id += 1
            prev_qkv_b.add_child(edge)
            edge.add_child(curr_layer_n_2_b)

        # ---------- 4. Embedding backward sink ----------
        emb_b = Node("embedding_b", op_id, 0, self.T_embedding_b); op_id += 1
        # connect every layer-0 qkv_b(t) to embedding_b
        # for t in range(self.num_seq):
        if qkv_projection_b[0][0].hw_id == emb_b.hw_id:
            edge = Edge("qkv0_emb_b", op_id, -1, 0)
        else:
            edge = Edge("qkv0_emb_b", op_id, -1, self.Tb)
        op_id += 1
        # connect qkv_b(t) to embedding_b
        qkv_projection_b[0][0].add_child(edge)
        edge.add_child(emb_b)
        # for t in range(self.num_seq):
            # connect every softmax_b(t) to embedding_b
        if linear_softmax_b[0].hw_id == emb_b.hw_id:
            edge = Edge("softmax_layer_n_b", op_id, -1, 0)
        else:
            edge = Edge("softmax_layer_n_b", op_id, -1, self.Tb)
        op_id += 1
        # connect softmax_b(t) to embedding_b
        linear_softmax_b[0].add_child(edge)
        edge.add_child(layer_n_2_b)  # connect to qkv_b(t) for embedding gradient
        # optional: Embedding gradient All-Reduce
        # emb_reduce = Node("reduce_emb", op_id, 0, self.Re); op_id += 1
        # emb_b.add_child(emb_reduce)

        # ---------- 5. return the list of backward roots ----------
        return linear_softmax_b

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
    color = "lightblue" if isinstance(root, Node) else "lightgreen"

    dot.node(node_id, label=label, style='filled', fillcolor=color, shape='box')

    for child in root.children:
        child_id = str(id(child))
        child_label = f"{child.name}\n(op_id={child.op_id}, hw_id={child.hw_id}, dur={child.duration})"
        child_color = "lightblue" if isinstance(child, Node) else "lightgreen"
        dot.node(child_id, label=child_label, style='filled', fillcolor=child_color, shape='box')
        dot.edge(node_id, child_id)
        visualize_graph(child, filename, visited, dot)

    return dot

    
    
# dedeepyo : 27-May-25

def main():
    g = Graph(
        num_seq=7,
        num_layer=1,
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
        
        # Cf=2,
        
        Tf=3,
        T_embedding_b=5,
        # Cb=2,
        
        Tb=3,
        # Re=5,
        # Rc=10,
        # Rs=5,
    )

    g.save_graph()

    
    
    

if __name__ == "__main__":
    main()
