import math
from heapq import heappush, heappop

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
      obj.parents.append(self)

class Graph:
  def __init__(self, num_seq, num_layer, lp, Ef, Cf, Sf, Tf, Eb, Cb, Sb, Tb, Re, Rc, Rs):
    self.num_seq = num_seq
    self.num_layer = num_layer
    self.lp = lp
    hlp = lp
    if lp > 2:
      hlp = lp - 2
    self.layer_per_device = math.ceil(num_layer / hlp)
    self.Ef = Ef
    self.Cf = Cf
    self.Sf = Sf
    self.Tf = Tf
    self.Eb = Eb
    self.Cb = Cb
    self.Sb = Sb
    self.Tb = Tb
    self.Re = Re
    self.Rc = Rc
    self.Rs = Rs

  def construct_fwd_graph(self):
    embedding_node = []
    rnn_node = []
    softmax_node = []


    #Create Nodes
    for i in range(0, self.num_seq):
      embedding_node.append(Node("embedding", i, 0, self.Ef))
      softmax_node.append(Node("softmax", i, self.lp - 1, self.Sf))
    
    for i in range(0, self.num_layer):
      rnn_node.append([])
      for j in range(0, self.num_seq):
        rnn_node[i].append(Node("lstm_cell", i * self.num_seq + j, (0 if self.lp == 1 else int(i // self.layer_per_device) + 1), self.Cf))


    #Create Edges
    rnn_edge_intra = []
    rnn_edge_cross = []
    E2H_edge = [] #Embedding to Hidden
    H2S_edge = [] #Hidden to Softmax

    #Edges within layer
    for i in range(0, self.num_layer):
      rnn_edge_intra.append([])
      for j in range(0, self.num_seq):
        #hw_id = -1 indicates it is goign through memory
        rnn_edge_intra[i].append(Edge("intra_edge", i * self.num_seq + j, -1, 0)) 
        
    #Edges across layers
    for i in range(0, self.num_seq):
      E2H_edge.append(Edge("E2H_edge", i, (-1 if self.lp == 1 else 0), self.Tf))
      H2S_edge.append(Edge("H2S_edge", i, self.lp-2, self.Tf))
    
    for i in range(0, self.num_layer-1):
      rnn_edge_cross.append([])
      cross_edge = True if (i+1) % self.layer_per_device == 0 and self.lp > 1 else False
      for j in range(0, self.num_seq):
        rnn_edge_cross[i].append(Edge("cross_edge", 
                                        i * self.num_seq + j, 
                                        int((i + 1) // self.layer_per_device) if cross_edge else -1, 
                                        self.Tf if cross_edge else 0)) 
    #Dummy edges across layers within one device 
    #to prioritize scheduling across nodes 
    #that open up the next device scheduling first
    rnn_edge_dummy = []
    for i in range(0, self.num_layer - 1):
      rnn_edge_dummy.append([])
      for j in range(0, self.num_seq - 1):
          rnn_edge_dummy[i].append(Edge("rnn_dummy_edge", i * self.num_seq + j, -1, 0))

    #Dummy Edges between embedding instances to
    #prioritize schedluing across nodes that can opne up next time steps
    embedding_edge_dummy = []
    for i in range(0, self.num_seq - 1):
      embedding_edge_dummy.append(Edge("E_dummy_edge", i, -1, 0))

    #Attach Nodes to Edges Horizontally
    for j in range(0, self.num_layer):
      for i in range(0, self.num_seq-1):
        rnn_node[j][i].add_child(rnn_edge_intra[j][i])
        rnn_edge_intra[j][i].add_child(rnn_node[j][i+1])
    
    #Attach Nodes to Edges Vertically
    for i in range(0, self.num_seq):
      embedding_node[i].add_child(E2H_edge[i])
      E2H_edge[i].add_child(rnn_node[0][i])
      rnn_node[self.num_layer-1][i].add_child(H2S_edge[i])
      H2S_edge[i].add_child(softmax_node[i])
    
    for i in range(0, self.num_layer -1):
      for j in range(0, self.num_seq):
        rnn_node[i][j].add_child(rnn_edge_cross[i][j])
        rnn_edge_cross[i][j].add_child(rnn_node[i+1][j])

    #Attach dummy edges
    for i in range(1, self.num_layer):
      for j in range(0, self.num_seq - 1):
        if i % self.layer_per_device == 0: 
          continue
        rnn_node[i][j].add_child(rnn_edge_dummy[i-1][j])
        rnn_edge_dummy[i-1][j].add_child(rnn_node[i-1][j+1])
    
    for i in range(0, self.num_seq - 1):
      embedding_node[i].add_child(embedding_edge_dummy[i])
      embedding_edge_dummy[i].add_child(embedding_node[i+1])

    return embedding_node


  def construct_bwd_graph(self):
    embedding_node = []
    rnn_node = []
    softmax_node = []

    #Create Nodes
    for i in range(0, self.num_seq):
      embedding_node.append(Node("embedding", i, 0, self.Eb))
      softmax_node.append(Node("softmax", i, self.lp - 1, self.Sb))
    
    for i in range(0, self.num_layer):
      rnn_node.append([])
      for j in range(0, self.num_seq):
        rnn_node[i].append(Node("lstm_cell", i * self.num_seq + j, (0 if self.lp ==1 else int (i // self.layer_per_device) + 1), self.Cb))


    #Create Edges
    rnn_edge_intra = []
    rnn_edge_cross = []
    H2E_edge = [] #Hidden to Embedding
    S2H_edge = [] #Softmax to Hidden

    #Edges within layer
    for i in range(0, self.num_layer):
      rnn_edge_intra.append([])
      for j in range(0, self.num_seq - 1):
        #hw_id = -1 indicates it is goign through memory
        rnn_edge_intra[i].append(Edge("intra_edge", i * self.num_seq + j, -1, 0)) 
        
    #Edges across layers
    for i in range(0, self.num_seq):
      H2E_edge.append(Edge("H2E_edge", i, (-1 if self.lp == 1 else 0), self.Tf))
      S2H_edge.append(Edge("S2H_edge", i, self.lp-2, self.Tf))
    
    for i in range(0, self.num_layer-1):
      rnn_edge_cross.append([])
      cross_edge = True if (i+1) % self.layer_per_device == 0 and self.lp > 1 else False
      for j in range(0, self.num_seq):
        rnn_edge_cross[i].append(Edge("cross_edge", 
                                        i * self.num_seq + j, 
                                        int((i + 1) // self.layer_per_device) if cross_edge else -1, 
                                        self.Tb if cross_edge else 0)) 
    
    #Dummy edges across layers within one device 
    #to prioritize scheduling across nodes 
    #that open up the next device scheduling first
    rnn_edge_dummy = []
    for i in range(0, self.num_layer - 1):
      rnn_edge_dummy.append([])
      for j in range(0, self.num_seq - 1):
          rnn_edge_dummy[i].append(Edge("rnn_dummy_edge", i * self.num_seq + j, -1, 0))


    #Dummy Edges between softmax instances to
    #prioritize schedluing across nodes that can opne up next time steps
    softmax_edge_dummy = []
    for i in range(0, self.num_seq - 1):
      softmax_edge_dummy.append(Edge("S_dummy_edge", i, -1, 0))
      

    #All-Reduce Edges
    R_edge = []

    R_edge.append(Edge("Reduce_Embedding", 0, self.lp - 1, self.Re))

    for i in range(0, self.num_layer):
      R_edge.append(Edge("Reduce_H", i, (0 if self.lp == 1 else int(i // self.layer_per_device) + self.lp) , self.Rc))
    
    R_edge.append(Edge("Reduce_Softmax", 0, 2 * self.lp - 2, self.Rs))

    #Attach Nodes to Edges Horizontally
    for j in range(0, self.num_layer):
      for i in range(self.num_seq - 1, 0, -1):
        rnn_node[j][i].add_child(rnn_edge_intra[j][i-1])
        rnn_edge_intra[j][i-1].add_child(rnn_node[j][i-1])
    
    #Attach Nodes to Edges Vertically
    for i in range(0, self.num_seq):
      rnn_node[0][i].add_child(H2E_edge[i])
      H2E_edge[i].add_child(embedding_node[i])
      softmax_node[i].add_child(S2H_edge[i])
      S2H_edge[i].add_child(rnn_node[self.num_layer-1][i])
    
    for i in range(self.num_layer - 1, 0, -1):
      for j in range(0, self.num_seq):
        rnn_node[i][j].add_child(rnn_edge_cross[i-1][j])
        rnn_edge_cross[i-1][j].add_child(rnn_node[i-1][j])

    #Attach dummy edges
    for i in range(0, self.num_layer - 1):
      for j in range(1, self.num_seq):
        if (i + 1) % self.layer_per_device == 0: 
          continue
        rnn_node[i][j].add_child(rnn_edge_dummy[i][j-1])
        rnn_edge_dummy[i][j-1].add_child(rnn_node[i+1][j-1])
    
    for i in range(1, self.num_seq):
      softmax_node[i].add_child(softmax_edge_dummy[i-1])
      softmax_edge_dummy[i-1].add_child(softmax_node[i-1])

    #Attach All-Reduce Edges
    softmax_node[0].add_child(R_edge[self.num_layer + 1])
    embedding_node[0].add_child(R_edge[0])
    for i in range(0, self.num_layer):
        rnn_node[i][0].add_child(R_edge[i+1])

    return softmax_node

  def simulate(self, root, gid):
    time = 0
    counter = 0
    event_queue = []
    done_list = []
    ready_list = []

    #for r in roots:
    #  ready_list.append(r)
    ready_list.append(root)


    GPU_list = [True for i in range(0, self.lp)]
    link_list = [True for i in range(0, 2 * self.lp - 1)]
    
    heappush(event_queue, (root.duration, counter, root))
    ready_list.remove(root)
    counter = counter + 1
    GPU_list[gid] = False


    #print("Start simulation...")
    #print("root: {}.{}".format(root.name, root.op_id))
    #for i in GPU_list:
    #    if i:
    #        print "_",
    #    else:
    #        print "A",
    #    print " ",
    #print " | ",
    #for i in link_list:
    #    if i:
    #        print "_",
    #    else:
    #        print "A",
    #print

    while len(event_queue) > 0:
        time, _, event = heappop(event_queue)
        event.done = True
        event.finish_time = time

        #if event.name != "intra_edge":
        #  print("{}.{} finished at time {}".format(event.name, event.op_id, time))

        #Update ready_list
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
            GPU_list[event.hw_id] = True;
        else:
            link_list[event.hw_id] = True;
        
        #Schedule work

        #print "READY NODES: ",
        #for event in ready_list:
        #    print event.name,
        #    print " ",
        #    print event.op_id,
        #    print " ",
        #print

        for event in ready_list[:]:
          enqueued = False
          if event.hw_id == -1:
            new_time = time + event.duration
            heappush(event_queue, (new_time, counter, event))
            #print("{}.{} enqueued at time {} at device {}".format(event.name, event.op_id, time, event.hw_id)) 
            enqueued = True
            counter = counter + 1
            ready_list.remove(event)
          elif isinstance(event, Node): 
            if GPU_list[event.hw_id] == True:
              new_time = time + event.duration
              heappush(event_queue, (new_time, counter, event))
              enqueued = True
              #print("{}.{} enqueued at time {} at device {}".format(event.name, event.op_id, time, event.hw_id)) 
              counter = counter + 1
              GPU_list[event.hw_id] = False
              ready_list.remove(event)
          elif isinstance(event, Edge): 
            if link_list[event.hw_id] == True:
              new_time = time + event.duration
              heappush(event_queue, (new_time, counter, event))
              enqueued = True
              #print("{}.{} enqueued at time {} at device {}".format(event.name, event.op_id, time, event.hw_id)) 
              counter = counter + 1
              link_list[event.hw_id] = False
              ready_list.remove(event)
          #if not enqueued:
          #  print "can't schedule: " + event.name + " " + str(event.op_id)

          #for i in GPU_list:
          #    if i:
          #        print "_",
          #    else:
          #        print "A",
          #    print " ",
          #print " | ",
          #for i in link_list:
          #    if i:
          #        print "_",
          #    else:
          #        print "A",
          #print
    return time


def main():
    g = Graph(num_seq = 4, num_layer = 6, lp = 3, Ef = 2, Cf = 2, Sf = 5, Tf = 3, Eb = 5, Cb = 2, Sb = 2, Tb = 3, Re = 5, Rc = 10, Rs = 5)
    
    fw_roots = g.construct_fwd_graph()
    bw_roots = g.construct_bwd_graph()

    time_fw = g.simulate(fw_roots[0], 0)
    
    time_bw = g.simulate(bw_roots[g.num_seq - 1], g.lp + 1)
    
    print("time_fw: {}, time_bw: {}".format(time_fw, time_bw))

if __name__ == "__main__":
    main()







