import util
import sys as _sys

class Topology:
  def __init__(self, exp_config):
    #System parameters
    self.par2Dev            = exp_config.system_config.device_placement.par2Dev
    self.num_wafers         = exp_config.system_config.num_wafers
    self.num_nodes_per_wafer= exp_config.system_config.num_nodes_per_wafer
    self.tot_nodes          = exp_config.system_config.tot_nodes
    self.adj                = [[0 for x in range(self.tot_nodes)] for x in range(self.tot_nodes)]

    #Parallelization Params
    self.lp_dim             = exp_config.sch_config.lp
    self.dp_dim             = exp_config.sch_config.dp
    h1                      = exp_config.sch_config.kp_hidden_dim1
    s1                      = exp_config.sch_config.kp_softmax_dim1
    e1                      = exp_config.sch_config.kp_embedding_dim1
    p1                      = exp_config.sch_config.kp_projection_dim1
    h2                      = exp_config.sch_config.kp_hidden_dim2
    s2                      = exp_config.sch_config.kp_softmax_dim2
    e2                      = exp_config.sch_config.kp_embedding_dim2
    p2                      = exp_config.sch_config.kp_projection_dim2
    self.kp_dim             = max(h1 * h2, s1 * s2, p1 * p2, e1 * e2)

    #Verify system_hierarchy configuration is valid
    try:
      self.sanityCheckSysHierarchy()
    except Exception as e:
      print("Unexpected error occurred during sanity check of system hierarchy:\n"
            "{}".format(e), flush=True)
      _sys.exit(0)

    #Network parameters
    self.data_intra         = True;
    self.kernel_intra       = True;
    self.layer_intra        = True;


    self.mem_frac           = exp_config.perimeter_breakdown.DRAM
    self.inter_frac         = exp_config.perimeter_breakdown.inter_node
    self.intra_frac         = exp_config.perimeter_breakdown.intra_node

    self.createAdjacancyMatrix(kp = self.kp_dim, lp = self.lp_dim, dp = self.dp_dim);
    self.interNodeDegree, self.intraNodeDegree = self.findMaxDegree()
    
    self.intra_par          = True if self.intraNodeDegree > 0 else False
    self.inter_par          = True if self.interNodeDegree > 0 else False
  
    
  
  def sanityCheckSysHierarchy(self):
      assert (self.tot_nodes == self.dp_dim * self.kp_dim * self.lp_dim), "tot_nodes != dp * kp * lp"

      for key, val in self.par2Dev.items():
        wafer_id, node_id = val
        dp, lp, kp = key
        #assert (dp < self.dp_dim), "data shard index out of bound"
        assert (dp < self.dp_dim), "@wafer {}, node {}, data shard index ({}) >= data parallel shards ({})".format(wafer_id, node_id, dp, self.dp_dim)
        assert (kp < self.kp_dim), "@wafer {}, node {}, kernel shard index ({}) >= kernel parallel shards ({})".format(wafer_id, node_id, kp, self.kp_dim)
        assert (lp < self.lp_dim), "@wafer {}, node {}, layer shard index ({}) >= layer parallel shards ({})".format(wafer_id, node_id, lp, self.lp_dim)

  def node_id(self, point):
    wafer_id, node_id = point
    return wafer_id * self.num_nodes_per_wafer + node_id

  def createAdjacancyMatrix(self, kp, lp, dp):
    #0 not connected
    #1 connected internally
    #2 connected externally
    #connect kernel parallel connections
    #Assumption: reduction is performed through ring-all-reduce algorithm
    for i in range(0, dp):
      for j in range(0, lp):
        for k in range(0, kp):
          start_point               = self.par2Dev[(i,j,k)];
          end_point                 = self.par2Dev[(i,j,(k+1) % kp)];

          start_point_id            = self.node_id(start_point);
          end_point_id              = self.node_id(end_point);
          
          if start_point_id != end_point_id:
            start_point_wafer_id,_   = start_point
            end_point_wafer_id,_     = end_point
            self.adj[start_point_id][end_point_id] = \
                        (1 if (start_point_wafer_id == end_point_wafer_id) else 2)
            if start_point_wafer_id != end_point_wafer_id:
              self.kernel_intra = False;
      
    #connect layer parallel connections
    #Assumption: across layers, for a given data shard, each kernel shard
    #need to have connections to all kernel shards in previous layers.
    #FIXME: This can be an overkill depending on the type of kernel parallelism.
    for i in range(0, dp):
      for j in reversed(range(1, lp)):
        for k in range(0, kp):
          end_point        = self.par2Dev[(i,j,k)];
          for m in range(0, kp):
            start_point    = self.par2Dev[(i,j-1,m)];

            start_point_id = self.node_id(start_point);
            end_point_id   = self.node_id(end_point);
          
            if start_point_id != end_point_id:
              start_point_wafer_id,_   = start_point
              end_point_wafer_id,_     = end_point
              self.adj[start_point_id][end_point_id] = \
                          (1 if (start_point_wafer_id == end_point_wafer_id) else 2)
              if start_point_wafer_id != end_point_wafer_id:
                self.layer_intra = False;
  
    #connect data parallel connections
    #Assumption: within a layer, each parallel kernel can be reduced
    for j in range(0, lp):
      for k in range(0, kp):
        for i in range(0, dp):
          start_point = self.par2Dev[(i,j,k)];
          end_point   = self.par2Dev[((i + 1) % dp,j,k)];
          
          start_point_id = self.node_id(start_point);
          end_point_id   = self.node_id(end_point);
          
          if start_point_id != end_point_id:
            start_point_wafer_id,_   = start_point
            end_point_wafer_id,_    = end_point
            self.adj[start_point_id][end_point_id] = \
                        (1 if (start_point_wafer_id == end_point_wafer_id) else 2)
            if start_point_wafer_id != end_point_wafer_id:
              self.data_intra = False;
  
  #Across all wafers, across all nodes, find maximum inter and intra node degree
  def findMaxDegree(self):
    max_interNodeDegree = 0
    max_intraNodeDegree = 0
    for wid in range(0, self.num_wafers):
      for cid in range(0, self.num_nodes_per_wafer):
        nid = self.node_id((wid,cid));
        interNodeDegree = 0
        intraNodeDegree = 0
        for i in range(0, self.tot_nodes):
          if (self.adj[nid][i] == 1):
            intraNodeDegree = intraNodeDegree + 1
          elif (self.adj[nid][i] == 2):
            interNodeDegree = interNodeDegree + 1
        if (interNodeDegree > max_interNodeDegree):
            max_interNodeDegree = interNodeDegree
        if (intraNodeDegree > max_intraNodeDegree):
            max_intraNodeDegree = intraNodeDegree
            
    return max_interNodeDegree, max_intraNodeDegree

  def get_fractions(self):
    return self.inter_frac, self.intra_frac

  #get P2P bandwidth between data shards
  def getDataThroughput(self, intra_bw, inter_bw, intra_lat, inter_lat):
    return ((intra_bw, intra_lat) if self.data_intra 
             else (inter_bw, inter_lat))

  #get P2P bandwidth between kernel shards
  def getKernelThroughput(self, intra_bw, inter_bw, intra_lat, inter_lat):
    return ((intra_bw, intra_lat) if self.kernel_intra 
            else (inter_bw, inter_lat))

  #get P2P bandwidth between layer shards
  def getLayerThroughput(self, intra_bw, inter_bw, intra_lat, inter_lat):
    return ((intra_bw, intra_lat) if self.layer_intra 
            else (inter_bw, inter_lat))

