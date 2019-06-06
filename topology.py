class Topology:
  def __init__(self, exp_config):
    #Topology Params
    self.topology = exp_config.sch_config.topology

    #Parallelization Params
    self.lp    = exp_config.sch_config.lp
    self.hlp    = exp_config.sch_config.hlp
    self.kp_hidden_dim1    = exp_config.sch_config.kp_hidden_dim1
    self.kp_softmax_dim1   = exp_config.sch_config.kp_softmax_dim1
    self.kp_embedding_dim1 = exp_config.sch_config.kp_embedding_dim1
    self.kp_projection_dim1 = exp_config.sch_config.kp_projection_dim1
    self.kp_hidden_dim2    = exp_config.sch_config.kp_hidden_dim2
    self.kp_softmax_dim2   = exp_config.sch_config.kp_softmax_dim2
    self.kp_embedding_dim2 = exp_config.sch_config.kp_embedding_dim2
    self.kp_projection_dim2 = exp_config.sch_config.kp_projection_dim2
    self.dp    = exp_config.sch_config.dp
    self.kp_hidden_type  =  exp_config.sch_config.kp_hidden_type #1: CR, 2: RC
    self.kp_softmax_type  =  exp_config.sch_config.kp_softmax_type #1: CR, 2: RC
    self.kp_embedding_type  =  exp_config.sch_config.kp_embedding_type #1: CR, 2: RC
    self.kp_projection_type  =  exp_config.sch_config.kp_projection_type #1: CR, 2: RC

    self.grid_dim   = 0
    self.num_links  = 0

  #Calculating the total number of links based on the network topology
  #and parallelism strategy
  def getNumLinks(self):
    if (self.topology == "mesh"):
        #Assuming Homogenous Cores, i.e. if we have 
        #3 links out of one core within one layers, 
        #we have 3 links out of every core regardless of if that layers need it.
        self.grid_dim = 0

        if self.dp > 1:
          self.grid_dim += 1
        
        if self.lp > 1:
          self.grid_dim += 1

        h1 = self.kp_hidden_dim1
        h2 = self.kp_hidden_dim2
        
        s1 = self.kp_softmax_dim1
        s2 = self.kp_softmax_dim2
      
        p1 = self.kp_projection_dim1
        p2 = self.kp_projection_dim2

        e1 = self.kp_embedding_dim1
        e2 = self.kp_embedding_dim2

        if(h1 > 1 or s1 > 1 or p1 > 1 or e1 > 1 or h2 > 1 or s2 > 1 or p2 > 1 or e2 > 1):
            self.grid_dim += 1
       
        print("grid_dim: {}".format(self.grid_dim))

        #FIXME: hack! Fix after ISCA deadline.
        self.num_links = 1
        if (self.grid_dim == 1):
            self.num_links = self.dp * h1
        elif (self.grid_dim == 2):
            self.num_links = 2 * self.dp * h1
        else:
            self.num_links = 3 * self.dp * h1 * self.lp
       
        return self.num_links
        print("num_links: {}".format(self.num_links))

    elif (self.topology == "tree"):
        return NotImplemented
    else: 
        print("Topology: mesh/tree")
        return NotImplemented

