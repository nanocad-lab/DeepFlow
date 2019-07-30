import util

class Topology:
  def __init__(self, net_config, exp_config):
    #Topology Params
    self.topology           = net_config.topology

    #Parallelization Params
    self.lp                 = exp_config.sch_config.lp
    self.dp                 = exp_config.sch_config.dp
    h1                      = exp_config.sch_config.kp_hidden_dim1
    s1                      = exp_config.sch_config.kp_softmax_dim1
    e1                      = exp_config.sch_config.kp_embedding_dim1
    p1                      = exp_config.sch_config.kp_projection_dim1
    h2                      = exp_config.sch_config.kp_hidden_dim2
    s2                      = exp_config.sch_config.kp_softmax_dim2
    e2                      = exp_config.sch_config.kp_embedding_dim2
    p2                      = exp_config.sch_config.kp_projection_dim2

    self.kp_dim             = max(h1, h2, s1, s2, p1, p2, e1, e2)

    data                    = (net_config.parallelMap.data and 
                               self.dp > 1)
    layer                   = (net_config.parallelMap.layer and
                               self.lp > 1)
    kernel                  = (net_config.parallelMap.kernel and
                               self.kp_dim > 1)
    
    self.num_links          = self.calcNumLinks(data, layer, kernel)

    

  def calcLinks1D(self, x, y, z):
    num_links = 0
    if (x > 2):
      assert(y==1 and z==1)
      num_links = x
    elif (y > 2):
      assert(x==1 and z==1)
      num_links = y
    elif (z > 2):
      assert(x==1 and y==1)
      num_links = z
    elif (x == 2):
      assert(y==1 and z==1)
      num_links = 1
    elif (y == 2):
      assert(x==1 and z==1)
      num_links = 1
    elif (z == 2):
      assert(y==1 and x==1)
      num_links = 1
    else:
      printError("Something went wrong!")
    return num_links

  def calcLinks2D(self, x, y):
    num_links = 0
    if (x > 2 and y > 2):
      num_links = 2 * x * y
    elif (x > 2 and y == 2):
      num_links = 3 * x
    elif (y > 2 and x == 2):
      num_links = 3 * y
    elif (x == 2 and y == 2):
      num_links = 4
    return num_links


  def calcLinks3D(self, x, y, z):
    num_links = 0
    if (x > 2 and y > 2 and z > 2):
      num_links = 3 * self.dp * kp_dim * self.lp
    elif (x > 2 and y > 2 and z == 2):
      num_links = 5 * x * y
    elif (x > 2 and z > 2 and y == 2):
      num_links = 5 * x * z
    elif (y > 2 and z > 2 and x == 2):
      num_links = 5 * y * z
    elif (x > 2 and y == 2 and z == 2):
      num_links = 8 * x
    elif (y > 2 and x == 2 and z == 2):
      num_links = 8 * y
    elif (z > 2 and y == 2 and x == 2):
      num_links = 8 * z
    elif (x == 2 and y == 2 and z == 2):
      num_links = 8
    else:
      printError("Something went wrong!")
    return num_links

  #Calculating the total number of links based on the network topology
  #and parallelism strategy
  def calcNumLinks(self, data, layer, kernel):
    grid_dim = 0
    num_links = 0
    
    if ("torus" in self.topology):
        if data:
          grid_dim += 1
        
        if layer:
          grid_dim += 1

        if kernel:
          grid_dim += 1

        if (grid_dim == 1):
            num_links = self.calcLinks1D(self.dp, self.lp, self.kp_dim)
        elif (grid_dim == 2):
            assert((data and kernel and layer) == False)
            if (data and kenrel):
              num_links = self.calcLinks2D(self.dp, self.kp_dim)
            if (data and layer):
              num_links = self.calcLinks2D(self.dp, self.lp)
            if (layer and kernel):
              num_links = self.calcLinks2D(self.lp, self.kp_dim)

        elif (grid_dim == 3):
            assert((data and kernel and layer) == True)
            num_links = self.calcLinks3D(self.dp, self.kp_dim, self.lp)
       
        return num_links
        print("num_links: {}".format(num_links))

    elif (self.topology == "tree"):
        return NotImplemented
    else: 
        print("Topology: mesh/tree")
        return NotImplemented

