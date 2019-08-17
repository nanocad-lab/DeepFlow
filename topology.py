import util

class Topology:
  def __init__(self, net_config, exp_config):
    self.parallelMap     = exp_config.system_hierarchy.device_placement.par2Dev
    self.interNodeDegree, self.intraNodeDegree = self.findMaxDegree(self.parallelMap)
    self.inter_frac = self.interNodeDegree / (self.intraNodeDegree + self.interNodeDegree)
    self.intra_frac = self.intraNodeDegree / (self.intraNodeDegree + self.interNodeDegree)
  
  def findMaxDegree(self):
    interNodeDegree = 0
    intraNodeDegree = 0

    return NotImplemented
    #return interNodeDegree, intraNodeDegree

  def get_fractions(self):
    return self.inter_frac, self.intra_frac
  
  def getDataThroughput(self):
    return NotImplemented

  def getKernelThroughput(self):
    return NotImplemented

  def getLayerThroughput(self):
    return NotImplemented


