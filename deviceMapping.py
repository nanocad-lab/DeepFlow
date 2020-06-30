from itertools import permutations 
import math
import functools 
import operator 


class Point():
    def __init__(self, x, y, wid):
        self.x = x
        self.y = y
        self.wid = wid

class Block():
    def __init__(self, dim_x, dim_y):
        self.dim_x = dim_x
        self.dim_y = dim_y
class Size():
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Coverage():
    wafer_dim = 0
    def __init__(self, start, size, wafers):
        self.start = start
        self.size = size
        self.wafers = wafers

        if len(wafers) == 1:
            end_x = start.x + size.x
            end_y = start.y + size.y
            wid = -1
        else:
            end_x = self.wafer_dim
            end_y = self.wafer_dim
            wid = wafers[-1]

        self.end = Point(end_x, end_y, wid)

class Projection():
    def __init__(self, dp, kp1, kp2, lp, wafer_dim, num_wafer):
        self.dp = dp
        self.kp1 = kp1
        self.kp2 = kp2
        self.lp = lp
        self.wafer_dim = wafer_dim
        self.num_wafer = num_wafer
        self.par2Dev = []
        self.dev2Par = []
        self.x_edges = []
        self.y_edges = []
        self.cross_edges = []
        self.index_order = []
        self.dim_order = []

        nw = int(math.ceil(dp * kp1 * kp2 * lp / float(wafer_dim * wafer_dim)))
        assert(num_wafer == nw)
        #print(kp1, kp2, dp, lp, num_wafer, nw)
        #Parallelism strategies
        self.ps = [self.kp1, self.kp2, self.dp, self.lp]

        for i in range(0, num_wafer):
            self.x_edges.append([0] * (wafer_dim * wafer_dim))
            self.y_edges.append([0] * (wafer_dim * wafer_dim))
            self.cross_edges.append([0] * (wafer_dim))
        

        non_one_list = [i for i, value in enumerate(self.ps) if value != 1] 
        one_list = [i for i, value in enumerate(self.ps) if value == 1] 

        order = [list(l) + one_list for l in list(permutations(non_one_list))]
        dim = []
        for l in order:
            dim.append([self.ps[i] for i in l])


        unique_dim = [list(x) for x in set(tuple(x) for x in dim)]
        unique_index = [dim.index(x) for x in unique_dim]
        unique_order = [order[i] for i in unique_index]

        self.order = unique_order
        self.dim = unique_dim

        
        self.par2Dev = [None] * len(self.order)
        self.dev2Par = [None] * len(self.order)

    #Project 4D hypercube to 2D mesh for a given layout order
    def project(self, layout_id):
        ps = self.ps
        order = self.order[layout_id]
        dim = self.dim[layout_id]
        L = self.wafer_dim
        nw = self.num_wafer

        par2Dev = {}
        dev2Par = {}
      
        #print("order: {}".format(order))

        for wid in range(0, self.num_wafer):
            for xid in range(0, self.wafer_dim):
                for yid in range(0, self.wafer_dim):
                    dev2Par[(xid,yid, wid)]={0:-1, 1:-1, 2:-1, 3:-1}

        Coverage.wafer_dim = self.wafer_dim
        num_parallel_workers = self.dp * self.lp * self.kp1 * self.kp2
        
        dim_x = int(math.ceil(num_parallel_workers / float(self.wafer_dim)))
        dim_y = num_parallel_workers % self.wafer_dim if num_parallel_workers % self.wafer_dim != 0 else (self.wafer_dim if num_parallel_workers != 0 else 0)
        if num_parallel_workers > 1:
            dim_x = L
            dim_y = L

        coverage = Coverage(start=Point(x=0, y=0, wid=0), size=Size(x=dim_x, y=dim_y), wafers=range(0,nw))

        self.place(order, coverage, dev2Par)
        self.populatePar2Dev(dev2Par, par2Dev)
        
        self.par2Dev[layout_id] = par2Dev
        self.dev2Par[layout_id] = dev2Par
    
    def populatePar2Dev(self, dev2Par, par2Dev):
        for wid in range(0, self.num_wafer):
            for xid in range(0, self.wafer_dim):
                for yid in range(0, self.wafer_dim):
                    par_dic = dev2Par[(xid,yid, wid)]
                    par_tuple = (par_dic[0], par_dic[1], par_dic[2], par_dic[3])
                    par2Dev[par_tuple] = (xid, yid, wid)
                    #print("({}) mapped to ({},{},{})".format(par_tuple, xid,yid, wid))

    def place(self, order, coverage, dev2Par):
        L = self.wafer_dim
        nw = self.num_wafer
        ps = self.ps

        if len(order) == 0:
            return
        elif len(order) == 1:
            dim_y = ps[order[0]]
            if dim_y <= coverage.size.y:
               block = Block(dim_x=1, dim_y=1)
               parallel_dim = order[0]
               new_order = []
               self.vertical_traversal_placement(block, coverage, parallel_dim, new_order, dev2Par)
            else:
               block = Block(dim_x=1, dim_y = 1)
               parallel_dim = order[0]
               new_order = []
               self.alternate_vertical_placement(block, coverage, parallel_dim, new_order, dev2Par)
        elif len(order) == 2:
            dim_y = ps[order[0]]
            dim_x = ps[order[1]]

            if dim_y <= coverage.size.y:
               block = Block(dim_x=1, dim_y = dim_y)
               parallel_dim = order[1]
               new_order = [order[0]]
               self.alternate_traversal_placement(block, coverage, parallel_dim, new_order, dev2Par)
            else:
               dim_x = (dim_y // coverage.size.y) 
               dim_y = coverage.size.y
               if dim_x <= coverage.size.x:
                    block = Block(dim_x=dim_x, dim_y = dim_y)
                    parallel_dim = order[-1]
                    new_order = order[0:-1]
                    self.horizontal_traversal_placement(block, coverage, parallel_dim, new_order, dev2Par)
               else:
                   #NotImplementedError("Should Implement This (order = 2)")
                   self.wafer_placement(order, coverage, dev2Par)
        elif (len(order) >= 3):
            dim_y = ps[order[0]]
            dim_x = functools.reduce(operator.mul,[ps[i] for i in order[1:-1]])
            if dim_y <= coverage.size.y:
               if dim_x <= coverage.size.x:
                   block = Block(dim_x=dim_x, dim_y = dim_y)
                   parallel_dim = order[-1]
                   new_order = order[0:-1]
                   self.alternate_traversal_placement(block, coverage, parallel_dim, new_order, dev2Par)
               else:
                   dim_y = ps[order[0]] * (dim_x // coverage.size.x) 
                   dim_x = coverage.size.x
                   if dim_y <= coverage.size.y:
                       block = Block(dim_x=dim_x, dim_y = dim_y)
                       parallel_dim = order[-1]
                       new_order = order[0:-1]
                       self.vertical_traversal_placement(block, coverage, parallel_dim, new_order, dev2Par)
                   else:
                       #NotImplementedError("Should Implement This (order = 3)")
                        self.wafer_placement(order, coverage, dev2Par)
            else:
                dim_x = (dim_y // coverage.size.y) * dim_x 
                dim_y = coverage.size.y
                if dim_x <= coverage.size.x:
                    block = Block(dim_x=dim_x, dim_y = dim_y)
                    parallel_dim = order[-1]
                    new_order = order[0:-1]
                    self.horizontal_traversal_placement(block, coverage, parallel_dim, new_order, dev2Par)
                else:
                    #NotImplemented
                    self.wafer_placement(order, coverage, dev2Par)
                    
        else:
            NotImplemented

    def wafer_placement(self, order, coverage, dev2Par):
        parallel_dim = order[-1]
        new_order = order[0:-1]
        nw = len(coverage.wafers)
        assert(nw > 1)
        if nw >= self.ps[parallel_dim]:
            step = nw // self.ps[parallel_dim] 
            index = 0
            for ss in range(0, nw, step):
                wafer_slice = coverage.wafers[ss:ss+step]
                new_coverage = Coverage(start=Point(x=coverage.start.x, y=coverage.start.y, wid=coverage.start.wid), size=Size(x=coverage.size.x, y=coverage.size.y), wafers=wafer_slice)
                for wid in wafer_slice:
                    for ii in range(0, self.wafer_dim, 1):
                        for jj in range(0, self.wafer_dim, 1):
                            dev2Par[(ii,jj,wid)][parallel_dim] = index
                index = index + 1
                self.place(new_order, new_coverage, dev2Par)
        else:
            NotImplemented

    def alternate_traversal_placement(self, block, coverage, parallel_dim, order, dev2Par):
        #assert(len(coverage.wafers) == 1)
        index = 0
        wid = coverage.wafers[0]
        
        while index < self.ps[parallel_dim]:
            moveRight = True
            for j in range(coverage.start.y, coverage.end.y, block.dim_y):
                if moveRight:
                    for i in range(coverage.start.x, coverage.end.x, block.dim_x):
                        for ii in range(i, i + block.dim_x, 1):
                            for jj in range(j, j + block.dim_y, 1):
                                dev2Par[(ii,jj,wid)][parallel_dim] = index
                        index = index + 1
                        new_coverage = Coverage(start=Point(x=i, y=j, wid=wid), size=Size(x=block.dim_x, y=block.dim_y), wafers=[wid])
                        self.place(order[:], new_coverage, dev2Par)
                else:
                    for i in range(coverage.end.x-1, coverage.start.x-1, -1 * block.dim_x):
                        for ii in range(i - block.dim_x + 1, i + 1, 1):
                            for jj in range(j, j + block.dim_y, 1):
                                dev2Par[(ii,jj,wid)][parallel_dim] = index
                        index = index + 1
                        new_coverage = Coverage(start=Point(x=i-block.dim_x+1, y=j, wid=wid), size=Size(x=block.dim_x, y=block.dim_y), wafers=[wid])
                        self.place(order[:], new_coverage, dev2Par)
                moveRight = not moveRight
            wid = wid + 1

    
    def alternate_vertical_placement(self, block, coverage, parallel_dim, new_order, dev2Par):
        #assert(len(coverage.wafers) == 1)
        index = 0
        wid = coverage.wafers[0]
        
        while index < self.ps[parallel_dim]:
            moveSouth = True
            for i in range(coverage.start.x, coverage.end.x, block.dim_x):
                if moveSouth:
                    for j in range(coverage.start.y, coverage.end.y, block.dim_y):
                        for ii in range(i, i + block.dim_x, 1):
                            for jj in range(j, j + block.dim_y, 1):
                                dev2Par[(ii,jj,wid)][parallel_dim] = index
                        index = index + 1
                else:
                    for j in range(coverage.end.y-1, coverage.start.y-1, -1*block.dim_y):
                        for ii in range(i, i + block.dim_x, 1):
                            for jj in range(j - block.dim_y + 1, j + 1, 1):
                                dev2Par[(ii,jj,wid)][parallel_dim] = index
                        index = index + 1
                moveSouth = not moveSouth
            wid = wid + 1
    
    def vertical_traversal_placement(self, block, coverage, parallel_dim, order, dev2Par):
        #assert(len(coverage.wafers) == 1)
        index = 0
        wid = coverage.wafers[0]
       
        while index < self.ps[parallel_dim]:
            for j in range(coverage.start.y, coverage.end.y, block.dim_y):
                for jj in range(j, j+block.dim_y, 1):
                    for ii in range(coverage.start.x, coverage.end.x, 1):
                        dev2Par[(ii,jj,wid)][parallel_dim] = index
                index = index + 1
                new_coverage = Coverage(start=Point(x=coverage.start.x, y=j, wid=wid), size=Size(x=block.dim_x, y=block.dim_y), wafers=[wid])
                self.place(order[:], new_coverage, dev2Par)
            wid = wid + 1
    
    def horizontal_traversal_placement(self, block, coverage, parallel_dim, order, dev2Par):
        #assert(len(coverage.wafers) == 1)
        index = 0
        wid = coverage.wafers[0]
        while index < self.ps[parallel_dim]: 
            for i in range(coverage.start.x, coverage.end.x, block.dim_x):
                for ii in range(i, i + block.dim_x, 1):
                    for jj in range(coverage.start.y, coverage.end.y, 1):
                        dev2Par[(ii,jj,wid)][parallel_dim] = index
                index = index + 1
                new_coverage = Coverage(start=Point(x=i, y=coverage.start.y, wid=wid), size=Size(x=block.dim_x, y=block.dim_y), wafers=[wid])
                self.place(order[:], new_coverage, dev2Par)
            wid = wid + 1
    

    def all_connect(self, par2Dev):
        dp = self.dp
        lp = self.lp
        kp1 = self.kp1
        kp2 = self.kp2

        par2cross = {"kp1": False, 
                     "kp2": False, 
                     "dp": False,
                     "lp": False} 
        
        for i in range(0, lp):
          for j in range(0, dp):
            for k2 in range(0, kp2):
              for k1 in range(0, kp1 if kp1>2 else kp1-1):
                start_point = par2Dev[(k1, k2, j, i)]
                end_point = par2Dev[((k1+1) % kp1, k2, j, i)]
                self.route(start_point, end_point)
                self.updateConnectionType(start_point, end_point, "kp1", par2cross)
       
        for i in range(0, lp):
          for j in range(0, dp):
            for k1 in range(0, kp1):
              for k2 in range(0, kp2 if kp2>2 else kp2-1):
                start_point = par2Dev[(k1, k2, j, i)]
                end_point = par2Dev[(k1, (k2+1) % kp2, j, i)]
                self.route(start_point, end_point)
                self.updateConnectionType(start_point, end_point, "kp2", par2cross)
       
        for k1 in range(0, kp1):
          for j in range(0, dp):
            for k2 in range(0, kp2):
              for i in range(0, lp-1):
                start_point = par2Dev[(k1, k2, j, i)]
                end_point = par2Dev[(k1, k2, j, i+1)]
                self.route(start_point, end_point)
                self.updateConnectionType(start_point, end_point, "lp", par2cross)
       
        for i in range(0, lp):
          for k1 in range(0, kp1):
            for k2 in range(0, kp2):
              for j in range(0, dp if dp>2 else dp-1):
                start_point = par2Dev[(k1, k2, j, i)]
                end_point = par2Dev[(k1, k2, (j+1)%dp, i)]
                self.route(start_point, end_point)
                self.updateConnectionType(start_point, end_point, "dp", par2cross)
        
        return par2cross

    def pidx(self, x, y):
        return y * self.wafer_dim + x

    def pidy(self, x, y):
        return x * self.wafer_dim + y

    def updateConnectionType(self, start, end, pid, par2cross):
        wid_start = start[2]
        wid_end = end[2]

        if wid_start != wid_end:
            par2cross[pid] |= True
        

    def route(self, start, end):
        x_edges = self.x_edges
        y_edges = self.y_edges
        cross_edges = self.cross_edges

        x_start = start[0]
        y_start = start[1]
        wid_start = start[2]
    
        x_end = end[0]
        y_end = end[1]
        wid_end = end[2]

        assert(wid_start < self.num_wafer)
        assert(wid_end < self.num_wafer)
    
        if wid_start == wid_end: #in the same wafer
            if x_start <= x_end:
                x = x_start
                y = y_start
                while (x < x_end): 
                    x_edges[wid_start][self.pidx(x,y)] += 1
                    x = x + 1
                if (y_start < y_end): 
                    while (y < y_end):
                        y_edges[wid_start][self.pidy(x,y)] += 1
                        #if(wid_start == 3 and self.pidy(x,y) == 59):
                        #    print("route {} to {}, y_edges[3][59] = {}".format(start, end, y_edges[wid_start][self.pidy(x,y)]))
                        y = y + 1
                else:
                    while (y >  y_end):
                        y_edges[wid_start][self.pidy(x,y)] += 1
                        #if(wid_start == 3 and self.pidy(x,y) == 59):
                        #    print("route {} to {}, y_edges[3][59] = {}".format(start, end, y_edges[wid_start][self.pidy(x,y)]))
                        y = y - 1
            else:
                x = x_end
                y = y_end
                while (x < x_start):
                    x_edges[wid_start][self.pidx(x,y)] += 1
                    x = x + 1
                if (y_end < y_start):
                    while (y < y_start):
                        y_edges[wid_start][self.pidy(x,y)] += 1
                        #if(wid_start == 3 and self.pidy(x,y) == 59):
                        #    print("route {} to {}, y_edges[3][59] = {}".format(start, end, y_edges[wid_start][self.pidy(x,y)]))
                        y = y + 1
                else:
                    while (y >  y_start):
                        y_edges[wid_start][self.pidy(x,y)] += 1
                        #if(wid_start == 3 and self.pidy(x,y) == 59):
                        #    print("route {} to {}, y_edges[3][59] = {}".format(start, end, y_edges[wid_start][self.pidy(x,y)]))
                        y = y - 1
        else: #not in the same wafer
            #assert(wid_end == ((wid_start + 1) % self.num_wafer) or wid_start == ((wid_end + 1) % self.num_wafer))
            if (wid_start < wid_end):
                x = x_start
                y = y_start
                while (x < self.wafer_dim - 1): 
                    x_edges[wid_start][self.pidx(x,y)] += 1
                    x = x + 1
                if (y_start < y_end): 
                    while (y < y_end):
                        y_edges[wid_start][self.pidy(x,y)] += 1
                        #if(wid_start == 3 and self.pidy(x,y) == 59):
                        #    print("route {} to {}, y_edges[3][59] = {}".format(start, end, y_edges[wid_start][self.pidy(x,y)]))
                        y = y + 1
                else:
                    while (y > y_end):
                        y_edges[wid_start][self.pidy(x,y)] += 1
                        #if(wid_start == 3 and self.pidy(x,y) == 59):
                        #    print("route {} to {}, y_edges[3][59] = {}".format(start, end, y_edges[wid_start][self.pidy(x,y)]))
                        y = y - 1
                x = 0
                y = y_end
                cross_edges[wid_start][y_end] += 1
                while (x < x_end): 
                    x_edges[wid_end][self.pidx(x,y)] += 1
                    x = x + 1
            else:
                x = x_end
                y = y_end
                while (x < self.wafer_dim - 1):
                    x_edges[wid_end][self.pidx(x,y)] += 1
                    x = x + 1
                if (y_end < y_start):
                    while (y < y_start):
                        y_edges[wid_start][self.pidy(x,y)] += 1
                        #if(wid_start == 3 and self.pidy(x,y) == 59):
                        #    print("route {} to {}, y_edges[3][59] = {}".format(start, end, y_edges[wid_start][self.pidy(x,y)]))
                        y = y + 1
                else:
                    while (y >  y_start):
                        y_edges[wid_start][self.pidy(x,y)] += 1
                        #if(wid_start == 3 and self.pidy(x,y) == 59):
                        #    print("route {} to {}, y_edges[3][59] = {}".format(start, end, y_edges[wid_start][self.pidy(x,y)]))
                        y = y - 1
                x = 0
                y = y_start
                cross_edges[wid_end][y_start] += 1
                while (x < x_start): 
                    x_edges[wid_start][self.pidx(x,y)] += 1
                    x = x + 1
        
        #if wid_start == 1 or wid_end == 1:
        #    if y_start == 0 or y_end == 0:
        #        print("route {} to {}".format(start, end))
        #        for eid in range(0, self.wafer_dim):
        #            print("e{}: {}".format(eid, x_edges[1][eid]))

    def checkMaxDegree(self):
        max_x = 0
        max_y = 0
        max_cross = 0
        dim = self.wafer_dim
        nw = self.num_wafer
        x_edges = self.x_edges
        y_edges = self.y_edges
        cross_edges = self.cross_edges

        #for wid in range(0, nw):
        #    for eid in range(0, dim):
        #        print("w{} e{}: {}".format(wid, eid, cross_edges[wid][eid]))

        for wid in range(0, self.num_wafer):
            for eid in range(0, dim * dim):
                max_x = max(max_x, x_edges[wid][eid])
                max_y = max(max_y, y_edges[wid][eid])
            for eid in range(0, dim):
                max_cross = max(max_cross, cross_edges[wid][eid])

        return max_cross, max_x, max_y

    def get_derate_factors(self, layout_id):
        par = ["kp1","kp2","dp","lp"]
        derate_factor_inter = []
        derate_factor_intra = []
        #par2cross_list = []

        ps = self.ps
        
        #print(layout_id)
        order = self.order[layout_id]
        #print("Parallelism order: {}({})-----{}({})-----{}({})-----{}({})".format(par[order[0]], ps[order[0]], par[order[1]], ps[order[1]], par[order[2]], ps[order[2]],par[order[3]], ps[order[3]]))
        
        #[wafer-2-wafer, x_edge, x_edge]
        for wid in range(0, self.num_wafer):
            for eid in range(0, self.wafer_dim * self.wafer_dim):
                self.x_edges[wid][eid] = 0
                self.y_edges[wid][eid] = 0
            for eid in range(0, self.wafer_dim):
                self.cross_edges[wid][eid] = 0

        par2cross = self.all_connect(self.par2Dev[layout_id])
        #par2cross_list.append(self.par2cross)
        #for key, val in self.par2cross.items():
        #    print("{}, {}".format(key, "inter" if val else "intra"))
        w_factor, x_factor, y_factor = self.checkMaxDegree()
        derate_factor_inter = w_factor
        derate_factor_intra = max(x_factor, y_factor)
        #print("w_factor: {}, x_factor: {}, y_factor: {}".format(w_factor, x_factor, y_factor))
        #print

        return derate_factor_inter, derate_factor_intra, par2cross

   
def main():
    for kp1 in [4]: #[1, 16, 32]:
        for kp2 in [2]: #[1, 16, 32]:
            for dp in [1]: #[1, 2, 4, 8]:
                for lp in [1]: #[2]:
                    print("==========")
                    print("({},{},{},{})".format(kp1, kp2, dp, lp))
                    print("==========")
                    nw = int(math.ceil(dp * kp1 * kp2 * lp))
                    p = Projection(dp = dp, kp1 = kp1, kp2 = kp2, lp = lp, wafer_dim = 1, num_wafer = nw)
                    for layout_id in range(0, len(p.order)):
                        p.project(layout_id)
                        derate_factor_inter, derate_factor_intra, par2cross = p.get_derate_factors(layout_id)
                        print("layout: {}, derate_factor_intra: {}, derate_factor_inter: {}, cross_wafer: {}".format(layout_id,derate_factor_intra,derate_factor_inter,par2cross))
                        print
if __name__ == "__main__":
    main()
