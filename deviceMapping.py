from itertools import permutations 
import math

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
        #Parallelism strategies
        self.ps = [self.kp1, self.kp2, self.dp, self.lp]

        for i in range(0, num_wafer):
            self.x_edges.append([0] * (wafer_dim * wafer_dim))
            self.y_edges.append([0] * (wafer_dim * wafer_dim))
            self.cross_edges.append([0] * (wafer_dim))
        
        #Per parallelism type check the communication type
        #if it crosses across wafers or not
        self.par2cross = {"kp1": False, 
                          "kp2": False, 
                          "dp": False, 
                          "lp": False}

    #Project 4D hypercube to 2D mesh
    def project(self):
        ps = self.ps
        order = []
        dim = []
        
        non_one_list = [i for i, value in enumerate(ps) if value != 1] 
        one_list = [i for i, value in enumerate(ps) if value == 1] 

        order = [list(l) + one_list for l in list(permutations(non_one_list))]
        for l in order:
            dim.append([ps[i] for i in l])

        #for l in order:
        #    print l

        self.index_order = order
        self.dim_order = dim

        par2Dev = []
        dev2Par = []
        #Walk over all the orders one by one and create a mapping
        for mid in range(0, len(order)):
            par2Dev.append({})
            dev2Par.append({})
            x = 0
            y = 0
            wid = 0
            y0 = 0
            x0 = 0
            movingLeft =  False
            movingSouth = True
            scrambled = [-1,-1,-1,-1]
            #for each given order, itearte over the elements in the hypercube in the specified order
            for i in range(0, ps[order[mid][3]]):         
                for j in range(0, ps[order[mid][2]]):         
                    for k in range(0, ps[order[mid][1]]):        
                        m = 0
                        for m in range(0, ps[order[mid][0]]):
                            
                            scrambled[order[mid][3]]=i
                            scrambled[order[mid][2]]=j
                            scrambled[order[mid][1]]=k
                            scrambled[order[mid][0]]=m
                            sc_tuple = tuple(scrambled)
                            
                            par2Dev[mid][sc_tuple] = (wid, x, y)
                            
                            if (wid, x, y) not in dev2Par[mid]:
                                dev2Par[mid][(wid, x, y)] = sc_tuple
                            else:
                                print("mid: {} Assigning ({},{},{},{}) to ({},{},{}) which already has been assigned to".format(mid, sc_tuple[0], sc_tuple[1], sc_tuple[2], sc_tuple[3], wid, x, y))
                                #exit(0)

                            if mid==3 and x==7 and y==1:
                                print("mid: {} Assigned ({},{},{},{}) to ({},{},{})".format(mid, sc_tuple[0], sc_tuple[1], sc_tuple[2], sc_tuple[3], wid, x, y))
                            #    print("movingLeft: {}".format(movingLeft))
                            y = (y+1) if movingSouth else (y-1)
                            
                            inc_x = False
                            if y == self.wafer_dim and m < ps[order[mid][0]]-1:
                                movingSouth = False
                                y = self.wafer_dim - 1
                                x = x + 1
                                inc_x = True
                            elif y == -1:
                                movingSouth = True
                                y = 0
                                x = x + 1
                                inc_x = True
                            else:
                                inc_x = False
                        
                        y = y0
                        if (movingLeft and k < ps[order[mid][1]] and ps[order[mid][1]] > self.wafer_dim):
                            x = x - 1
                        else:
                            if not inc_x:
                                x = x + 1
                        if x == -1:
                            movingLeft = False
                            y0 = y0 + ps[order[mid][0]]
                            x0 = 0
                            x = x0
                            y = y0
                        elif x == self.wafer_dim and m < ps[order[mid][0]]-1:
                            movingLeft = True
                            y0 = y0 + ps[order[mid][0]]

                        #y = y0
                        if  (ps[order[mid][1]] <= self.wafer_dim):
                            if movingLeft:
                                if (x % ps[order[mid][1]] == 0):
                                    step = (ps[order[mid][0]]//self.wafer_dim) if (ps[order[mid][0]] > self.wafer_dim) else ps[order[mid][1]]
                                    x0 = x0 - step
                                    x = x0
                            else:
                                if (x % ps[order[mid][1]] == 0):
                                    step = (ps[order[mid][0]]//self.wafer_dim) if (ps[order[mid][0]] > self.wafer_dim) else ps[order[mid][1]]
                                    x0 = x0 + step
                                    x = x0

                        if (x > self.wafer_dim-1):
                            y0 = y0 + ps[order[mid][0]]
                            #x0 = (x0 - ps[order[mid][1]])
                            x0 = self.wafer_dim-1 if (x == self.wafer_dim and k < ps[order[mid][1]]) else (x0 - ps[order[mid][1]])
                            movingLeft = True
                            y = y0
                            x = x0
                        if x0 < 0:
                            movingLeft = False
                            x0 = 0
                            y0 = y0 + ps[order[mid][0]]
                            x = x0
                            y = y0
                        if y0 >= self.wafer_dim:
                            wid = wid + 1
                            x = 0
                            y = 0
                            y0 = 0
                            x0 = 0
                            movingSouth = True
                            movingLeft = False
                        #if mid == 6:
                        #    print(x)

        #if mid == 4:
        #    for wid in range(0, self.num_wafer):
        #        for x in range(0, self.wafer_dim):
        #            for y in range(0, self.wafer_dim):
        #                print("({},{},{}): {}".format(wid, x, y, dev2Par[0][(wid, x, y)]))
        #

        self.par2Dev = par2Dev
        self.dev2Par = dev2Par

    def all_connect(self, par2Dev):
        dp = self.dp
        lp = self.lp
        kp1 = self.kp1
        kp2 = self.kp2
    
        for i in range(0, lp):
          for j in range(0, dp):
            for k2 in range(0, kp2):
              for k1 in range(0, kp1 if kp1>2 else kp1-1):
                start_point = par2Dev[(k1, k2, j, i)]
                end_point = par2Dev[((k1+1) % kp1, k2, j, i)]
                self.route(start_point, end_point)
                self.updateConnectionType(start_point, end_point, "kp1")
       
        for i in range(0, lp):
          for j in range(0, dp):
            for k1 in range(0, kp1):
              for k2 in range(0, kp2 if kp2>2 else kp2-1):
                start_point = par2Dev[(k1, k2, j, i)]
                end_point = par2Dev[(k1, (k2+1) % kp2, j, i)]
                self.route(start_point, end_point)
                self.updateConnectionType(start_point, end_point, "kp2")
       
        for k1 in range(0, kp1):
          for j in range(0, dp):
            for k2 in range(0, kp2):
              for i in range(0, lp-1):
                start_point = par2Dev[(k1, k2, j, i)]
                end_point = par2Dev[(k1, k2, j, i+1)]
                self.route(start_point, end_point)
                self.updateConnectionType(start_point, end_point, "lp")
       
        for i in range(0, lp):
          for k1 in range(0, kp1):
            for k2 in range(0, kp2):
              for j in range(0, dp if dp>2 else dp-1):
                start_point = par2Dev[(k1, k2, j, i)]
                end_point = par2Dev[(k1, k2, (j+1)%dp, i)]
                self.route(start_point, end_point)
                self.updateConnectionType(start_point, end_point, "dp")
    
    def pidx(self, x, y):
        return y * self.wafer_dim + x

    def pidy(self, x, y):
        return x * self.wafer_dim + y

    def updateConnectionType(self, start, end, pid):
        wid_start = start[0]
        wid_end = end[0]

        if wid_start != wid_end:
            self.par2cross[pid] |= True


    def route(self, start, end):
        x_edges = self.x_edges
        y_edges = self.y_edges
        cross_edges = self.cross_edges

        wid_start = start[0]
        x_start = start[1]
        y_start = start[2]
    
        wid_end = end[0]
        x_end = end[1]
        y_end = end[2]

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
        #    for eid in range(0, dim * dim):
        #        print("w{} e{}: {}".format(wid, eid, x_edges[wid][eid]))

        for wid in range(0, self.num_wafer):
            for eid in range(0, dim * dim):
                max_x = max(max_x, x_edges[wid][eid])
                max_y = max(max_y, y_edges[wid][eid])
            for eid in range(0, dim):
                max_cross = max(max_cross, cross_edges[wid][eid])

        return max_cross, max_x, max_y

    def get_derate_factors(self):
        par = ["kp1","kp2","dp","lp"]
        derate_factor_inter = []
        derate_factor_intra = []
        par2cross_list = []

        ps = self.ps
        for layout_id in range(0, len(self.par2Dev)):
            print(layout_id)
            order = self.index_order[layout_id]
            print("Parallelism order: {}({})-----{}({})-----{}({})-----{}({})".format(par[order[0]], ps[order[0]], par[order[1]], ps[order[1]], par[order[2]], ps[order[2]],par[order[3]], ps[order[3]]))
            self.par2cross = {"kp1": False, 
                              "kp2": False, 
                              "dp": False,
                              "lp": False} 
            #[wafer-2-wafer, x_edge, x_edge]
            for wid in range(0, self.num_wafer):
                for eid in range(0, self.wafer_dim * self.wafer_dim):
                    self.x_edges[wid][eid] = 0
                    self.y_edges[wid][eid] = 0
                for eid in range(0, self.wafer_dim):
                    self.cross_edges[wid][eid] = 0

            self.all_connect(self.par2Dev[layout_id])
            par2cross_list.append(self.par2cross)
            #for key, val in self.par2cross.items():
            #    print("{}, {}".format(key, "inter" if val else "intra"))
            w_factor, x_factor, y_factor = self.checkMaxDegree()
            derate_factor_inter.append(w_factor)
            derate_factor_intra.append(max(x_factor, y_factor))
            #print("w_factor: {}, x_factor: {}, y_factor: {}".format(w_factor, x_factor, y_factor))
            #print

        return derate_factor_inter, derate_factor_intra, par2cross_list

   
def main():
    for kp1 in [1]: #[1, 16, 32]:
        for kp2 in [16]: #[1, 16, 32]:
            for dp in [2]: #[1, 2, 4, 8]:
                for lp in [2]:
                    print("==========")
                    print("({},{},{},{})".format(kp1, kp2, dp, lp))
                    print("==========")
                    nw = int(math.ceil(dp * kp1 * kp2 * lp / 64.0))
                    p = Projection(dp = dp, kp1 = kp1, kp2 = kp2, lp = lp, wafer_dim = 8, num_wafer = nw)
                    p.project()
                    derate_factor_inter, derate_factor_intra, par2cross = p.get_derate_factors()
                    for i, (x,y,z) in enumerate(zip(derate_factor_intra, derate_factor_inter, par2cross)):
                        print("layout: {}, derate_factor_intra: {}, derate_factor_inter: {}, cross_wafer: {}".format(i,x,y,z))
                        print
if __name__ == "__main__":
    main()
