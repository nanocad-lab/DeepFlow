import numpy as np
import argparse

def mmm_breakup(B, D, S, h, nheads, h_MLP1, h_MLP2):
    mmm =  {}
    dims = {}
    numlevels = 6
    levels = ["X.W=KQV", "Q.K=R", "R.V=Z", "Z.W=O", "O.WL1=O1", "O1.WL2=O2"]
    print("matrix dimensions accounting for all heads & batched dimension")
    dims[str(levels[0])]=[B*S, D, h*nheads]
    dims[str(levels[1])]=[B*S, h, S*nheads]
    dims[str(levels[2])]=[B*S, S, h*nheads]
    dims[str(levels[3])]=[B*S, D, D]
    dims[str(levels[4])]=[B*S, D, h_MLP1]
    dims[str(levels[5])]=[B*S, h_MLP1, h_MLP2]
    
    print("levels:",levels)
    print("writting the matrix dimensions ...")
    file = open("mat_dims.txt","w")
    #file.write('#'+str(levels)+'\n')
    for i in range(numlevels):
        mmm[str(levels[i])]=[]
        mmm[str(levels[i])].append(dims[str(levels[i])])
        tmp = str(mmm[str(levels[i])]).replace("[", "").replace("]","").replace(",", " ")
        print(tmp)
        file.write(tmp+'\n')

def main():
    parser = argparse.ArgumentParser(description='Generate Matrix dimensions')
                                     
    # Add the command line arguments
    parser.add_argument('B', type=int, help='batch size')
    parser.add_argument('D', type=int, help='dimensionality')
    parser.add_argument('S', type=int, help='sequence length')
    parser.add_argument('h', type=int, help='hid dim of attn sublayer')
    parser.add_argument('nheads', type=int, help='num of attn heads')
    parser.add_argument('h_MLP1', type=int, help='hid layer dim for 1st MLP layer')
    parser.add_argument('h_MLP2', type=int, help='hid layer dim for 2st MLP layer')

    # Parse the command line arguments
    args = parser.parse_args()

    # Extract the arguments
    B = args.B
    D = args.D
    S = args.S
    h = args.h
    nheads = args.nheads
    h_MLP1 = args.h_MLP1
    h_MLP2 = args.h_MLP2
    
    print(B, D, S, h, nheads, h_MLP1, h_MLP2)
    #assert(D == nheads*h) #"dimensionality is not equal to nheads x hidden dim")

    mmm_breakup(B, D, S, h, nheads, h_MLP1, h_MLP2)
     
if __name__ == '__main__':
    main()
