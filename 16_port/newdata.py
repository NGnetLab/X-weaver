from util import *
import random
from pystream import *
width = 100
max_width = 200
max_depth = 50
demand_length = 128
topo_length = 320
max_score = 999


topo_path   = "./topo.txt"
topo = readfile(topo_path)
count = 0
for j in range(2000) :
    out_demand_list = []
    out_topo_list = []
    out_socre_list = []
    _,demand = demand_generate(0, demand_length)
    demand_normal = demandAddZero(demand_length, topo_length, demand)
    if (j == 0) :
        py = pystream(timeLength=int(999), demand=demand_normal, topo=topo, TcpFlag=0)
    else:
        py.Reset(topo, demand_normal)
    aa, _, __ = py.streamMain()
    topo_cur = topo
    score_list = []
    score_list.append(aa)
    demand_T = demand.transpose()
    demand = demand + demand_T
    for i in range(50) :
        for hh in range(10) :
            topo_neib = changeEdge(topo_cur,demand_length)
            topo_cur = topo_neib
        py.Reset(topo_neib, demand_normal)
        score1, _, __ = py.streamMain()
        # py.Reset(topo_cur, demand_normal)
        # score2, _, __ = py.streamMain()
        print(score1, score_list[-1])
        # if score1 != score_list[-1] :
        out_demand_list.append(demand)
        out_topo_list.append(topo_neib)
        out_socre_list.append(score1)
        topo_cur = topo_neib
        score_list.append(score1)
    print(len(out_topo_list))
    for i in range(len(out_topo_list)) :
        outpath_topo = "../bigdata_yang/topo/topo" + str(count) + ".txt"
        outpath_demand = "../bigdata_yang/demand/demand" + str(count) + ".txt"
        outpath_score = "../bigdata_yang/score/score" + str(count) + ".txt"
        writefile(out_demand_list[i],outpath_demand)
        writefile(out_topo_list[i],outpath_topo)
        with open(outpath_score, "w") as f:
            f.write(str(out_socre_list[i]))
        count += 1
