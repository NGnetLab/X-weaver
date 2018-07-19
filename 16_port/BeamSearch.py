from util import *
import random
from pystream import *
import time
width = 10
max_width = 10
max_depth = 10
demand_length = 10
topo_length = 45
max_score = 999
ll = 250


topo_list = []
# topo_path = "./topo.txt"  #topo 320
# topo_path = "./optimalTopo-port4/topo_fattree/topo0.txt"
topo_path = "./topo_6_10.txt"
topo = readfile(topo_path)
topo_list.append(topo)


topo_cur = topo_list[-1]
topo_best = topo_cur
# topo_path  = "./optimalTopo-port4/topo_fattree/topo1.txt"
# topo_path   = "./optimalTopo-port4/topo_fattree/topo"
# demand_path = "./4pattern/demand_sample/demand"
demand_path = "../bigdata/demand/demand"
# topo_path = "../bigdata/topo/topo"
# score_path = "../bigdata/topo/topo"
demand_list = []
# topo_list = []
# score_list = []
# basic = fatTreeInit(16)
for i in range(ll) :
    # demand_path1 = demand_path + str(i) + ".txt"
    # demand = readfile(demand_path1)
    # demand_list.append(demand)
    _, demand = demand_generate(0, demand_length)
    demand_list.append(demand)
# _,demand = demand_generate(0, demand_length)
demand_normal = demandAddZero(demand_length, topo_length, demand_list[0])
py = pystream(timeLength=int(1e6), demand=demand_normal, topo=topo, TcpFlag=0)
aa, _, __ = py.streamMain()
print(aa)


test = []
result = []
for num in range(ll) :
    count = 0
    Beam = []
    Beamput = []
    topo_list = []
    topo_list.append(topo)
    topo_cur = topo_list[-1]
    topo_best = topo_cur
    max_score = 999
    for i in range(width):
        Beam.append(max_score)
        Beamput.append(topo)
    print("-------",num)
    print(Beam)
    for s_depth in range(1, max_depth) :
        progress = False
        topo_cur = topo_best
        for s_width in range(1, max_width):
            # print(count)
            topo_neib = changeEdge(topo_cur,10)
            # topo_temp = np.zeros([8,8])
            # for jj in range(8) :
            #     for ii in range(8) :
            #         topo_temp[jj][ii] = topo_neib[jj][ii]
            # writefile(topo_temp,"./hah/sss" + str(count) +".txt")
            count += 1
            demand_normal = demandAddZero(demand_length, topo_length, demand_list[num])
            py.Reset(topo_neib,demand_normal)
            score1, _, __ = py.streamMain()

            py.Reset(topo_cur,demand_normal)
            score2, _, __ = py.streamMain()
            print(score1,score2)
            if score1 <= score2 :
                topo_cur = topo_neib
                progress = True
                max_list = indexAll(Beam,max(Beam))
                max_index = random.randint(0,len(max_list)-1)
                Beam[max_list[max_index]] = score1
                Beamput[max_list[max_index]] = topo_cur
            if max_score > score1 :
                max_score = score1
                topo_best = topo_cur

        # set_max
        if not progress :
            E = random.randint(0,100)
            # if E < Pro:
            #     topo_best = topo_cur
            #     index = Beam.index(min(Beam))
            #     min_list = indexAll(Beam,min(Beam))
            #     min_index = random.randint(0,len(min_list))
            #     Beam[min_list[min_index]] = max_score
            #     Beamput[min_list[min_index]] = topo_best
            # else :
            tt = random.randint(0,width-1)
            topo_cur = Beamput[tt]
    result.append(min(Beam))
for i in range(ll) :
    path_d ='./beam_data/10_6_10/data'
    path_d1 = path_d + str(i) + ".txt"
    file = open(path_d1,"w")
    file.write(str(result[i]))
    path_w='./beam_data/10_6_10/demand'
    path_w1 = path_w + str(i) + ".txt"
    writefile(demand_list[i],path_w1)
