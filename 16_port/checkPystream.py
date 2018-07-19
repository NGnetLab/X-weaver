from var_dump import var_dump
from util import *
from pystream import *
max_set_num = 104
demand_num = 12000
demand_length = 8
topo_length = 20
topo_path = "./optimalTopo-port4/topo_fattree/topo"
demand_path = "./4pattern/demand_sample_yang/demand"
output_path = "./4pattern/topo_sample_yang/topo"
topo_list = []
for i in range(max_set_num) :
    topo_path1 = topo_path + str(i) + ".txt"
    topo = readfile(topo_path1)
    topo_list.append(topo)
demand_list = []
for i in range(demand_num):
    # demand_path1 = demand_path + str(i) + ".txt"
    # demand = readfile(demand_path1)
    _,demand = demand_generate(0,demand_length)
    demand_list.append(demand)
demand_normal = demandAddZero(demand_length, topo_length, demand_list[0])
py = pystream(timeLength=int(1e6), demand=demand_normal, topo=topo[0], TcpFlag=0)
min_list = []
output_list = []
for i in range (demand_num):
    demand_normal = demandAddZero(demand_length, topo_length, demand_list[i])
    min = 1e6
    topo_min = topo_list[0]
    for j in range(max_set_num):
        py.Reset(topo_list[j],demand_normal)
        score, _, __ = py.streamMain()
        if score < min :
            min = score
            topo_min = topo_list[j]
        print(score)
    min_list.append(min)
    output_list.append(topo_min)
    print(i,"---",min)
for i in range(demand_num) :
    path_topo = output_path + str(i) +".txt"
    writefile(output_list[i], path_topo)
    path_demand = demand_path + str(i) +".txt"
    writefile(demand_list[i],path_demand)
