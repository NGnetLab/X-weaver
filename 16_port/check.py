import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import tensorflow as tf
import numpy as np
import pandas as pd
from util import *
from pystream import *
from var_dump import var_dump
import math

demand_length = 10
topo_length = 45
ll = 250

demand_path = "./beam_data/10_6_10/demand"
# topo_path = "./optimalTopo-port4/topo_fattree/topo"
topo_path = "../../topo_data/10_port/topo"
score_path = "./beam_data/10_6_10/data"
demand_list = []
score_list = []
score_list_dd = []
result = 0
out = []
topo_basic = fatTreeInit(6)
for i in range(ll):
    path_d = demand_path + str(i) + ".txt"
    demand = readfile(path_d)
    demand_list.append(demand)
    # path_w = score_path + str(i) + ".txt"
    # score = open(path_w,"r").read()
    # score_list .append(int(score))
    # score_list_dd.append(score)
for i in range(ll) :
    min_m = 999
    for j in range(945):
        print(j)
        # topo_path1 = topo_path + str(j) + ".txt"
        n = '%03d'%(j+1)
        topo_path1 = topo_path + n + ".txt"
        topo = readfile(topo_path1)
        topo_extend = np.lib.pad(topo, ((0, topo_length - demand_length), (0, topo_length - demand_length)),
                                 'constant', constant_values=(0, 0))
        topo_temp = topo_basic + topo_extend
        demand_normal = demandAddZero(demand_length, topo_length, demand_list[i])
        if (i == 0 and j == 0):
            py = pystream(timeLength=int(1e6), demand=demand_normal, topo=topo_temp, TcpFlag=0)
        else:
            py.Reset(topo_temp, demand_normal)
        aa, _, __ = py.streamMain()
        if aa < min_m :
            min_m = aa
    out.append(str(min_m))
    # print(score_list[i],min_m)
out_str = ",".join(out)
# score_str = ",".join(score_list_dd)
file = open("./beam_data/bianli_10_6_10.csv","w")
file.write(out_str)
# file = open("./beam_data/search_10_6_10.csv","w")
# file.write(score_str)
