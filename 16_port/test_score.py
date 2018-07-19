import random
from util import *
from pystream import *
import pandas as pd
import time
import matplotlib.pyplot as plt

demand_length = 128
topo_length = 320
max_score = 999
ll = 21000
sc = [1, 10, 2, 2]
cdfx = [[0, 0.5, 0.8, 1], [0, 0.35, 0.6, 0.804, 0.99, 1], [0, 0.203, 0.53, 1], [0, 0.104, 0.94, 1]]
cdfy = [[0, 0.05, 1.02, 6], [0, 0.18, 1.98, 3.3, 3.6, 4.71], [0, 1.296, 1.906, 5.7], [0, 1.554, 2.872, 5.7]]
multi_factor = 5e2

"""def demand_generate1( distribution_id, num_ToR):
    standard_demand = np.zeros([num_ToR, num_ToR])
    normalized_demand = np.zeros([num_ToR, num_ToR])
    for i in range(num_ToR):
        for j in range(i+1,num_ToR):
            if i == j :
                continue
            if distribution_id == "uniform":
                normalized_demand[i][j] = random.uniform(0, 1)
                standard_demand[i][j] = normalized_demand[i][j] * multi_factor
                # self.standard_demand[j][i] = self.standard_demand[i][j]
            else:
                r = random.random()
                l = 1
                while cdfx[distribution_id][l] < r:
                    l += 1
                    standard_demand[i][j] = sc[distribution_id] * pow(10, (
                    cdfy[distribution_id][l] - cdfy[distribution_id][l - 1]) /
                                                                      (cdfx[distribution_id][l] -
                                                                       cdfx[distribution_id][l - 1]) *
                                                                      (r - cdfx[distribution_id][l - 1]) +
                                                                      cdfy[distribution_id][l - 1])
                    normalized_demand[i][j] = int(standard_demand[i][j] / multi_factor)
    return standard_demand, normalized_demand
# topo_list = []
# topo_path = "./topo.txt"  #topo 320
# topo_path = "./optimalTopo-port4/topo_fattree/topo0.txt"
# topo_path = "../data/beam_search/6-port/topo0.txt"
# topo = readfile(topo_path)
# topo_list.append(topo)


demand_path = "../fpnn_data/demand/demand"
demand_list = []

random.seed(625)
for i in range(8684,ll) :
    print(i)
    _, demand = demand_generate1(0, demand_length)
    # demand_list.append(demand)
# _,demand = demand_generate(0, demand_length)
    output_path = demand_path + str(i) + ".txt"
    writefile(demand,output_path)"""

length = 1000
demand_path = "../fpnn_data/demand/demand"
topo_path = "../fpnn_data/weight_matching/topo/topo"
score_path = "../fpnn_data/weight_matching/score/score"
basic = fatTreeInit(16)
for i in range(20000,20000 + length):
    demand_path1 = demand_path + str(i) + ".txt"
    demand = readfile(demand_path1)
    topo1 = weight_matching(demand,demand_length)
    demand_normal = demandAddZero(demand_length,topo_length,demand)
    topo_extend = np.lib.pad(topo1, ((0, topo_length - demand_length), (0, topo_length - demand_length)),
                             'constant', constant_values=(0, 0))
    topo = topo_extend + basic
    if i == 20000 :
        py = pystream(timeLength=int(1e6), demand=demand_normal, topo=topo, TcpFlag=0)
    else:
        py.Reset(topo,demand_normal)
    score, _, __ = py.streamMain()
    writefile(topo,topo_path + str(i) + ".txt")
    input = open(score_path + str(i) + ".txt", 'w')
    input.write(str(score))

# fig = plt.figure()
# ax0 = fig.add_subplot(121)
# im0 = ax0.imshow(result)
# plt.colorbar(im0, fraction=0.046, pad=0.04)
# plt.show()
# writefile(demand,"./demand.txt")
