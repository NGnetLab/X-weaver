import numpy as np
import random
import tensorflow as tf
import copy
from var_dump import var_dump
# data mining,cache,web search,hadoop
sc = [1, 10, 2, 2]
cdfx = [[0, 0.5, 0.8, 1], [0, 0.35, 0.6, 0.804, 0.99, 1], [0, 0.203, 0.53, 1], [0, 0.104, 0.94, 1]]
cdfy = [[0, 0.05, 1.02, 6], [0, 0.18, 1.98, 3.3, 3.6, 4.71], [0, 1.296, 1.906, 5.7], [0, 1.554, 2.872, 5.7]]
multi_factor = 5e2


def demand_generate( distribution_id, num_ToR):
    standard_demand = np.zeros([num_ToR, num_ToR])
    normalized_demand = np.zeros([num_ToR, num_ToR])
    for i in range(num_ToR):
        for j in range(i+1,num_ToR):
            if random.uniform(0,1) < 0.8:
                 continue
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
                    normalized_demand[i][j] = standard_demand[i][j] / multi_factor
    t= np.where(normalized_demand > 50)
    return standard_demand, normalized_demand
# write file to filePath
def writefile(data, file_path):
    with open(file_path, "w") as f:
        for i in range(len(data)):
            for j in range(len(data[i])):
                f.write(str(data[i][j])+" ")
            f.write("\n")
        # print("successfully write data to file")

def readfile(file_path):
    file_object = open(file_path)
    try:
        all_the_text = file_object.read()
    finally:
        file_object.close()
    line = all_the_text.split("\n")
    element = []
    for num in range(len(line) - 1):
        temp = line[num].split()
        tempInt = []
        for index in range(len(temp)):
            tempInt.append(float(temp[index]))
            # tempInt.append(int(temp[index]))
        element.append(tempInt)
    return element
def readfile1(file_path,length):
    file_object = open(file_path)
    try:
        all_the_text = file_object.read()
    finally:
        file_object.close()
    line = all_the_text.split("\n")
    element = []
    for num in range(length):
        temp = line[num].split()
        tempInt = []
        for index in range(length):
            tempInt.append(float(temp[index]))
            # tempInt.append(int(temp[index]))
        element.append(tempInt)
    return element
def handleStr(str) :
    line = str.split("\n")
    element = []
    for num in range(len(line) - 1):
        temp = line[num].split()
        tempInt = []
        for index in range(len(temp)):
            tempInt.append(float(temp[index]))
            # tempInt.append(int(temp[index]))
        element.append(tempInt)
    return element
def readJson(file_path) :
    file_object = open(file_path, 'r')
    try:
        all_the_text = file_object.read()
    finally:
        file_object.close()
    return all_the_text
def demandAddZero(demandlength,topolength,demand) :
    result = np.zeros([topolength,topolength])
    for i in range(topolength) :
        for j in range(topolength) :
            if i < demandlength and j < demandlength:
                result[i][j] = demand[i][j]
            else :
                result[i][j] = 0
    return result
def topoReduceZero(demandlength,topolength,topo) :
    result = np.zeros([demandlength, demandlength])
    for i in range(demandlength):
        for j in range(demandlength):
                result[i][j] = topo[i][j]
    return result
def handle(demand,factor) :
    demand_copy = copy.deepcopy(demand)
    for i in range(len(demand)):
        for j in range(len(demand)):
            demand_copy[i][j] = demand[i][j]/factor
    return demand_copy
# fatTree topo init
def fatTreeInit(K):  # K is only even
    torNum = int(K * K / 2)
    switchNum = int(K * K * 5 / 4)
    coreNum = int(K * K / 4)
    adj = np.zeros([switchNum, switchNum],dtype=np.int)
    for i in range(K):
        for j in range(int(K / 2)):
            for k in range(int(K / 2)):
                src = int(i * K / 2 + j)
                dest = int(torNum + i * K / 2 + k)
                adj[src][dest] = 1
                adj[dest][src] = 1
    for i in range(coreNum):
        for j in range(K):
            src = int(i + torNum * 2)
            dest = int(torNum + i / (K / 2) + j * K / 2)
            adj[src][dest] = 1
            adj[dest][src] = 1
    return adj
def handleTopo(topo) :
    result = []
    for i in range(len(topo)) :
        for j in range(len(topo)) :
            if i < j :
                result.append(topo[i][j])
    return result
#weight
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial,name = "W")
#bias
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial,name = "bias")

def conv2d(x, W):
#strides [1,x_movement,y_movement,1]
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
def tranResult(topo) :
    output = np.zeros([len(topo),len(topo[0])])
    output_diff = np.zeros([len(topo),len(topo[0])])
    for j in range(len(topo)):
        for i in range(len(topo[0])) :
            if topo[j][i] > 0.5 :
                output[j][i] = 1
                output_diff[j][i] = 1
            else :
                output[j][i] = 0
                output_diff[j][i] = -1
    return output,output_diff
def rightEdges(topo,output,output_diff) :
    diff = np.zeros([len(topo)])
    link = np.zeros([len(topo),len(topo[0])])
    rightLink = np.zeros([len(topo)])
    # linkSum = np.zeros(len(topo))
    for i in range(len(topo)):
        diff[i] = sum(abs(topo[i] - output[i]))
        link[i] = topo[i] - output_diff[i]
        rightLink[i] = np.sum(link[i] == 0)
        # linkSum[i] = np.sum(output[i] == 1)
    return diff,rightLink
def buildDepMap(num):
    count = 0
    demap = np.zeros([int(num * (num-1)/2),2],dtype=int)
    for i in range (num-1):
        for j in range(i+1,num):
            demap[count][0] = i
            demap[count][1] = j
            count +=1
    return demap
def buildA(num,demap):
    A = np.zeros([num,int(num * (num-1)/2)])
    for ii in range(2):
        for jj in range(int(num * (num-1)/2)):
            pos = demap[jj][ii]
            A[pos][jj] = 1
    return A

def changeEdge(topo,width):
    coy = copy.deepcopy(topo)
    choose = random.randint(0,width -1)
    y= topo[choose].index(1.0)
    chooseFlag = True
    while(chooseFlag) :
        new = random.randint(0,width - 1)
        if new != y and new != choose:
            chooseFlag = False
    x = topo[new].index(1.0)
    coy[choose][y] = 0
    coy[y][choose] = 0
    coy[new][x] = 0
    coy[x][new] = 0

    coy[choose][x] = 1.0
    coy[x][choose] = 1.0
    coy[new][y] = 1.0
    coy[y][new] = 1.0
    return coy
def indexAll(data,value) :
    out = []
    for i in range(len(data)):
        if data[i] ==  value :
            out.append(i)
    return out

def weight_matching(demand, size):
    topo = np.zeros([size, size])
    # topo_ref = np.tril(size,k=-1)
    # topo_ref += topo_ref.transpose()
    topo_ref = np.zeros([size, size])
    demand = np.array(demand)
    demand += 1e-8
    demand_transpose = demand.transpose()
    demand = (demand + demand_transpose)/2
    demand = np.triu(demand, k=1)
    demand_index = np.where(demand > 0)
    demand_len = len(demand_index[0])
    demand_struct = np.zeros([demand_len, 3])
    cnt = 0
    for i in range(demand_len):
        demand_struct[cnt, 0] = demand[demand_index[0][i]][demand_index[1][i]]
        demand_struct[cnt, 1] = demand_index[0][i]
        demand_struct[cnt, 2] = demand_index[1][i]
        cnt += 1

    demand_arg = np.argsort(-demand_struct[:, 0])
    demand_struct = demand_struct[demand_arg]

    while demand_struct.size != 0:
        index = 0
        id_l = int(demand_struct[index][1])
        id_r = int(demand_struct[index][2])
        # if topo_ref[id_l][id_r] == 0:
        #     topo_ref[id_l, :] = 1
        #     topo_ref[id_r, :] = 1
        #     topo_ref[:, id_l] = 1
        #     topo_ref[:, id_r] = 1
        demand_struct = np.delete(demand_struct, 0, 0)
        demand_index_temp = np.where(demand_struct[:, 1] == id_l)
        demand_struct = np.delete(demand_struct, demand_index_temp, 0)
        demand_index_temp = np.where(demand_struct[:, 1] == id_r)
        demand_struct = np.delete(demand_struct, demand_index_temp, 0)
        demand_index_temp = np.where(demand_struct[:, 2] == id_l)
        demand_struct = np.delete(demand_struct, demand_index_temp, 0)
        demand_index_temp = np.where(demand_struct[:, 2] == id_r)
        demand_struct = np.delete(demand_struct, demand_index_temp, 0)
        topo[id_l, id_r] = 1
        topo[id_r, id_l] = 1

    return topo