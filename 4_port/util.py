import tensorflow as tf
import numpy as np
import random
import time
from var_dump import var_dump
# some functions of neural networks
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial,name = "W")

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name = "biaes")

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def add_demand_zero(demand, demandlength, topolength):
    result = np.zeros([topolength,topolength])
    for i in range(topolength) :
        for j in range(topolength) :
            if i < demandlength and j < demandlength:
                result[i][j] = demand[i][j]
            else :
                result[i][j] = 0
    return result

# some function to read the data
def get_demand_and_topo(train_path, demand_length, topo_length, index_list):

    demand = []
    topo = []
    label = []
 
    each_file_choose = 5000
    each_file_num = 200000

    #index_list = random.sample(range(each_file_num), each_file_choose)

    demand_file_name = train_path + "train_demand.csv"
    topo_file_name = train_path + "train_topo.csv"
    label_file_name = train_path + "train_label.csv"

    

    demand_content = open(demand_file_name)
    topo_content = open(topo_file_name)
    label_content =  open(label_file_name)
   
    lines_d = demand_content.readlines()
    lines_t = topo_content.readlines()
    lines_l = label_content.readlines()
    for i in range( len(index_list)):
        demand_line_str = lines_d[index_list[i]].split(',')
        temp_d = list(map(float,demand_line_str))
        demand_matrix = np.reshape(temp_d, (demand_length, demand_length))
        #demand_matrix_x = np.reshape(temp_d, (demand_length, demand_length))
        #demand_matrix = add_demand_zero(demand_matrix_x, 8,20)
        demand.append(demand_matrix)

        topo_line_str = lines_t[index_list[i]].split(',')
        temp_t = list(map(float,topo_line_str))
        topo_matrix = np.reshape(temp_t, (topo_length, topo_length))
        topo.append(topo_matrix)


        label_line_str = lines_l[index_list[i]].split(',')
        temp_l = list(map(float,label_line_str))
        label_matrix = np.reshape(temp_l, 1)
        label.append(label_matrix)


    # print(np.shape(demand))
    # print(np.shape(topo))
    # print(np.shape(label))
    return demand, topo, label

    

# get the test_set data
def get_test_set(test_path, demand_length, topo_length):

    demand = []
    topo = []
    label = []

    demand_file_name = test_path + "test_demand.csv"
    topo_file_name = test_path + "test_topo.csv"
    label_file_name = test_path + "test_label.csv"

    demand_content = np.loadtxt(demand_file_name, dtype=np.float64, delimiter=",")
    topo_content = np.loadtxt(topo_file_name, dtype=np.float64, delimiter=",")
    label_content = np.loadtxt(label_file_name, dtype=np.float64, delimiter=",")

    for i in range(len(demand_content)):
        temp_d = demand_content[i]
        demand_matrix = np.reshape(temp_d, (demand_length, demand_length))
        #demand_matrix_x = np.reshape(temp_d, (demand_length, demand_length))
        #demand_matrix = add_demand_zero(demand_matrix_x, 8,20)
        demand.append(demand_matrix)

        temp_t = topo_content[i]
        topo_matrix = np.reshape(temp_t, (topo_length, topo_length))
        topo.append(topo_matrix)

        temp_l = label_content[i]
        label_matrix = np.reshape(temp_l, 1)
        label.append(label_matrix)


    # print(np.shape(demand))
    # print(np.shape(topo))
    # print(np.shape(label))
    return demand, topo, label
