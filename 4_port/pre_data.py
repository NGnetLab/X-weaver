import numpy as np
import random
import csv

def pre_data(demand_path, topo_path, label_path, demand_length, topo_length):

    demand = []
    topo = []
    label = []

    trace_num = 40
    each_file_choose = 500
    each_file_num = 105000

    for num in range(trace_num):
        index_list = random.sample(range(each_file_num), each_file_choose)
        print(index_list)
        file_d = open(demand_path + str(num+1) + '.csv')
        lines_d = file_d.readlines()

        file_t = open(topo_path + str(num+1) + '.csv')
        lines_t = file_t.readlines()

        file_l = open(label_path + str(num+1) + '.csv')
        lines_l = file_l.readlines()

        train_d = open("../../sc_data/train/demand/sc_demand_" + str(num+1) + ".csv", "w")
        train_t = open("../../sc_data/train/topo/sc_topo_" + str(num+1) + ".csv", "w")
        train_l = open("../../sc_data/train/label/sc_label_" + str(num+1) + ".csv", "w")

        for i in range(len(index_list)) :

            temp_d = lines_d[index_list[i]]
            demand.append(temp_d)

            temp_t = lines_t[index_list[i]]
            topo.append(temp_t)

            temp_l = lines_l[index_list[i]]
            label.append(temp_l)
        for l in range(len(lines_d)) :
            if l in index_list :
                continue
            else:
                train_d.write(lines_d[l])
                train_t.write(lines_t[l])
                train_l.write(lines_l[l])


    csv_d = open("../../sc_data/test/demand/demand_test.csv", "w")
    csv_t = open("../../sc_data/test/demand/topo_test.csv", "w")
    csv_l = open("../../sc_data/test/demand/label_test.csv", "w")

    for i in range(len(demand)):
        csv_d.write(demand[i])
        csv_t.write(topo[i])
        csv_l.write(label[i])
    # for i in range(len(demand)):
    # return demand,topo,label

demand_path = '../../sc_data/demand/sc_demand_'
topo_path = '../../sc_data/topo/sc_topo_'
label_path = '../../sc_data/label/sc_label_'

pre_data(demand_path,topo_path,label_path,8,20)