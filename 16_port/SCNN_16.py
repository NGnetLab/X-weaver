import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
import numpy as np
import pandas as pd
from util import *
from pystream import *
from var_dump import var_dump
import math
import time

import pymysql
# 打开数据库连接
db = pymysql.connect("localhost","root","sayd199511","xweaver" )
# 使用 cursor() 方法创建一个游标对象 cursor
cursor = db.cursor()

demand_length = 128
topo_length = 320
maxepisode = 100000
DEMAND_ROW_NUM = 128
TOPO_ROW_NUM = 320
OCS_NUM = 1
L_rate = 1e-3
train_set_num = 17000
batch_size = 25
test_size = 10
test_length = 1200
max_set_num = 18200
# train_path = "../bigdata/"
# test_path = "../bigdata/"

demand_path = "../bigdata/demand/demand"
topo_path = "../bigdata/topo/topo"
score_path = "../bigdata/score/score"

demand_path0 = "../bigdata/demand/demand"
topo_path0 = "../bigdata/topo/topo"
score_path0 = "../bigdata/score/score"

multi_factor = 2e3

channel_num = 8
kernel_size = 64
def build_net() :
    y_score = tf.placeholder("float", [None, 1])
    # first Demand cnn
    with tf.name_scope("inputs"):
        demand_tf = tf.placeholder("float", [None, DEMAND_ROW_NUM * DEMAND_ROW_NUM],name= "D_input")
        topo_tf = tf.placeholder("float", [None, TOPO_ROW_NUM * TOPO_ROW_NUM],name= "T_input")
    with tf.name_scope("demand_conv1"):
        with tf.name_scope("weight"):
            Demand_w_conv1 = weight_variable([kernel_size, kernel_size, 1, channel_num])
        with tf.name_scope("bias"):
            Demand_b_conv1 = bias_variable([channel_num])
        D_image = tf.reshape(demand_tf, [-1, DEMAND_ROW_NUM, DEMAND_ROW_NUM, 1])
        D_conv1 = tf.nn.relu(conv2d(D_image, Demand_w_conv1) + Demand_b_conv1)
        D_pool1 = max_pool(D_conv1)
    # seconde Demand cnn
    with tf.name_scope("demand_conv2"):
        with tf.name_scope("weight"):
            Demand_w_conv2 = weight_variable([int(kernel_size/2), int(kernel_size/2), channel_num, channel_num * 2 ])
        with tf.name_scope("bias"):
            Demand_b_conv2 = bias_variable([channel_num * 2 ])
        D_conv2 = tf.nn.relu(conv2d(D_pool1, Demand_w_conv2) + Demand_b_conv2)
        D_pool2 = max_pool(D_conv2)

    # first Topo cnn
    with tf.name_scope("topo_conv1"):
        with tf.name_scope("weight"):
            Topo_w_conv1 = weight_variable([kernel_size, kernel_size, 1, channel_num])
        with tf.name_scope("bias"):
            Topo_b_conv1 = bias_variable([channel_num])
        T_image = tf.reshape(topo_tf, [-1, TOPO_ROW_NUM, TOPO_ROW_NUM, 1])
        T_conv1 = tf.nn.relu(conv2d(T_image, Topo_w_conv1) + Topo_b_conv1)
        T_pool1 = max_pool(T_conv1)
    # second Topo cnn
    with tf.name_scope("topo_conv2"):
        with tf.name_scope("weight"):
            Topo_w_conv2 = weight_variable([int(kernel_size/2), int(kernel_size/2), channel_num, channel_num * 2])
        with tf.name_scope("bias"):
            Topo_b_conv2 = bias_variable([channel_num * 2])
        T_conv2 = tf.nn.relu(conv2d(T_pool1, Topo_w_conv2) + Topo_b_conv2)
        T_pool2 = max_pool(T_conv2)
    # Topo + Demand
    D_conv2_seq = tf.reshape(D_pool2,[-1,int(DEMAND_ROW_NUM/4) * int(DEMAND_ROW_NUM/4),channel_num * 2])
    T_conv2_seq = tf.reshape(T_pool2,[-1,int(TOPO_ROW_NUM/4) * int(TOPO_ROW_NUM/4),channel_num * 2])
    input_data = tf.concat([D_conv2_seq, T_conv2_seq], 1)
    # full connectivity
    with tf.name_scope("dense"):
        with tf.name_scope("weight"):
            W_fc1 = weight_variable([(int(TOPO_ROW_NUM / 4) **2 + int(DEMAND_ROW_NUM/4)**2) * channel_num * 2, 1024])
        with tf.name_scope("bias"):
            b_fc1 = bias_variable([1024])
        out_flat = tf.reshape(input_data, [-1, (int(TOPO_ROW_NUM / 4) **2 + int(DEMAND_ROW_NUM/4)**2) * channel_num * 2])
        h_fc1 = tf.nn.relu(tf.matmul(out_flat, W_fc1) + b_fc1)

    # dropout
    # with tf.name_scope("dropout")
    #     keep_prob = tf.placeholder("float")
    #     h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    with tf.name_scope("output"):
        W_fc2 = weight_variable([1024, 1])
        b_fc2 = bias_variable([1])
    # output
        y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2
    # y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # return demand_tf,topo_tf,y_conv
    with tf.name_scope("loss"):
        loss = tf.losses.mean_squared_error(y_score,y_conv)

    with tf.name_scope("train"):
        tt = tf.train.AdamOptimizer(L_rate)
        train_step = tt.minimize(loss)

    # return loss,train_step,demand_tf,topo_tf,y_score,y_conv,keep_prob
    # sess = tf.Session()
    # writer = tf.summary.FileWriter("./logs/",sess.graph)
    return loss, train_step, demand_tf, topo_tf, y_score, y_conv
    # cross_entropy = -tf.reduce_sum(y_score * tf.log(y_conv))
    # train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    # correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_score, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # return accuracy,train_step

if __name__ == '__main__':
    # print("ss")
    step = 0
    loss,train_step,demand_tf,topo_tf,y_score,y_conv = build_net()
    # print(loss)
    # loss, train_step, demand_tf, topo_tf, y_score, y_conv,keep_prob = build_net()

    saver = tf.train.Saver()

    #train
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.7
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        load_path = saver.restore(sess, "./scnn_log_model/")
        # save_path = saver.save(sess, "save_net.ckpt")
        score = 0
        # saver.restore(sess, "my_net/save_net.ckpt")
        # print("weights:", sess.run(W))
        # print("biases:", sess.run(b))
        index_list = np.arange(1,train_set_num+1).reshape(train_set_num)

        demand_list = []
        topo_list = []
        score_list = []
        demand_test_list = []
        topo_test_list = []
        score_test_list = []


        # cursor.execute("SELECT * FROM scnn where id > 8000 limit " + str(test_length) )
        # temp_test = cursor.fetchall()

        for i in range(train_set_num,train_set_num + test_length):
            print(i)
            demand_test_path1 = demand_path + str(i) + ".txt"
            topo_test_path1 = topo_path + str(i) + ".txt"
            score_test_path1 = score_path + str(i) + ".txt"
            demand_ele = readfile(demand_test_path1)
            handle(demand_ele,multi_factor)
            topo_ele = readfile(topo_test_path1)
            input = open(score_test_path1, 'r')
            try:
                score = input.read()
            finally:
                input.close()
            score_ele = int(score)
            demand_test_list.append(demand_ele)
            topo_test_list.append(topo_ele)
            score_test_list.append(score_ele)
        print("test read over")
        basic = fatTreeInit(16)
        for episode in range(maxepisode) :

            index_list = np.random.permutation(index_list)
            # print(index_list)
            for i in range(int(train_set_num/batch_size)):
                if step % 200 == 0 :
                    save_path = saver.save(sess, "./scnn_log_model/")

                # get demand and topo
                # print("-----------------------------------------------------------------------------",i)
                demand_temp = []
                topo_temp = []
                score_temp = []
                # temp = demand_pd.loc[]
                id_list = index_list[i*batch_size: i*batch_size+batch_size]

                # for j in range(i*batch_size, i*batch_size+batch_size):
                a = time.time()
                for j in range(batch_size):
                    # demand_normal = demandAddZero(DEMAND_ROW_NUM, TOPO_ROW_NUM, demand_list[index_list[j]])
                    demand_path1 = demand_path + str(index_list[j]) + ".txt"
                    demand = readfile(demand_path1)
                    handle(demand, multi_factor)
                    demand_temp.append(demand)
                    topo_path1 = topo_path + str(index_list[j]) + ".txt"
                    topo = readfile1(topo_path1, demand_length)
                    topo_extend = np.lib.pad(topo, ((0, topo_length - demand_length), (0, topo_length - demand_length)),
                                             'constant', constant_values=(0, 0))
                    topo_temp.append(topo_extend + basic)
                    score_path1 = score_path + str(index_list[j]) + ".txt"
                    input = open(score_path1)
                    try:
                        score = input.read()
                    finally:
                        input.close()
                    score_temp.append(int(score))
                    # print(demand_normal)
                # print(topo_list)
                b = time.time()
                print("time",b - a)
                demand_normal_dict = np.reshape(np.array(demand_temp),(-1,DEMAND_ROW_NUM*DEMAND_ROW_NUM))
                topo_dict = np.reshape(np.array(topo_temp),(-1,TOPO_ROW_NUM*TOPO_ROW_NUM))
                score_dict = np.reshape(np.array(score_temp),(-1,1))

                # train, training_cost  = sess.run([train_step,loss],feed_dict={demand_tf: demand_normal_dict,topo_tf : topo_dict, y_score: score_dict,keep_prob: 0.5})
                c = time.time()
                train, training_cost, y_result = sess.run([train_step,loss,y_conv],
                         feed_dict={demand_tf: demand_normal_dict, topo_tf: topo_dict, y_score: score_dict})
                # training_cost = sess.run(
                #     loss, feed_dict={demand_tf: demand_normal_dict,topo_tf : topo_dict, y_score: score_dict})
                #
                # print("socre--->",score_tt, "loss--->",training_cost)
                print("loss",training_cost, "step",step)
                d = time.time()
                print("train_time",d-c)
                if step % 50 == 0 :
                    print("------------------------------------------------")
                    print("~~~~~~~~~~~~~label~~~~~~~~~~~\n",score_dict, "\n~~~~~~~~~ prediction~~~~~~~~\n",y_result,)
                    print("------------------------------------------------")
                step += 1

            count = episode % int(test_length/test_size)

            demand_test_temp = demand_test_list[count * test_size: (count + 1) * test_size]
            topo_test_temp = topo_test_list[count * test_size: (count + 1) * test_size]
            score_test_temp = score_test_list[count * test_size: (count + 1) * test_size]

            demand_test_dict = np.reshape(np.array(demand_test_temp),(-1,DEMAND_ROW_NUM*DEMAND_ROW_NUM))
            topo_test_dict = np.reshape(np.array(topo_test_temp),(-1,TOPO_ROW_NUM*TOPO_ROW_NUM))
            score_test_dict = np.reshape(np.array(score_test_temp),(-1,1))

            result = sess.run(y_conv,feed_dict = {demand_tf: demand_test_dict, topo_tf: topo_test_dict, y_score: score_test_dict})
            print("~~~~~~~~label test~~~~~\n",score_test_dict,"\n~~~~prediction test~~~~~\n",result)
