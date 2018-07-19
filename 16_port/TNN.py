import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
import numpy as np
import pandas as pd
from util import *
from pystream import *
from var_dump import var_dump
import math

maxepisode = 10000000
DEMAND_ROW_NUM = 128
TOPO_ROW_NUM = 320
OCS_NUM = 1
L_rate = 1e-5
max_set_num = 1000
train_set_num = 900
batch_size = 50

demand_length = 128
topo_length = 320

test_length = 100
test_size = 5

# demand_path = "./4pattern/demand_sample_yang/demand"
# topo_path = "./4pattern/topo_sample_yang/topo"
# demand_test_path = "./4pattern/demand_sample_yang/demand"
# topo_test_path = "./4pattern/topo_sample_yang/topo"
demand_path = "../fpnn_data/demand/demand"
topo_path = "../fpnn_data/sample/topo/topo"
demand_test_path = "../fpnn_data/demand/demand"
topo_test_path = "../fpnn_data/topo/topo.txt"
ALPHA = 0
BETA = 0
changeFlag1 = False
changeFlag2 = False

kernel_size = 64
channel_num = 8

def build_net() :

    deepmap = buildDepMap(DEMAND_ROW_NUM)
    A = buildA(DEMAND_ROW_NUM,deepmap)
    A_tf = tf.constant(A,dtype=tf.float32)

    with tf.name_scope("inputs"):
        demand_tf = tf.placeholder("float", [None, DEMAND_ROW_NUM , DEMAND_ROW_NUM, 1], name="D_input")
        topo_tf = tf.placeholder("float", [None, int(DEMAND_ROW_NUM*(DEMAND_ROW_NUM - 1)/2)], name="T_input")
        B_tf = tf.placeholder("float",[None,DEMAND_ROW_NUM],name="constraint")
        Alpha_tf = tf.placeholder("float")
        Beta_tf = tf.placeholder("float")
        L_rate_tf = tf.placeholder("float")
    #conv1 demand
    with tf.name_scope("demand_conv1"):
        with tf.name_scope("weight"):
            Demand_w_conv1 = weight_variable([kernel_size, kernel_size, 1, channel_num])
        with tf.name_scope("bias"):
            Demand_b_conv1 = bias_variable([channel_num])
        D_conv1 = tf.nn.relu(conv2d(demand_tf, Demand_w_conv1) + Demand_b_conv1)
        D_pool1 = max_pool(D_conv1)

    # seconde Demand cnn
    with tf.name_scope("demand_conv2"):
        with tf.name_scope("weight"):
            Demand_w_conv2 = weight_variable([int(kernel_size/2), int(kernel_size/2), channel_num, channel_num * 2])
        with tf.name_scope("bias"):
            Demand_b_conv2 = bias_variable([channel_num * 2])
        D_conv2 = tf.nn.relu(conv2d(D_pool1, Demand_w_conv2) + Demand_b_conv2)
        D_pool2 = max_pool(D_conv2)
    # third Demand cnn
    with tf.name_scope("demand_conv3"):
        with tf.name_scope("weight"):
            Demand_w_conv3 = weight_variable([int(kernel_size/4), int(kernel_size/4), channel_num * 2, channel_num * 4])
        with tf.name_scope("bias"):
            Demand_b_conv3 = bias_variable([channel_num* 4])
        D_conv3 = tf.nn.relu(conv2d(D_pool2, Demand_w_conv3) + Demand_b_conv3)
        D_pool3 = max_pool(D_conv3)
        # third Demand cnn
    with tf.name_scope("demand_conv4"):
        with tf.name_scope("weight"):
            Demand_w_conv4 = weight_variable([int(kernel_size/8), int(kernel_size/8), channel_num * 4, channel_num * 8])
        with tf.name_scope("bias"):
            Demand_b_conv4 = bias_variable([channel_num * 8])
        D_conv4 = tf.nn.relu(conv2d(D_pool3, Demand_w_conv4) + Demand_b_conv4)
        D_pool4 = max_pool(D_conv4)
    #full connectivity
    D_conv_seq = tf.reshape(D_pool4, [-1, int(DEMAND_ROW_NUM/16) * int(DEMAND_ROW_NUM/16) * channel_num * 8])
    # D_conv_seq = tf.reshape(D_conv4, [-1, TOPO_ROW_NUM * TOPO_ROW_NUM * 256])
    # out_flat = tf.reshape(D_conv_seq, [-1, TOPO_ROW_NUM * TOPO_ROW_NUM * 256])
    with tf.name_scope("dense"):
        with tf.name_scope("weight"):
            W_fc1 = weight_variable([int(DEMAND_ROW_NUM/16)**2  * channel_num * 8,4096])
        with tf.name_scope("bias"):
            b_fc1 = bias_variable([4096])
        h_fc1 = tf.nn.relu(tf.matmul(D_conv_seq, W_fc1) + b_fc1)
    # out_flat1 = tf.reshape(h_fc1, [-1, TOPO_ROW_NUM * TOPO_ROW_NUM * 256])
    # with tf.name_scope("dense"):
    #     with tf.name_scope("weight"):
    #         W_fc2 = weight_variable([TOPO_ROW_NUM * TOPO_ROW_NUM * 256, 4096])
    #     with tf.name_scope("bias"):
    #         b_fc2 = bias_variable([4096])
    #     h_fc2 = tf.nn.relu(tf.matmul(out_flat1, W_fc2) + b_fc2)
    with tf.name_scope("output"):
        W_fc3 = weight_variable([4096, int(DEMAND_ROW_NUM*(DEMAND_ROW_NUM-1)/2)])
        b_fc3 = bias_variable([int(DEMAND_ROW_NUM*(DEMAND_ROW_NUM - 1)/2)])
    # output

        y_prediction = tf.matmul(h_fc1, W_fc3) + b_fc3
        y_reshape = tf.reshape(y_prediction,[-1,int(DEMAND_ROW_NUM * (DEMAND_ROW_NUM - 1)/2),1])
        # return demand_tf,topo_tf,y_conv
        y_square_tf = tf.multiply(y_prediction,y_prediction)
        y_constraint = tf.einsum('lk,ikj->ilj',A_tf,y_reshape)
        y_constraint_tf = tf.reshape(y_constraint,[-1,DEMAND_ROW_NUM])
    with tf.name_scope("loss"):
        first_loss = tf.losses.mean_squared_error(topo_tf,y_prediction)
        second_loss = Alpha_tf * tf.losses.mean_squared_error(y_constraint_tf, B_tf)
        third_loss = Beta_tf * tf.losses.mean_squared_error(y_prediction, y_square_tf)
        loss = first_loss + second_loss + third_loss
    with tf.name_scope("train"):
        tt = tf.train.AdamOptimizer(L_rate_tf)
        train_step = tt.minimize(loss)
    return loss, first_loss,second_loss,third_loss,train_step, demand_tf, topo_tf, B_tf,Beta_tf,Alpha_tf,L_rate_tf,y_prediction
if __name__ == '__main__':
    # print("ss")
    step = 0
    loss,loss1,loss2,loss3,train_step,demand_tf,topo_tf,B_tf,Beta_tf,Alpha_tf,L_rate_tf,y_prediction = build_net()

    demand_list = []
    topo_list = []
    demand_test_list = []
    topo_test_list = []
    for i in range(max_set_num) :
        demand_path1 = demand_path + str(i) + ".txt"
        topo_path1 = topo_path + str(i) + ".txt"
        # topo_path1 = topo_path
        demand = readfile(demand_path1)
        topo = readfile(topo_path1)
        # demand_normal = demandAddZero(demand_length, topo_length, demand)
        topo2 = topoReduceZero(demand_length,topo_length,topo)
        topoNormal = handleTopo(topo2)
        if i < train_set_num :
            demand_list.append(demand)
            topo_list.append(topoNormal)
        else :
            demand_test_list.append(demand)
            topo_test_list.append(topoNormal)
    # print(topo_test_list)
    # print(len(topoNormal))
    # demand_json = readJson(train_path)
    # demand_pd = pd.read_json(demand_json, orient='split')
    # test_json = readJson(test_path)
    # test_pd = pd.read_json(test_json, orient='split')
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # save_path = saver.save(sess, "save_net.ckpt")
        # saver.restore(sess, "my_net/save_net.ckpt")
        # load_path = saver.restore(sess, "./Tnn_log_model/")

        index_list = np.arange(train_set_num).reshape(train_set_num)
        for episode in range(maxepisode) :

            # demand_test_list = []
            # topo_test_list = []
            # score_test_list = []

            index_list = np.random.permutation(index_list)
            # print(index_list)
            for i in range(int(train_set_num/batch_size)):
                if step % 200 == 0:
                    save_path = saver.save(sess, "./Tnn_log_model1 /")
                # get demand and topo
                # print("-----------------------------------------------------------------------------",i)
                demand_temp = []
                topo_temp = []
                for j in range(i*batch_size, i*batch_size+batch_size):
                    demand_temp.append(demand_list[index_list[j]])
                    topo_temp.append(topo_list[index_list[j]])
                    # demand_temp = demand_list[i * batch_size: (i + 1) * batch_size]
                    # topo_temp = topo_list[i * batch_size: (i + 1) * batch_size]
                demand_dict = np.reshape(np.array(demand_temp),(-1,DEMAND_ROW_NUM,DEMAND_ROW_NUM,1))
                topo_dict = np.reshape(np.array(topo_temp),(-1,int(DEMAND_ROW_NUM * (DEMAND_ROW_NUM - 1 ) / 2)))
                bias_tf = np.ones([batch_size,DEMAND_ROW_NUM])

                train, training_cost,cost1,cost2,cost3, y_result = sess.run([train_step, loss, loss1,loss2,loss3,y_prediction],
                                                          feed_dict={demand_tf: demand_dict, topo_tf: topo_dict,B_tf:bias_tf,
                                                                     Alpha_tf:ALPHA,Beta_tf: BETA, L_rate_tf:L_rate})
                # training_cost = sess.run(
                #     loss, feed_dict={demand_tf: demand_normal_dict,topo_tf : topo_dict, y_score: score_dict})
                #
                if step % 100 == 0:
                    out, outdiff = tranResult(y_result)
                    diff, rightLink = rightEdges(topo_dict,out,outdiff)
                    pro = []
                    linkSum = []
                    for tt in range(len(rightLink)):
                        pro.append(rightLink[tt] / sum(topo_dict[0]))
                        linkSum.append(np.sum(out[tt] == 1))
                    pro_average = sum(pro)/len(pro)
                    linkSum_average = sum(linkSum)/len(linkSum)
                    print("loss", training_cost,
                          "   loss1", cost1,
                          "   loss2", cost2,
                          "   loss3", cost3,
                          "\nlabel",topo_dict[0],
                          "\nprediction", y_result[0],
                          "rigthLink_prob", pro_average,
                          "linkNum",linkSum_average)
                    # if cost3  < 1 and changeFlag1 and not changeFlag2:
                    #     ALPHA = 1
                    #     changeFlag2 = True
                    # if cost3 < 1 and not changeFlag1 :
                    #     BETA = 1
                    #     changeFlag1 = True
                    if training_cost < 0.001 and not changeFlag1:
                        L_rate = 5e-5
                        changeFlag1 = True

                print("loss",   training_cost,
                      "  loss1",cost1,
                      "  loss2",cost2,
                      "  loss3",cost3,
                      "step",step)
                step += 1
            count = episode % int(test_length/test_size)
            demand_test_dict = demand_test_list[count * test_size: (count + 1) * test_size]
            topo_test_dict = topo_test_list[count * test_size: (count + 1) * test_size]
            demand_dict1 = np.reshape(np.array(demand_test_dict), (-1, DEMAND_ROW_NUM, DEMAND_ROW_NUM, 1))
            topo_dict1 = np.reshape(np.array(topo_test_dict), (-1, int(DEMAND_ROW_NUM * (DEMAND_ROW_NUM - 1) / 2)))
            bias_tf1 = np.ones([test_size,DEMAND_ROW_NUM])
            y_result1 = sess.run(y_prediction, feed_dict={demand_tf: demand_dict1, topo_tf: topo_dict1,
                                                          B_tf:bias_tf1,Alpha_tf:ALPHA,Beta_tf: BETA,L_rate_tf:L_rate})
            out1,outdiff1 = tranResult(y_result1)
            diff1, rightLink1 = rightEdges(topo_dict1, out1,outdiff1)
            pro1= []
            linkSum1 = []
            for tt in range(len(rightLink1)):
                pro1.append(rightLink1[tt] / sum(topo_dict1[0]))
                linkSum1.append(np.sum(out1[tt] == 1))
            pro_average1  = sum(pro1)/len(pro1)
            linkSum_average1 = sum(linkSum1)/len(linkSum1)
            print("------------test  start---------------------",
                  "\nlabel", topo_dict1,
                  "\nprediction", out1,
                  "rightLink",pro_average1,
                  "linkNum",linkSum_average1)
            print("------------test  end------------------------")
