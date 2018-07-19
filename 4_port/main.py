'''
   # developed by Dan Yang
   # X-weaver SCNN Fully-connected GraphNN
   # time : 2018.1.24
'''
import tensorflow as tf
import numpy as np
import NN as nn
import util as util
import logging
import random
import time
import os
from var_dump import var_dump
#os.environ['CUDA_VISIBLE_DEVICES'] = '2'

np.random.seed(4)
random.seed(4)
# parameters of the training
Batch_size = 5000
L_rate = 1e-3
MAX_STEP = 5000000
# train_set and test_set settings
demand_length = 8
topo_length = 20
train_set_sum = 210000
train_sum = 210000
test_sum = 21000
label_sum = test_sum + train_sum

# data settings
train_path = '../../sc_data/FCT_RANDOM/train/'
test_path = '../../sc_data/FCT_RANDOM/test/'
maxm = 31
# nn logs settings
logs_train_dir = "./logs_mean_cnn_final/train/"
logs_test_dir  = "./logs_mean_cnn_final/test/"

# log_record setting
LOG_FILE = './logs_mean_cnn_final/result/'
log_file = open(LOG_FILE + "_record" , "a")
c = time.time()
demand_test, topo_test, label_test = util.get_test_set(test_path,demand_length, topo_length)
with tf.Session() as sess:
    # input
    demand_tf = tf.placeholder(tf.float32, shape=[None, demand_length, demand_length], name = "demand_in")
    #demand_tf = tf.placeholder(tf.float32, shape=[None, topo_length, topo_length], name = "demand_in")
    topo_tf   = tf.placeholder(tf.float32, shape=[None, topo_length,topo_length], name = "topo_in")
    label_tf  = tf.placeholder(tf.float32, shape=[None, 1],name = "label_in")

    #y_out, keep_prob1,keep_prob2, keep_prob_d, keep_prob_t = nn.SCNN(demand_tf, topo_tf)
    y_out, keep_prob1,keep_prob2 = nn.SCNN(demand_tf, topo_tf)
    #y_out, keep_prob1 = nn.SFNN(demand_tf, topo_tf)
    loss = nn.loss_rate(label_tf, y_out)
    train_step = nn.train(loss,L_rate)
    acc = nn.accurate(label_tf, y_out, 0.05, Batch_size)
    acc_test0 = nn.accurate(label_tf, y_out, 0.05, 10500)
    acc_test1 = nn.accurate(label_tf, y_out, 0.1, 10500)
    acc_test2 = nn.accurate(label_tf, y_out, 0.2, 10500)
    acc_test3 = nn.accurate(label_tf, y_out, 0.3, 10500)

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    checkpoint = tf.train.get_checkpoint_state(logs_train_dir)
    #checkpoint.model_checkpoint_path = "./logs/train/model.ckpt-13000"
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Load train data.")

    summary_op = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    val_writer = tf.summary.FileWriter(logs_test_dir, sess.graph)

    cnt = 0 
    cnt_id = 0 
    index_list_permutaition = np.random.permutation(np.arange(0,train_set_sum))
    index_data = index_list_permutaition[ cnt_id  * train_sum : ( cnt_id + 1 ) * train_sum ]
    demand_train, topo_train, label_train = util.get_demand_and_topo(train_path, demand_length, topo_length, index_data)
    epoch = 243
    try:
        for step in np.arange(MAX_STEP):
            a = time.time()
            if coord.should_stop():
                break
            if step % 42 == 0 and step != 0:
                epoch += 1
                #ss1 = time.time()
                index_list_permutaition = np.random.permutation(np.arange(0,train_set_sum))
                #index_data = index_list_permutaition[ cnt_id  * train_sum : ( cnt_id + 1 ) * train_sum ]
                #demand_train, topo_train, label_train = util.get_demand_and_topo(train_path, demand_length, topo_length, index_data)
                time_now = time.time()
                log_file.write( str(epoch) + '\t' + 
                                str(tra_loss) + '\t' +
                                str(tra_acc) + '\t' +
                                str(val_loss_average) + '\t' +
                                str(time_now) + '\t' + 
                                str(val_acc0_average) + '\t' +
                                str(val_acc1_average) + '\t' +
                                str(val_acc2_average) + '\t' +
                                str(val_acc3_average) + '\t' +
                                "\n"
                )
                #print("chongzu",time_now - ss1)
            #if step % 1  == 0 :
            #    cnt_id = cnt % 42
                #print("---------------------exchange data -------id------------------", cnt_id)
            #    index_data = index_list_permutaition[ cnt_id  * train_sum : ( cnt_id + 1 ) * train_sum ]
            #    demand_train, topo_train, label_train = util.get_demand_and_topo(train_path, demand_length, topo_length, index_data)
            #    cnt +=1
           
            ids = step %  42
            index_list = index_list_permutaition[ Batch_size * ids : Batch_size * (ids + 1)]
            demand_dict  = np.array(demand_train)[index_list]
            #iprint(demand_dict)
            topo_dict    = np.array(topo_train)[index_list]
            label_dict   = np.array(label_train)[index_list]
            #demand_dict  = np.array(demand_train)
            #topo_dict    = np.array(topo_train)
            #label_dict   = np.array(label_train)
            #ss2 = time.time()
            _, tra_loss, y_train = sess.run([train_step, loss, y_out],
                                   #feed_dict={demand_tf: demand_dict, topo_tf: topo_dict, label_tf: label_dict,keep_prob1: 0.2, keep_prob2: 0.2 , keep_prob_d: 1.0, keep_prob_t: 1.0})
                                   feed_dict={demand_tf: demand_dict, topo_tf: topo_dict, label_tf: label_dict,keep_prob1: 1.0, keep_prob2: 1.0})
            #ss3 = time.time()
            #print("yunxing",ss3 - ss2)
            if step % 20 == 0:
                tra_acc = sess.run(acc,
                                feed_dict={demand_tf: demand_dict, topo_tf: topo_dict, label_tf: label_dict, keep_prob1: 1.0, keep_prob2: 1.0})
                print('Step %d, train loss = %.6f ,  accuracy = %.4f ' % (step, tra_loss, tra_acc))
                #print("  label",y_train,"value ",label_dict)
                # summary_str = sess.run(summary_op)
                # summary_str = sess.run(summary_op, feed_dict={demand_tf: demand_dict, topo_tf: topo_dict, label_tf: label_dict, keep_prob: 0.5})
                # train_writer.add_summary(summary_str, step)

            if step % 42 == 0 or (step + 1) == MAX_STEP:

                index_list_test = random.sample(range(test_sum), Batch_size)

                #demand_test_dict = np.array(demand_test)[index_list_test]
                #topo_test_dict = np.array(topo_test)[index_list_test]
                #label_test_dict = np.array(label_test)[index_list_test]
                demand_test_dict_all = np.array(demand_test)
                topo_test_dict_all = np.array(topo_test)
                label_test_dict_all = np.array(label_test)
                val_loss_sum = 0
                val_acc0_sum = 0
                val_acc1_sum = 0
                val_acc2_sum = 0
                val_acc3_sum = 0
                for j in range(2):
                     demand_test_dict = demand_test_dict_all[j * 10500: (j+1) * 10500]
                     topo_test_dict = topo_test_dict_all[j * 10500: (j+1) * 10500]
                     label_test_dict = label_test_dict_all[j * 10500: (j+1) * 10500]
                     val_loss,y_result, val_acc0, val_acc1, val_acc2,  val_acc3 = sess.run([loss, y_out, acc_test0, acc_test1,  acc_test2,  acc_test3],
                                             feed_dict={demand_tf: demand_test_dict, topo_tf: topo_test_dict, label_tf: label_test_dict,keep_prob1: 1.0, keep_prob2: 1.0 })
                     val_loss_sum += val_loss
                     val_acc0_sum += val_acc0
                     val_acc1_sum += val_acc1
                     val_acc2_sum += val_acc2
                     val_acc3_sum += val_acc3
                val_loss_average = val_loss_sum / 2
                val_acc0_average = val_acc0_sum / 2
                val_acc1_average = val_acc1_sum / 2
                val_acc2_average = val_acc2_sum / 2
                val_acc3_average = val_acc3_sum / 2
                print('********  Step %d, val loss = %.6f  , accuracy = %.4f ***********' % (step, val_loss_average, val_acc0_average))
                print("********  label",y_result[ int(step % 10500)],"***********",label_test_dict[ int(step % 10500)])
                # summary_str = sess.run(summary_op)
                # summary_str = sess.run(summary_op, feed_dict={demand_tf: demand_test_dict, topo_tf: topo_test_dict, keep_prob: 1.0})
                # val_writer.add_summary(summary_str, step)

            if step % 500 == 0 or (step + 1) == MAX_STEP:
               checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
               saver.save(sess, checkpoint_path, global_step=step)
            #if step > 10200 :
            #   exit()
            b = time.time()
            #print(b-a)
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
    coord.join(threads)
