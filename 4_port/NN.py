import tensorflow as tf
import numpy as np
import util

def SCNN(demand,topo) :

    x_demand = tf.reshape(demand, [-1,8,8,1])
    x_topo   = tf.reshape(topo, [-1,20,20,1])

    # convolution 1
    with tf.name_scope('D_cnn_1_layer'):
      with tf.name_scope('Weight'):
        W_conv1_d = util.weight_variable([3, 3, 1, 16])
      with tf.name_scope('biases'):
        b_conv1_d = util.bias_variable([16])
    # h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
      d_conv1 = tf.nn.relu(util.conv2d(x_demand, W_conv1_d) + b_conv1_d)
      #d_conv1 = util.conv2d(x_demand, W_conv1_d) + b_conv1_d
    # h_pool1 = max_pool_2x2(h_conv1)

    # convolution 2
    with tf.name_scope('D_cnn_2_layer'):
      with tf.name_scope('Weight'):
        W_conv2_d = util.weight_variable([3, 3, 16, 32])
      with tf.name_scope('biases'):
        b_conv2_d = util.bias_variable([32])
    # h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
      d_conv2 = tf.nn.relu(util.conv2d(d_conv1, W_conv2_d) + b_conv2_d)
      #d_conv2 = util.conv2d(d_conv1, W_conv2_d) + b_conv2_d
      #d_conv2_flat = tf.reshape(d_conv2, [-1, 8 * 8 * 32])
    # h_pool2 = max_pool_2x2(h_conv2)

    # dropout
    #with tf.name_scope('dropout'):
     #  keep_prob_d = tf.placeholder(tf.float32, name = "drop_in")
      # d_drop = tf.nn.dropout(d_conv2, keep_prob_d)

    # convolution 1
    with tf.name_scope('T_cnn_1_layer'):
      with tf.name_scope('Weight'):
        W_conv1_t = util.weight_variable([5, 5, 1, 16])
      with tf.name_scope('biases'):
        b_conv1_t = util.bias_variable([16])
    # h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
      t_conv1 = tf.nn.relu(util.conv2d(x_topo, W_conv1_t) + b_conv1_t)
      #t_conv1 = util.conv2d(x_topo, W_conv1_t) + b_conv1_t
    # h_pool1 = max_pool_2x2(h_conv1)

    # convolution 2
    with tf.name_scope('T_cnn_2_layer'):
      with tf.name_scope('Weight'):
        W_conv2_t = util.weight_variable([5, 5, 16, 32])
      with tf.name_scope('biases'):
        b_conv2_t = util.bias_variable([32])
    # h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
      t_conv2 = tf.nn.relu(util.conv2d(t_conv1, W_conv2_t) + b_conv2_t)
      #t_conv2 = util.conv2d(t_conv1, W_conv2_t) + b_conv2_t
      #t_conv2_flat = tf.reshape(t_conv2, [-1, 8 * 8 * 32])
    # h_pool2 = max_pool_2x2(h_conv2)

     #dropout
   # with tf.name_scope('dropout'):
   #    keep_prob_t = tf.placeholder(tf.float32, name = "drop_in")
   #    t_drop = tf.nn.dropout(t_conv2, keep_prob_t)
    
    with tf.name_scope('Contact'):
        D_conv2_seq = tf.reshape(d_conv2,[-1,8 * 8,32])
        T_conv2_seq = tf.reshape(t_conv2,[-1,20 * 20,32])
        Demand_Topo = tf.concat([D_conv2_seq, T_conv2_seq], 1)
    # full-connected 1
    Demand_Topo_flat = tf.reshape(Demand_Topo, [-1, (64 + 400) * 32])
    with tf.name_scope('fully_1_layer'):
      with tf.name_scope('Weight'):
        W_fc1 = util.weight_variable([ (64 + 400)* 32, 512])
      with tf.name_scope('biases'):
        b_fc1 = util.bias_variable([512])
      with tf.name_scope('relu'):
        h_fc1 = tf.nn.relu(tf.matmul(Demand_Topo_flat, W_fc1) + b_fc1)

    # dropout
    with tf.name_scope('dropout'):
       keep_prob1 = tf.placeholder(tf.float32, name = "drop_in")
       h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob1)

    # full-connected 2
    with tf.name_scope('fully_2_layer'):
      with tf.name_scope('Weight'):
        W_fc2 = util.weight_variable([512, 128])
      with tf.name_scope('biases'):
        b_fc2 = util.bias_variable([128])
      with tf.name_scope('relu'):
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)



    # dropout
    with tf.name_scope('dropout'):
       keep_prob2 = tf.placeholder(tf.float32, name = "drop_in")
       h_drop = tf.nn.dropout(h_fc2, keep_prob2)

    # readout
    with tf.name_scope('fully_layer'):
      with tf.name_scope('Weight'):
        W_out = util.weight_variable([128, 1])
      with tf.name_scope('biases'):
        b_out = util.bias_variable([1])
    
      y_out = tf.matmul(h_drop, W_out) + b_out

    #return y_out, keep_prob1, keep_prob2, keep_prob_d, keep_prob_t
    return y_out, keep_prob1, keep_prob2

def SFNN(demand,topo) :

    x_demand = tf.reshape(demand, [-1,8 * 8])
    x_topo   = tf.reshape(topo, [-1,20 * 20])

    # Demand Fully Connected 1
    with tf.name_scope('D_fully_1_layer'):
      with tf.name_scope('Weight'):
        D_W_fc1 = util.weight_variable([64, 128])
      with tf.name_scope('biases'):
        D_b_fc1 = util.bias_variable([128])
      with tf.name_scope('relu'):
        D_h_fc1 = tf.nn.relu(tf.matmul(x_demand, D_W_fc1) + D_b_fc1)

    # Demand Fully Connected 1
    with tf.name_scope('D_fully_2_layer'):
      with tf.name_scope('Weight'):
        D_W_fc2 = util.weight_variable([ 128, 64])
      with tf.name_scope('biases'):
        D_b_fc2 = util.bias_variable([64])
      with tf.name_scope('relu'):
        D_h_fc2 = tf.nn.relu(tf.matmul(D_h_fc1, D_W_fc2) + D_b_fc2)


    # Topo Fully Connected 1
    with tf.name_scope('T_fully_1_layer'):
      with tf.name_scope('Weight'):
        T_W_fc1 = util.weight_variable([400, 512])
      with tf.name_scope('biases'):
        T_b_fc1 = util.bias_variable([512])
      with tf.name_scope('relu'):
        T_h_fc1 = tf.nn.relu(tf.matmul(x_topo, T_W_fc1) + T_b_fc1)

    # Topo Fully Connected 2
    with tf.name_scope('T_fully_2_layer'):
      with tf.name_scope('Weight'):
        T_W_fc2 = util.weight_variable([ 512, 256])
      with tf.name_scope('biases'):
        T_b_fc2 = util.bias_variable([256])
      with tf.name_scope('relu'):
        T_h_fc2 = tf.nn.relu(tf.matmul(T_h_fc1, T_W_fc2) + T_b_fc2)
   
    with tf.name_scope('Contact'):
        Demand_Topo = tf.concat([D_h_fc2, T_h_fc2], 1)
    # full-connected 1
    Demand_Topo_flat = tf.reshape(Demand_Topo, [-1,  256 + 64])
    with tf.name_scope('fully_1_layer'):
      with tf.name_scope('Weight'):
        W_fc1 = util.weight_variable([ 256 + 64, 512])
      with tf.name_scope('biases'):
        b_fc1 = util.bias_variable([512])
      with tf.name_scope('relu'):
        h_fc1 = tf.nn.relu(tf.matmul(Demand_Topo_flat, W_fc1) + b_fc1)



    # full-connected 2
    with tf.name_scope('fully_2_layer'):
      with tf.name_scope('Weight'):
        W_fc2 = util.weight_variable([512, 128])
      with tf.name_scope('biases'):
        b_fc2 = util.bias_variable([128])
      with tf.name_scope('relu'):
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

    # dropout
    with tf.name_scope('dropout'):
       keep_prob1 = tf.placeholder(tf.float32, name = "drop_in")
       h_drop = tf.nn.dropout(h_fc2, keep_prob1)
    # readout
    with tf.name_scope('fully_layer'):
      with tf.name_scope('Weight'):
        W_out = util.weight_variable([128, 1])
      with tf.name_scope('biases'):
        b_out = util.bias_variable([1])

      y_out = tf.matmul(h_drop, W_out) + b_out

    return y_out, keep_prob1

def Dang_NN(demand, topo):

    x_demand = tf.reshape(demand, [-1,8,8,1])
    x_topo   = tf.reshape(topo, [-1,20,20,1])

    # convolution 1
    with tf.name_scope('D_cnn_1_layer'):
      with tf.name_scope('Weight'):
        W_conv1_d = util.weight_variable([10, 10, 1, 15])
      with tf.name_scope('biases'):
        b_conv1_d = util.bias_variable([15])
    # h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
      d_conv1 = tf.nn.relu(util.conv2d(x_demand, W_conv1_d) + b_conv1_d)
    # h_pool1 = max_pool_2x2(h_conv1)

    # convolution 2
    with tf.name_scope('D_cnn_2_layer'):
      with tf.name_scope('Weight'):
        W_conv2_d = util.weight_variable([5, 5, 15, 25])
      with tf.name_scope('biases'):
        b_conv2_d = util.bias_variable([25])
    # h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
      d_conv2 = util.conv2d(d_conv1, W_conv2_d) + b_conv2_d
      #d_conv2_flat = tf.reshape(d_conv2, [-1, 8 * 8 * 32])
    # h_pool2 = max_pool_2x2(h_conv2)

    # dropout
    #with tf.name_scope('dropout'):
     #  keep_prob_d = tf.placeholder(tf.float32, name = "drop_in")
      # d_drop = tf.nn.dropout(d_conv2, keep_prob_d)

    # convolution 1
    with tf.name_scope('T_cnn_1_layer'):
      with tf.name_scope('Weight'):
        W_conv1_t = util.weight_variable([10, 10, 1, 15])
      with tf.name_scope('biases'):
        b_conv1_t = util.bias_variable([15])
    # h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
      t_conv1 = tf.nn.relu(util.conv2d(x_topo, W_conv1_t) + b_conv1_t)
    # h_pool1 = max_pool_2x2(h_conv1)

    # convolution 2
    with tf.name_scope('T_cnn_2_layer'):
      with tf.name_scope('Weight'):
        W_conv2_t = util.weight_variable([5, 5, 15, 25])
      with tf.name_scope('biases'):
        b_conv2_t = util.bias_variable([25])
    # h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
      t_conv2 = util.conv2d(t_conv1, W_conv2_t) + b_conv2_t
      #t_conv2_flat = tf.reshape(t_conv2, [-1, 8 * 8 * 32])
    # h_pool2 = max_pool_2x2(h_conv2)
 
    with tf.name_scope('Contact'):
        d_conv2_relu = tf.nn.relu(d_conv2)
        t_conv2_relu = tf.nn.relu(t_conv2)
        D_conv2_seq = tf.reshape(d_conv2_relu,[-1,8 * 8,25])
        T_conv2_seq = tf.reshape(t_conv2_relu,[-1,20 * 20,25])
        Demand_Topo = tf.concat([D_conv2_seq, T_conv2_seq], 1)
    # full-connected 1
    Demand_Topo_flat = tf.reshape(Demand_Topo, [-1, (64 + 400) * 25])
    with tf.name_scope('fully_1_layer'):
      with tf.name_scope('Weight'):
        W_fc1 = util.weight_variable([ (64 + 400)* 25, 10])
      with tf.name_scope('biases'):
        b_fc1 = util.bias_variable([10])
      with tf.name_scope('relu'):
        h_fc1 = tf.nn.relu(tf.matmul(Demand_Topo_flat, W_fc1) + b_fc1)

    # dropoutd
    with tf.name_scope('dropout'):
       keep_prob1 = tf.placeholder(tf.float32, name = "drop_in")
       h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob1)

    # readout
    with tf.name_scope('fully_layer'):
      with tf.name_scope('Weight'):
        W_out = util.weight_variable([10, 1])
      with tf.name_scope('biases'):
        b_out = util.bias_variable([1])

      y_out = tf.matmul(h_fc1_drop, W_out) + b_out

    #return y_out, keep_prob1, keep_prob2, keep_prob_d, keep_prob_t
    return y_out, keep_prob1

def SCNN_min(demand,topo) :

    x_demand = tf.reshape(demand, [-1,8,8,1])
    x_topo   = tf.reshape(topo, [-1,20,20,1])

    # convolution 1
    with tf.name_scope('D_cnn_1_layer'):
      with tf.name_scope('Weight'):
        W_conv1_d = util.weight_variable([3, 3, 1, 16])
      with tf.name_scope('biases'):
        b_conv1_d = util.bias_variable([16])
    # h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
      d_conv1 = tf.nn.relu(util.conv2d(x_demand, W_conv1_d) + b_conv1_d)
      #d_conv1 = util.conv2d(x_demand, W_conv1_d) + b_conv1_d
    # h_pool1 = max_pool_2x2(h_conv1)

    # convolution 2
    with tf.name_scope('D_cnn_2_layer'):
      with tf.name_scope('Weight'):
        W_conv2_d = util.weight_variable([3, 3, 16, 32])
      with tf.name_scope('biases'):
        b_conv2_d = util.bias_variable([32])
    # h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
      d_conv2 = tf.nn.relu(util.conv2d(d_conv1, W_conv2_d) + b_conv2_d)
      #d_conv2 = util.conv2d(d_conv1, W_conv2_d) + b_conv2_d
      #d_conv2_flat = tf.reshape(d_conv2, [-1, 8 * 8 * 32])
    # h_pool2 = max_pool_2x2(h_conv2)

    # dropout
    #with tf.name_scope('dropout'):
     #  keep_prob_d = tf.placeholder(tf.float32, name = "drop_in")
      # d_drop = tf.nn.dropout(d_conv2, keep_prob_d)

    # convolution 1
    with tf.name_scope('T_cnn_1_layer'):
      with tf.name_scope('Weight'):
        W_conv1_t = util.weight_variable([5, 5, 1, 16])
      with tf.name_scope('biases'):
        b_conv1_t = util.bias_variable([16])
    # h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
      t_conv1 = tf.nn.relu(util.conv2d(x_topo, W_conv1_t) + b_conv1_t)
      #t_conv1 = util.conv2d(x_topo, W_conv1_t) + b_conv1_t
    # h_pool1 = max_pool_2x2(h_conv1)

    # convolution 2
    with tf.name_scope('T_cnn_2_layer'):
      with tf.name_scope('Weight'):
        W_conv2_t = util.weight_variable([5, 5, 16, 32])
      with tf.name_scope('biases'):
        b_conv2_t = util.bias_variable([32])
    # h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
      t_conv2 = tf.nn.relu(util.conv2d(t_conv1, W_conv2_t) + b_conv2_t)
      #t_conv2 = util.conv2d(t_conv1, W_conv2_t) + b_conv2_t
      #t_conv2_flat = tf.reshape(t_conv2, [-1, 8 * 8 * 32])
    # h_pool2 = max_pool_2x2(h_conv2)

    # dropout
    #with tf.name_scope('dropout'):
     #  keep_prob_t = tf.placeholder(tf.float32, name = "drop_in")
      # t_drop = tf.nn.dropout(t_conv2, keep_prob_t)
    
    with tf.name_scope('Contact'):
        D_conv2_seq = tf.reshape(d_conv2,[-1,8 * 8,32])
        T_conv2_seq = tf.reshape(t_conv2,[-1,20 * 20,32])
        Demand_Topo = tf.concat([D_conv2_seq, T_conv2_seq], 1)
    # full-connected 1
    Demand_Topo_flat = tf.reshape(Demand_Topo, [-1, (64 + 400) * 32])
    with tf.name_scope('fully_1_layer'):
      with tf.name_scope('Weight'):
        W_fc1 = util.weight_variable([ (64 + 400)* 32, 32])
      with tf.name_scope('biases'):
        b_fc1 = util.bias_variable([32])
      with tf.name_scope('relu'):
        h_fc1 = tf.nn.relu(tf.matmul(Demand_Topo_flat, W_fc1) + b_fc1)

    # dropout
    with tf.name_scope('dropout'):
       keep_prob1 = tf.placeholder(tf.float32, name = "drop_in")
       h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob1)

    # full-connected 2
    with tf.name_scope('fully_2_layer'):
      with tf.name_scope('Weight'):
        W_fc2 = util.weight_variable([32, 16])
      with tf.name_scope('biases'):
        b_fc2 = util.bias_variable([16])
      with tf.name_scope('relu'):
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)



    # dropout
    with tf.name_scope('dropout'):
       keep_prob2 = tf.placeholder(tf.float32, name = "drop_in")
       h_drop = tf.nn.dropout(h_fc2, keep_prob2)

    # readout
    with tf.name_scope('fully_layer'):
      with tf.name_scope('Weight'):
        W_out = util.weight_variable([16, 1])
      with tf.name_scope('biases'):
        b_out = util.bias_variable([1])
    
      y_out = tf.matmul(h_drop, W_out) + b_out

    #return y_out, keep_prob1, keep_prob2, keep_prob_d, keep_prob_t
    return y_out, keep_prob1, keep_prob2

def Dang_FNN(demand, topo):

    x_demand = tf.reshape(demand, [-1,20,20,1])
    x_topo   = tf.reshape(topo, [-1,20,20,1])
    x_in = tf.concat([ x_demand, x_topo], 3)

    x_in_all = tf.reshape(x_in ,[-1,800])
    # y
    with tf.name_scope('D_fully_1_layer'):
      with tf.name_scope('Weight'):
        D_W_fc1 = util.weight_variable([800, 512])
      with tf.name_scope('biases'):
        D_b_fc1 = util.bias_variable([512])
      with tf.name_scope('relu'):
        D_h_fc1 = tf.nn.relu(tf.matmul(x_in_all, D_W_fc1) + D_b_fc1)

    with tf.name_scope('D_fully_1_layer'):
      with tf.name_scope('Weight'):
        D_W_fc2 = util.weight_variable([512, 128])
      with tf.name_scope('biases'):
        D_b_fc2 = util.bias_variable([128])
      with tf.name_scope('relu'):
        D_h_fc2 = tf.nn.relu(tf.matmul(D_h_fc1, D_W_fc2) + D_b_fc2)
    # dropoutd
    with tf.name_scope('dropout'):
       keep_prob1 = tf.placeholder(tf.float32, name = "drop_in")
       h_fc1_drop = tf.nn.dropout(D_h_fc2, keep_prob1)

    # readout
    with tf.name_scope('fully_layer'):
      with tf.name_scope('Weight'):
        W_out = util.weight_variable([128, 1])
      with tf.name_scope('biases'):
        b_out = util.bias_variable([1])

      y_out = tf.matmul(h_fc1_drop, W_out) + b_out

    #return y_out, keep_prob1, keep_prob2, keep_prob_d, keep_prob_t
    return y_out, keep_prob1



def loss_rate(y, y_out) :
    # loss
    with tf.name_scope('loss'):
       loss = tf.reduce_mean(tf.square(y - y_out))  # if axis = None, reduce all dimensions!
    return loss

def train(loss, L_rate) :
    #train_step
    with tf.name_scope('trian_op'):
       train_step = tf.train.AdamOptimizer(L_rate).minimize(loss)
    return train_step
def accurate(y_label, y_out, error_bound, batch_size) :
    #accuracy
    with tf.name_scope('accurate'):
        error_tf = tf.constant(error_bound ,shape=[ batch_size, 1 ],dtype = tf.float32)
        one_tf = tf.constant(True ,shape=[ batch_size, 1 ], dtype = tf.bool)
        error = tf.div( tf.abs(y_label - y_out), y_label)
        error_bool = tf.less_equal( error,error_tf)
        correct_prediction = tf.equal(error_bool, one_tf)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy




