# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 11:36:45 2017

@author: tang
"""

import tensorflow as tf
import tensorlayer as tl
#import time
import os
import numpy as np
import random
"""
tf.__version__
tl.__version__
"""
from tzmtf import SR
from tzmtf import data_read
from tzmtf import NTIRE_Net

IM_CHANNELS = 3
SCALE = 2
EVAL_FREQUENCY = 3000
SHOW_FREQUENCY = 50
ITERATIONS = 0
BATCH_SIZE = 2
BATCH_SIZE_VAL = 5
EPOCH = 5
DATA_PATH = '/media/tang/RAID0/NTIRE/X2train.h5'
VAL_PATH = '/home/tang/NTIRE/X2_val.h5'
LR = 0.0001

SR.print_time()

sess = tf.InteractiveSession()

print("Building graph...")

with tf.name_scope('input'):
    im_input = tf.placeholder(tf.float32, shape=[None, None, None, IM_CHANNELS], name='im_input')
    im_label = tf.placeholder(tf.float32, shape=[None, None, None, IM_CHANNELS], name='im_label')
    im_input_norm = tf.div(im_input, 255.0)
    im_label_norm = tf.div(im_label, 255.0)

    im_b = im_input_norm[:,:,:,0:1]
    im_g = im_input_norm[:,:,:,1:2]
    im_r = im_input_norm[:,:,:,2:3]

    im_input_b = tl.layers.InputLayer(im_b, name='input_b')
    im_input_g = tl.layers.InputLayer(im_g, name='input_g')    
    im_input_r = tl.layers.InputLayer(im_r, name='input_r')

with tf.name_scope('share_net'):
    share_b = NTIRE_Net.share_net(im_input_b, variable_scope='share', scale=SCALE, channels=64, 
                                  repetition_1=10, kernel_size=3, reuse=False)

    share_g = NTIRE_Net.share_net(im_input_g, variable_scope='share', scale=SCALE, channels=64, 
                                  repetition_1=10, kernel_size=3, reuse=True)

    share_r = NTIRE_Net.share_net(im_input_r, variable_scope='share', scale=SCALE, channels=64, 
                                  repetition_1=10, kernel_size=3, reuse=True)
                                  

    share = tl.layers.ConcatLayer([share_b, share_g, share_r], concat_dim=3, name='share_concat')


with tf.name_scope('res_net'):
    net = share
    net = NTIRE_Net.com_res_net_1(net, variable_scope='res_net', channels=64, 
                                  name='res_net_1', kernel_size=3, reuse=False)
    net = NTIRE_Net.com_res_net_1(net, variable_scope='res_net', channels=64, 
                                  name='res_net_2', kernel_size=3, reuse=False)
    net = NTIRE_Net.com_res_net_1(net, variable_scope='res_net', channels=64, 
                                  name='res_net_3', kernel_size=3, reuse=False)
    net = NTIRE_Net.com_res_net_1(net, variable_scope='res_net', channels=64, 
                                  name='res_net_4', kernel_size=3, reuse=False)
    net = NTIRE_Net.com_res_net_1(net, variable_scope='res_net', channels=64, 
                                  name='res_net_5', kernel_size=3, reuse=False)
    net = NTIRE_Net.com_res_net_1(net, variable_scope='res_net', channels=64, 
                                  name='res_net_6', kernel_size=3, reuse=False)

output_net = net

with tf.name_scope('output'):
    output = tf.clip_by_value(output_net.outputs, clip_value_min=0, clip_value_max=1, 
                              name='output_clip')

with tf.variable_scope('global_step'):
    global_step = tf.Variable(0, trainable=False, name='global_step')
#    learning_rate = tf.train.exponential_decay(LR, global_step, 500, 0.96, staircase=True)
    learning_rate = LR

with tf.name_scope('train'):
    im_y = im_label_norm[:,5+SCALE:-(5+SCALE),5+SCALE:-(5+SCALE),:]
    im_y_ = output_net.outputs[:,5+SCALE:-(5+SCALE),5+SCALE:-(5+SCALE),:]
    example_cost = tf.reduce_mean(tf.squared_difference(im_y, im_y_), [1, 2, 3])
    example_cost = tf.add(example_cost, 0.000001)
    robust_cost = tf.reduce_mean( tf.sqrt(example_cost) )

    cost_mse = tf.losses.mean_squared_error(output_net.outputs[:,5+SCALE:-(5+SCALE),5+SCALE:-(5+SCALE),:], 
                                             im_label_norm[:,5+SCALE:-(5+SCALE),5+SCALE:-(5+SCALE),:])
#    cost_res1 = tf.losses.mean_squared_error(output_net.outputs, im_label_norm)
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(robust_cost, global_step=global_step)

###############################################################################################
tl.layers.print_all_variables()
tl.layers.initialize_global_variables(sess)
###############################################################################################
os.system("rm -rf tmp/save_graph_logs")
os.system("rm -rf tmp/load")
tf.train.write_graph(sess.graph_def, "tmp/load", "test.pb", True)
tf.train.export_meta_graph('tmp/load/graph', as_text=True)
saver = tf.train.Saver(tf.global_variables(), name='saver')

print("Finish graph!-->Start training!")

def batch_read(data_list, data, label, noising=True):
    batch_x = np.zeros([len(data_list),data.shape[1],data.shape[2],data.shape[3]], dtype='float32')
    batch_y = np.zeros([len(data_list),label.shape[1],label.shape[2],label.shape[3]], dtype='float32')
    
    for i in range(len(data_list)):
        batch_x[i,:,:,:] = data[data_list[i],:,:,:]
        batch_y[i,:,:,:] = label[data_list[i],:,:,:]
        
    if noising:
        if_noise = random.randint(0, 19)
        if if_noise<len(data_list):
            sigma = 0.5 * random.randint(1, 9)
            noise = np.random.normal(0, sigma, (1,data.shape[1],data.shape[2],data.shape[3]))
            batch_x[0,:,:,:] = batch_x[0,:,:,:] + noise
    
    return batch_y, batch_x

data, label, dataset = data_read.open_dataset(DATA_PATH)
data_v, label_v, dataset_v = data_read.open_dataset(VAL_PATH)
batch_size = BATCH_SIZE
temp_psnr = 0

for i in range(EPOCH):
    
    randomdata = range(0,len(data))
    randomlist = random.sample(randomdata, len(data))

    for j in range(0, (len(data)-len(data)%batch_size)-batch_size+1, batch_size):
        
        data_list = randomlist[j:j+batch_size]
        batch_y, batch_x = batch_read(data_list=data_list, data=data, label=label, noising=False)
        sess.run(train_step, feed_dict={im_input: batch_x, im_label: batch_y})
        
        ITERATIONS = ITERATIONS + 1

        if ITERATIONS%SHOW_FREQUENCY == 0:
            SR.print_time()
            print('ITERATIONS:', sess.run(global_step))
                
        if ITERATIONS%EVAL_FREQUENCY == 0:
            psnr = []
            for l in range(0, (len(data_v)-len(data_v)%BATCH_SIZE_VAL)-BATCH_SIZE_VAL+1, BATCH_SIZE_VAL):
                batch_x = data_v[l:l+BATCH_SIZE_VAL,:,:,:]
                batch_y = label_v[l:l+BATCH_SIZE_VAL,:,:,:]

                loss_val = sess.run(cost_mse, feed_dict={im_input: batch_x, im_label: batch_y})
                psnr_val = 10.0 * np.log(1.0 / loss_val)/np.log(10.0)
                psnr.append(psnr_val)
            
            tem = sum(psnr)/len(psnr)
            print('The psnr of val_set is', tem)

            if tem - 0.01 > temp_psnr:
                saver.save(sess, 'tmp/variables', global_step=global_step)
                print( 'Saving model on the', sess.run(global_step), 'th iterations' )
                temp_psnr = tem

saver.save(sess, 'tmp/variables', global_step=global_step)
print( 'Saving model on the', ITERATIONS, 'th iterations' )
SR.print_time()


