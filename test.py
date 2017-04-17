# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 10:25:36 2017

@author: tang
"""

import tensorflow as tf
import argparse
import os
import cv2
import time

parser = argparse.ArgumentParser()
parser.add_argument("--LR_path")
parser.add_argument("--save_path")
parser.add_argument("--model_path")
parser.add_argument("--aug")
args = parser.parse_args()
LR_path = args.LR_path
save_path = args.save_path
model_path = args.model_path
aug = args.aug

graph = model_path + '.meta'
variables = model_path

def main():

    path = LR_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    names = os.listdir(path)
    name_pathes = [path + names[i] for i in range(len(names)) ]

    sess = tf.Session()
    new_saver = tf.train.import_meta_graph(graph)
    new_saver = tf.train.Saver(tf.global_variables())
    new_saver.restore(sess, variables)

    ex_time = []
    for i in range(len(name_pathes)):
        im_o = cv2.imread(name_pathes[i])
        if aug:
            im_lr = cv2.flip(im_o,1)
            im_ud = cv2.flip(im_o,0)
            im_lu = cv2.flip(im_o,-1)
    
        im_o = im_o.reshape([1,im_o.shape[0],im_o.shape[1],3])
        if aug:
            im_lr = im_lr.reshape([1,im_lr.shape[0],im_lr.shape[1],3])
            im_ud = im_ud.reshape([1,im_ud.shape[0],im_ud.shape[1],3])
            im_lu = im_lu.reshape([1,im_lu.shape[0],im_lu.shape[1],3])

        start = time.clock()
        out_o = sess.run('output/output_clip:0', feed_dict={'input/im_input:0': im_o})
        if aug:
            out_lr = sess.run('output/output_clip:0', feed_dict={'input/im_input:0': im_lr})
            out_ud = sess.run('output/output_clip:0', feed_dict={'input/im_input:0': im_ud})
            out_lu = sess.run('output/output_clip:0', feed_dict={'input/im_input:0': im_lu})    
        end = time.clock()
        ex_time.append(end - start)
    
        out_o = out_o.reshape([out_o.shape[1],out_o.shape[2],3])
        if aug:
            out_lr = out_lr.reshape([out_lr.shape[1],out_lr.shape[2],3])
            out_ud = out_ud.reshape([out_ud.shape[1],out_ud.shape[2],3])
            out_lu = out_lu.reshape([out_lu.shape[1],out_lu.shape[2],3])    

            out_lr = cv2.flip(out_lr,1)
            out_ud = cv2.flip(out_ud,0)
            out_lu = cv2.flip(out_lu,-1)

            out = (out_o + out_lr + out_ud + out_lu) * 63.75
        else:
            out = out_o * 255

        cv2.imwrite(save_path + names[i], out)
    print('Average excutive time per image:')
    print(sum(ex_time)/len(ex_time))
#    cv2.imwrite(save_path + names[i], output, [cv2.IMWRITE_PNG_COMPRESSION, 0])
if __name__ == '__main__':
    main()

