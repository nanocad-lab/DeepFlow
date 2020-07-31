#!/usr/bin/env python3

###Use CUDA_VISIBLE_DEVICES=0,1,2... is used to make sure only the right GPUs
###are made visible

import argparse
import numpy as np
import os
import tensorflow as tf
import time as _time

tf.debugging.set_log_device_placement(True)

@tf.function
def RC(m, k, n, kern_para_a, kern_para_b, num_devices, a_shards, b_shards):
        c = [None] * kern_para_a 
        c_final = [None] * kern_para_a
        for i in range(kern_para_a):
            c[i] = [None] * kern_para_b
            for j in range(kern_para_b):
                gid = i * kern_para_b + j
                with tf.device('/device:gpu:{}'.format(gid)):
                    c[i][j] = tf.matmul(a_shards[i], b_shards[j])

            with tf.device('/device:gpu:{}'.format(i * kern_para_b)):
                c_final[i] = tf.concat(c[i],axis=0)

        return c_final

@tf.function
def CR(m, k, n, kern_para_a, num_devices, a_shards, b_shards):
    
    c = [None] * kern_para_a 
    for i in range(kern_para_a):
        with tf.device('/device:gpu:{}'.format(i)):
          c[i] = tf.matmul(a_shards[i], b_shards[i])
          
    for i in range(kern_para_a):
        with tf.device('/device:gpu:{}'.format(i)):
          c_final = tf.math.add_n(c)

    return c_final


def main():
    
    parser = argparse.ArgumentParser(formatter_class =
                argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-t', '--type', type=str, required=False, default='RC', help="for RC: parallelism along input dimension; for CR: parallelism along the inner dimension")
    parser.add_argument('-kp1', '--kern_para_a', type=int, required=False, default=1, help="for RC: parallelism along input dimension; for CR: parallelism along the inner dimension")
    parser.add_argument('-kp2', '--kern_para_b', type=int, required=False, default=1, help="for RC: parallelism along the outpiut dimension; for CR: NA")
    parser.add_argument('-N', '--num_gpus', type=int, required=False, default=1, help="Number of GPUs available for parallelization")

    args = parser.parse_args()

    op_type = args.type #Either RC or CR
    kern_para_a = args.kern_para_a
    kern_para_b = args.kern_para_b
    # TODO: Parameterize later
    m = 10000
    k = 20000
    n = 10000
    num_devices = args.num_gpus

    print("op_type: {}  kern_para_a: {} kern_para_b: {} num_devices: {}" .format(op_type, kern_para_a, kern_para_b, num_devices))

    initializer =  tf.random_normal_initializer(mean=1., stddev=0)
    weights = []
    activs  = []

    #Create and initialize Variables
    if op_type == "RC":
      a_dim = (m//kern_para_a, k)
      w_dim = (k, n//kern_para_b)

      for i in range(kern_para_a):
          did = i * kern_para_b
          curr_device = '/device:gpu:' + str(did)
          with tf.device(curr_device):
              a_shard = tf.Variable(
                  initial_value=tf.random.normal(shape=a_dim), 
                  name="a_{}".format(i))
              activs.append(a_shard)

      for j in range(kern_para_b):
          curr_device = '/device:gpu:' + str(j)
          with tf.device(curr_device):
              w_shard = tf.Variable(
                  initial_value=tf.random.normal(shape=w_dim), 
                  name="w_{}".format(j))
              weights.append(w_shard)


      c_final = RC(m, k, n, kern_para_a, kern_para_b, num_devices, activs, weights)

    elif op_type == "CR":
        a_dim = (m, k//kern_para_a)
        w_dim = (k//kern_para_a, n)
        for i in range(kern_para_a):
          curr_device = '/device:gpu:' + str(i)
          with tf.device(curr_device):
              a_shard = tf.Variable(initial_value=tf.random.normal(shape=a_dim), name="a_{}".format(i))
              w_shard = tf.Variable(initial_value=tf.random.normal(shape=w_dim), name="w_{}".format(i))
              
              activs.append(a_shard)
              weights.append(w_shard)

        c_final = CR(m, k, n, kern_para_a, num_devices, activs, weights)

    
    #Measure time

    start = _time.time()
    if op_type == "RC":
        c_final = RC(m, k, n, kern_para_a, kern_para_b, num_devices, activs, weights)
    elif op_type == "CR":
        c_final = CR(m, k, n, kern_para_a, num_devices, activs, weights)


    tot_time = _time.time() - start
    #tf.profiler.experimental.stop()

    #with writer.as_default():
    #    tf.summary.trace_export(name="trace",step=0,profiler_outdir="tensorboard.log")
    #writer.flush()
    #print(c_final)
    #print(a, b, c_final)
    print("========================{}======================".format(tot_time))

if __name__ == "__main__":
    main()
