#!/usr/bin/env python3

###Use CUDA_VISIBLE_DEVICES=0,1,2... is used to make sure only the right GPUs
###are made visible

import argparse
import numpy as np
import os
import tensorflow as tf
import time as _time
import timeit

tf.debugging.set_log_device_placement(True)
v = tf.Variable(1)

@tf.function
def matmul(a, b, gid):
  with tf.device('/GPU:{}'.format(gid)):
    return tf.matmul(a,b)

def all_gather(c, kern_para_a, kern_para_b, num_devices):
    tmp = [None] * kern_para_a
    c_new = [None] * kern_para_a
    for i in range(kern_para_a):
        tmp[i] = [None] * kern_para_b
        c_new[i] = [None] * kern_para_b
        for j in range(kern_para_b):
            tmp[i][j] = [None] * kern_para_b
            with tf.device('/device:gpu:{}'.format((i*kern_para_b + j)%num_devices)):
                tmp[i][j][j] = tf.identity(c[i][j])
                #c_new[i][j] = c[i][j]

        for k in range(kern_para_b - 1):
            for j in range(kern_para_b):
                with tf.device('/device:gpu:{}'.format((i*kern_para_b + j)%num_devices)):
                    tmp[i][j][(j-k+2*kern_para_b-1)%kern_para_b] = tf.identity(tmp[i][(j-1)%kern_para_b][(j-k+2*kern_para_b-1)%kern_para_b])
                    #tf.print("{}{} ::: {}{}" .format(j, (j-k+2*kern_para_b-1)%kern_para_b, (j-1)%kern_para_b, (j-k+2*kern_para_b-1)%kern_para_b))

        for j in range(kern_para_b):
            with tf.device('/device:gpu:{}'.format((i*kern_para_b + j)%num_devices)):
                c_new[i][j] = (tf.concat([tmp[i][j]],axis=1))
    return c_new

@tf.function
def RC(m, k, n, kern_para_a, kern_para_b, num_devices, a_shards, b_shards):
        c = [None] * kern_para_a
        for i in range(kern_para_a):
            c[i] = [None] * kern_para_b
            for j in range(kern_para_b):
                gid = i * kern_para_b + j
                with tf.device('/device:gpu:{}'.format(gid%num_devices)):
                    c[i][j] = tf.matmul(a_shards[i][j], b_shards[i][j])
        c_new = all_gather(c, kern_para_a, kern_para_b, num_devices)
        return c_new

#@tf.function
def Col(m, k, n, kern_para_a, kern_para_b, num_devices, a_shards, b_shards):
        c = [None] * kern_para_a
        #v.assign_add(1)
        for i in range(kern_para_a):
            with tf.device('/device:gpu:{}'.format(i%num_devices)):
                c[i] = tf.matmul(a_shards[i][0], b_shards[i][0])
            
        return c


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
    parser.add_argument('-m', '--input_dim', type=int, required=False, default=32768, help="input dimension")
    parser.add_argument('-n', '--output_dim', type=int, required=False, default=32768, help="output dimension")
    parser.add_argument('-k', '--inner_dim', type=int, required=False, default=32768, help="inner dimension")

    args = parser.parse_args()

    op_type = args.type #Either RC or CR
    kern_para_a = args.kern_para_a
    kern_para_b = args.kern_para_b
    m = args.input_dim 
    k = args.inner_dim
    n = args.output_dim
    num_devices = args.num_gpus

    print("op_type: {}  kern_para_a: {} kern_para_b: {} num_devices: {}" .format(op_type, kern_para_a, kern_para_b, num_devices))


    #Create and initialize Variables
    weights = [None] * kern_para_a
    activs  = [None] * kern_para_a

    '''
    if kern_para_a == 1 and kern_para_b == 1:
      a_dim = (m, k)
      w_dim = (k, n)
      activs = tf.Variable(initial_value=tf.random.normal(shape=a_dim), name="a")
      weights = tf.Variable(initial_value=tf.random.normal(shape=w_dim), name="w")
    '''
    if op_type == "RC":

      a_dim = (m//kern_para_a, k)
      w_dim = (k, n//kern_para_b)

      print("a_dim: {}".format(a_dim))
      print("b_dim: {}".format(w_dim))

      for i in range(kern_para_a):
        activs[i] = [None] * kern_para_b
        weights[i] = [None] * kern_para_b

        for j in range(kern_para_b):
          did = i * kern_para_b + j
          curr_device = '/device:gpu:' + str(did%num_devices)
          with tf.device(curr_device):
              a_shard = tf.Variable(
                  initial_value=tf.random.normal(shape=a_dim, mean=1.0, stddev=0.0, dtype=tf.dtypes.float16),
                  name="a_{}_{}".format(i,j))
              activs[i][j] = a_shard

              w_shard = tf.Variable(
                  initial_value=tf.random.normal(shape=w_dim,  mean=1.0, stddev=0.0, dtype=tf.dtypes.float16),
                  name="w_{}".format(j))
              weights[i][j] = w_shard

    elif op_type == "CR":
        a_dim = (m, k//kern_para_a)
        w_dim = (k//kern_para_a, n)
        for i in range(kern_para_a):
          curr_device = '/device:gpu:' + str(i)
          with tf.device(curr_device):
              a_shard = tf.Variable(initial_value=tf.random.normal(shape=a_dim, dtype=tf.dtypes.float16), name="a_{}".format(i))
              w_shard = tf.Variable(initial_value=tf.random.normal(shape=w_dim, dtype=tf.dtypes.float16), name="w_{}".format(i))

              activs[i] = a_shard
              weights[i] = w_shard



    #Measure time


    for i in range(100):
       start = _time.perf_counter()
       if op_type == "RC":
           if kern_para_b == 1:
              Col(m, k, n, kern_para_a, kern_para_b, num_devices, activs, weights)
           elif kern_para_b > 1:
              c_final = RC(m, k, n, kern_para_a, kern_para_b, num_devices, activs, weights)

       elif op_type == "CR":
           c_final = CR(m, k, n, kern_para_a, num_devices, activs, weights)

       #_ = c_final.numpy()  # Make sure to execute op and not just enqueue it
       tot_time = _time.perf_counter() - start
       print("Step{}: {}".format(i, tot_time))
    #tf.profiler.experimental.stop()

    #with writer.as_default():
    #    tf.summary.trace_export(name="trace",step=0,profiler_outdir="tensorboard.log")
    #writer.flush()
    #print(c_final)
    #print(a, b, c_final)

if __name__ == "__main__":
    main()
