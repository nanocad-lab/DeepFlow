###Use CUDA_VISIBLE_DEVICES=0,1,2... is used to make sure only the right GPUs
###are made visible

import argparse
import numpy as np
import os
import tensorflow as tf


#tf.debugging.set_log_device_placement(True)
def make_variables(k1, k2, initializer, type, k3=None):
    if k3 is None:
        if type == 'float32':
            return (tf.Variable(initializer(shape=[k1, k2], dtype=tf.float32)))
        elif type == 'float16':
            return (tf.Variable(initializer(shape=[k1, k2], dtype=tf.float16)))
        else:
            return NotImplemented
    else:
        if type == 'float32':
            return (tf.Variable(initializer(shape=[k1, k2, k3], dtype=tf.float32)))
        elif type == 'float16':
            return (tf.Variable(initializer(shape=[k1, k2, k3], dtype=tf.float16)))
        else:
            return NotImplemented


@tf.function
def matmul(a,b,device):
    with tf.device(device):
        return tf.matmul(a, b)

@tf.function
def RC(m, k, n, kern_para_a, kern_para_b, num_devices, a_shards, b_shards):
        #c_final = make_variables(m,n,tf.random_normal_initializer(mean=1., stddev=0.2),'float32')
        #c = tf.split(c_final, kern_para_b, axis=1)
        c_i = {}
        for i in range(kern_para_b):
            c_j = {}
            for j in range(kern_para_a):
                did = i*kern_para_a + j
                curr_device = '/device:gpu:' + str(did % num_devices)
                c_j[j] = matmul(a_shards[j], b_shards[i], curr_device)
            #c[1] = matmul(a_shards[0], b_shards[1], '/device:gpu:1')
            c_i[i] = tf.concat([c_j[k] for k in c_j],axis=0)
        with tf.device('/device:gpu:0'):
            c_final = tf.concat([c_i[i] for i in c_i], axis=1)
        return c_final

@tf.function
def CR(m, k, n, kern_para_a, num_devices, a_shards, b_shards):
    c_int = {}
    #start = tf.timestamp()
    for i in range(kern_para_a):
        curr_device = '/device:gpu:' + str(i % num_devices)
        c_int[i] = matmul(a_shards[i], b_shards[i], curr_device)
        #print(c_int[i])

    with tf.device('/device:gpu:0'):
        c_final = tf.math.add_n([c_int[c_i] for c_i in c_int])
    return c_final 


def main():
    # Graph:
    # a, b, weights, bias inputs
    # out = matmul(a, weights) + b * bias
    ###kern_para_a >1 for CR: kern_para_b>1 for RC. This isn't entirely true though
    ###TODO: Implement complete tiling for CR

    parser = argparse.ArgumentParser(formatter_class =
                argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-t', '--type', type=str, required=False, default='RC', help="for RC: parallelism along input dimension; for CR: parallelism along the inner dimension")
    parser.add_argument('-kp1', '--kern_para_a', type=int, required=False, default=1, help="for RC: parallelism along input dimension; for CR: parallelism along the inner dimension")
    parser.add_argument('-kp2', '--kern_para_b', type=int, required=False, default=1, help="for RC: parallelism along the outpiut dimension; for CR: NA")

    args = parser.parse_args()

    op_type = args.type #Either RC or CR
    kern_para_a = args.kern_para_a
    kern_para_b = args.kern_para_b
    # TODO: Parameterize later
    m = 10000
    k = 10000
    n = 10000
    num_devices = kern_para_a * kern_para_b 

    a = make_variables(m,k,tf.random_normal_initializer(mean=1., stddev=0),'float16')
    b = make_variables(k,n,tf.random_normal_initializer(mean=1., stddev=0),'float16')

    writer = tf.summary.create_file_writer("tensorboard.log")

    if op_type == "CR":
        a_shards = tf.split(a, kern_para_a, axis=1)
        b_shards = tf.split(b, kern_para_a, axis=0)
    if op_type == "RC":
        a_shards = tf.split(a, kern_para_a, axis=0)
        b_shards = tf.split(b, kern_para_b, axis=1)
    #else:
    #    b_shards = [b]

    #c_final = make_variables(m,n,tf.random_normal_initializer(mean=1., stddev=0.2),'float32')
    start = tf.timestamp()
    #tf.summary.trace_on(graph=True, profiler=False)
    
    #tf.profiler.experimental.start("logdir{}".format(num_devices))
    if op_type == "RC":
        c_final = RC(m, k, n, kern_para_a, kern_para_b, num_devices, a_shards, b_shards)
    elif op_type == "CR":
        c_final = CR(m, k, n, kern_para_a, num_devices, a_shards, b_shards)
    tot_time = tf.timestamp() - start
    #tf.profiler.experimental.stop()

    #with writer.as_default():
    #    tf.summary.trace_export(name="trace",step=0,profiler_outdir="tensorboard.log")
    #print(tf.shape(c_final), tot_time)
    #writer.flush()
    #print(c_final)
    #print(a, b, c_final)
    print("========================{}======================".format(tot_time))

if __name__ == "__main__":
    main()
