import numpy as np
import os
import tensorflow as tf


tf.debugging.set_log_device_placement(True)
def make_variables(k1, k2, initializer, type):
    if type == 'float32':
        return (tf.Variable(initializer(shape=[k1, k2], dtype=tf.float32)))
    elif type == 'float16':
        return (tf.Variable(initializer(shape=[k1, k2], dtype=tf.float16)))
    else:
        return NotImplemented

@tf.function
def matmul(a,b,device):
    with tf.device(device):
        return tf.matmul(a, b)

@tf.function
def RC(m, k, n, kern_para_b, num_devices, a_shards, b_shards, c_final):
        #c_final = make_variables(m,n,tf.random_normal_initializer(mean=1., stddev=0.2),'float32')
        c = tf.split(c_final, kern_para_b, axis=1)
        for i in range(kern_para_b):
            curr_device = '/device:gpu:' + str(i % num_devices)
            print(curr_device)
            c[i] = matmul(a_shards, b_shards[i], curr_device)
            #c[1] = matmul(a_shards[0], b_shards[1], '/device:gpu:1')
        with tf.device('/device:gpu:0'):
            c_final = tf.concat([ci for ci in c], axis=1)
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
    m = None
    k = None
    n = None
    ###kern_para_a >1 for CR: kern_para_b>1 for RC. This isn't entirely true though
    ###TODO: Implement complete tiling for RC and CR
    kern_para_a = 4
    kern_para_b = 1
    b_dims = (k, n)
    c_dims = (m, n)

    # TODO: Parameterize later
    m = 32000
    k = 12000
    n = 32000
    num_devices = 1

    a_dims = (m, k)
    a_initer = tf.random_normal_initializer()
    b_initer = tf.random_normal_initializer()
    a = make_variables(m,k,tf.random_normal_initializer(mean=1., stddev=0.),'float16')
    b = make_variables(k,n,tf.random_normal_initializer(mean=1., stddev=0.),'float16')

    writer = tf.summary.create_file_writer("tensorboard.log")
    tf.summary.trace_on(graph=True, profiler=True)

    if kern_para_a > 1:
        a_shards = tf.split(a, kern_para_a, axis=1)
        b_shards = tf.split(b, kern_para_a, axis=0)
    if kern_para_b > 1:
        b_shards = tf.split(b, kern_para_b, axis=1)
    #else:
    #    b_shards = [b]

    c_final = make_variables(m,n,tf.random_normal_initializer(mean=1., stddev=0.2),'float32')
    if ((kern_para_a == 1) and (kern_para_b >1)):
        RC(m, k, n, kern_para_b, num_devices, a, b_shards, c_final)
    elif ((kern_para_b == 1) and (kern_para_a >1)):
        CR(m, k, n, kern_para_a, num_devices, a_shards, b_shards)
    with writer.as_default():
        tf.summary.trace_export(name="trace",step=0,profiler_outdir="tensorboard.log")
    print(tf.shape(c_final))
    writer.flush()
    #print(a, b, c_final)


if __name__ == "__main__":
    main()
