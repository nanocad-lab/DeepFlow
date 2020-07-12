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

def main():
    # Graph:
    # a, b, weights, bias inputs
    # out = matmul(a, weights) + b * bias
    m = None
    k = None
    n = None
    ###kern_para_a >1 for CR: kern_para_b>1 for RC. This isn't entirely true though
    ###TODO: Implement complete tiling for RC and CR
    kern_para_a = 1
    kern_para_b = 4
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

    if kern_para_a > 1:
        a_shards = tf.split(a, kern_para_a, axis=1)
        b_shards = tf.split(b, kern_para_a, axis=0)
    if kern_para_b > 1:
        b_shards = tf.split(b, kern_para_b, axis=1)
    #else:
    #    b_shards = [b]

    c_final = make_variables(m,n,tf.random_normal_initializer(mean=1., stddev=0.2),'float32')
    if ((kern_para_a == 1) and (kern_para_b >1)):
        c = tf.split(c_final, kern_para_b, axis=1)
        start = tf.timestamp()
        for i in range(kern_para_b):
            curr_device = '/device:gpu:' + str(i % num_devices)
            print(curr_device)
            c[i] = matmul(a, b_shards[i], curr_device)
            #c[1] = matmul(a_shards[0], b_shards[1], '/device:gpu:1')
        with tf.device('/device:gpu:0'):
            c_final = tf.concat([ci for ci in c], axis=1)
        tot = tf.timestamp() - start
    elif ((kern_para_b == 1) and (kern_para_a >1)):
        c_int = {}
        start = tf.timestamp()
        for i in range(kern_para_a):
            curr_device = '/device:gpu:' + str(i % num_devices)
            c_int[i] = matmul(a_shards[i], b_shards[i], curr_device)
            #print(c_int[i])

        with tf.device('/device:gpu:0'):
            c_final = tf.math.add_n([c_int[c_i] for c_i in c_int])
        tot = tf.timestamp() - start
    print(tf.shape(c_final), tot)
    #print(a, b, c_final)


if __name__ == "__main__":
    main()
