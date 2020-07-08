import numpy as np
import os
import tensorflow as tf


class Optimizer():
    def __init__(self, learning_rate):
        self.opt = tf.train.GradientDescentOptimizer(learning_rate)

    def __call__(self, loss):
        with tf.variable_scope("Gradient"):
            with tf.variable_scope("Compute"):
                gradient = self.opt.compute_gradients(loss)

            with tf.variable_scope("Apply"):
                train_step = self.opt.apply_gradients(gradient)
        return train_step

def main():
    # Graph:
    # a, b, weights, bias inputs
    # out = matmul(a, weights) + b * bias
    m = None
    k = None
    n = None
    kern_para_a = 1
    kern_para_b = 3
    b_dims = (k, n)
    c_dims = (m, n)

    # TODO: Parameterize later
    m = 128
    k = 64
    n = 18

    a_dims = (m, k)
    a_initer = tf.random_normal_initializer()
    a = tf.get_variable(name='A', shape=a_dims, dtype=tf.float32, initializer=a_initer)
    b = tf.placeholder(tf.float32, shape=b_dims, name='B')
    if kern_para_a > 1:
        a_shards = tf.split(a, kern_para_a, axis=0)
    else:
        a_shards = [a]
    if kern_para_b > 1:
        b_shards = tf.split(b, kern_para_b, axis=1)
    else:
        b_shards = [b]
    matmuls = []
    for a_idx, a_shard in enumerate(a_shards):
        matmuls.append([])
        for b_idx, b_shard in enumerate(b_shards):
            assert len(matmuls[a_idx]) == b_idx
            matmuls[a_idx].append(tf.matmul(a_shard, b_shard, name='mm_{}_{}'.format(a_idx, b_idx)))

    concat_subs = []
    for a_idx in range(len(a_shards)):
        concat_subs.append(tf.concat(matmuls[a_idx], axis=1))
    c = tf.concat(concat_subs, axis=0)

    c_ph = tf.placeholder(tf.float32, shape=c_dims, name='C')
    loss = tf.subtract(c_ph, c)
    opt = Optimizer(0.1)
    train_op = opt(loss)    

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        a_in = sess.run(a)
        b_in = np.random.normal(size=(k, n))
        c_out = np.matmul(a_in, b_in)
        feed_dict = {b: b_in,
                     c_ph: c_out}
        c_final, _ = sess.run([c, train_op], feed_dict=feed_dict)
        assert np.allclose(c_out, c_final, atol=1e-3)
        tf.summary.FileWriter(os.path.join('.'), sess.graph)
        saver = tf.train.Saver()
        saver.save(sess, os.path.join('.', 'tf_gemm_sharding'))

if __name__ == "__main__":
    main()

