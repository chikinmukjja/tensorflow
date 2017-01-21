# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

xy = np.loadtxt('train.txt',unpack=True,dtype='float32')

x_data = xy[:-1]
y_data = xy[-1]

print x_data, y_data ,len(x_data)

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_uniform([1,len(x_data)], -1.0, 1.0))

# 행렬곱셈 1*3 3*6
h = tf.matmul(W,X)
hypothesis = tf.div(1., 1.+tf.exp(-h))

cost = -tf.reduce_mean(Y*tf.log(hypothesis)+(1-Y)*tf.log(1-hypothesis))

alpha = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(alpha)
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in xrange(2001):
    sess.run(train, feed_dict={X: x_data, Y: y_data})
    if step % 20 == 0:
        print step, sess.run(cost,feed_dict={X: x_data, Y: y_data}), sess.run(W)

print sess.run(hypothesis, feed_dict={X: [[1],[2],[2]]})
print sess.run(hypothesis, feed_dict={X: [[1],[1],[5]]})
print sess.run(hypothesis, feed_dict={X: [[1],[5],[8]]})



