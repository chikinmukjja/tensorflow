# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

xy = np.loadtxt('xor_train.txt',unpack=True,dtype='float32')
# reshape를 해주어야함
x_data = np.transpose(xy[0:-1])
y_data = np.reshape(xy[-1],(4,1))

print x_data, y_data ,len(x_data)
print np.array(x_data).shape, np.array(y_data).shape

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_uniform([2,2],-1.0, 1.0))
W2 = tf.Variable(tf.random_uniform([2,1],-1.0, 1.0))

b1 = tf.Variable(tf.zeros([2]),name='bias1')
b2 = tf.Variable(tf.zeros([1]),name='bias2')

L2 = tf.sigmoid(tf.matmul(X,W1)+b1)
hypothesis = tf.sigmoid(tf.matmul(L2,W2)+b2)

cost = -tf.reduce_mean(Y*tf.log(hypothesis)+(1-Y)*tf.log(1-hypothesis))

alpha = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(alpha)
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in xrange(11000):
    sess.run(train, feed_dict={X: x_data, Y: y_data})
    if step % 1000 == 0:
        print step, sess.run(cost,feed_dict={X: x_data, Y: y_data}), sess.run(W1), sess.run(W2)

print sess.run(tf.floor(hypothesis+0.5), feed_dict={X: [[0,0]]})
print sess.run(tf.floor(hypothesis+0.5), feed_dict={X: [[0,1]]})
print sess.run(tf.floor(hypothesis+0.5), feed_dict={X: [[1,0]]})
print sess.run(tf.floor(hypothesis+0.5), feed_dict={X: [[1,1]]})
