# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

xy = np.loadtxt("train_softmax.txt",unpack=True,dtype="float32")
x_data = np.transpose(xy[0:3])
y_data = np.transpose(xy[3:])

print x_data, y_data
print np.array(x_data).shape, np.array(y_data).shape



X = tf.placeholder("float",[None,3])
Y = tf.placeholder("float",[None,3])

W = tf.Variable(tf.zeros([3,3]))

# multinomial classification
# W : 3(lables)*3(features)
# X : 8(num of data) * 3features

hypothesis = tf.nn.softmax(tf.matmul(X,W))

learning_rate = 0.001

# Y*tf.log(hypothesis) is elementary multiplication
# tf.reduce_sum is sum of all elem
# reduction_indices is old(deprecated) name of axis, single value / num of dimension of axis(denominator)
# return a tensor with a single element
#cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), reduction_indices=1))

cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print x_data, sess.run(W), sess.run(tf.matmul(X,W),feed_dict={X: x_data}), sess.run(hypothesis, feed_dict={X:x_data})
    print sess.run(tf.reduce_sum(Y*tf.log(hypothesis)),feed_dict={X: x_data,Y: y_data})
    print '-'*10
    print sess.run(cost,feed_dict={X: x_data, Y: y_data})
    for step in xrange(2001):
        sess.run(optimizer,feed_dict={X:x_data, Y:y_data})
        if step % 20 == 0:
            print step, sess.run(cost,feed_dict={X:x_data, Y:y_data}), sess.run(W)

    a = sess.run(hypothesis,feed_dict={X:[[1,11,7]]})
    print a, sess.run(tf.arg_max(a,1))
    b = sess.run(hypothesis, feed_dict={X: [[1, 3, 4]]})
    print b, sess.run(tf.arg_max(b, 1))
    c = sess.run(hypothesis, feed_dict={X: [[1, 1, 0]]})
    print c, sess.run(tf.arg_max(c, 1))

