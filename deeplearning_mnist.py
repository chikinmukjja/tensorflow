# -*- coding: utf-8 -*-
from random import randint
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

import input_data
import xavier_init

learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1

# Reading data and set variables
mnist = input_data.read_data_sets("/tmp/data/",one_hot=True)

X = tf.placeholder("float",[None,784])
Y = tf.placeholder("float",[None,10])

"""
accuracy : 94.52
W1 = tf.Variable(tf.random_normal([784,256]))
W2 = tf.Variable(tf.random_normal([256,256]))
W3 = tf.Variable(tf.random_normal([256,10]))
"""

#use xavier init
#accuraccy : 98.05 -> 98.31
W1 = tf.get_variable("W1",shape=[28*28,600],initializer=xavier_init.init(28*28,600))
W2 = tf.get_variable("W2",shape=[600,512],initializer=xavier_init.init(600,512))
W3 = tf.get_variable("W3",shape=[512,324],initializer=xavier_init.init(512,324))
W4 = tf.get_variable("W4",shape=[324,256],initializer=xavier_init.init(324,256))
W5 = tf.get_variable("W5",shape=[256,10],initializer=xavier_init.init(256,10))


B1 = tf.Variable(tf.random_normal([600]))
B2 = tf.Variable(tf.random_normal([512]))
B3 = tf.Variable(tf.random_normal([324]))
B4 = tf.Variable(tf.random_normal([256]))
B5 = tf.Variable(tf.random_normal([10]))

""" 2layer
L1 = tf.nn.relu(tf.add(tf.matmul(X,W1),B1))
L2 = tf.nn.relu(tf.add(tf.matmul(L1,W2),B2))
hypothesis = tf.add(tf.matmul(L2,W3),B3)
"""

# dropout 98.15
dropout_rate = tf.placeholder("float")
_L1 = tf.nn.relu(tf.add(tf.matmul(X,W1),B1))
L1 = tf.nn.dropout(_L1,dropout_rate)
_L2 = tf.nn.relu(tf.add(tf.matmul(L1,W2),B2))
L2 = tf.nn.dropout(_L2,dropout_rate)
_L3 = tf.nn.relu(tf.add(tf.matmul(L2,W3),B3))
L3 = tf.nn.dropout(_L3,dropout_rate)
_L4 = tf.nn.relu(tf.add(tf.matmul(L3,W4),B4))
L4 = tf.nn.dropout(_L4,dropout_rate)
hypothesis = tf.add(tf.matmul(L4,W5),B5)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(hypothesis,Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            """
            sess.run(optimizer, feed_dict={X: batch_xs, Y: batch_ys})
            avg_cost += sess.run(cost, feed_dict={X: batch_xs, Y: batch_ys})/batch_size
            """
            sess.run(optimizer,feed_dict={X: batch_xs, Y:batch_ys,dropout_rate: 0.7})
            avg_cost += sess.run(cost, feed_dict={X: batch_xs, Y: batch_ys,dropout_rate:1}) / batch_size
        if epoch % display_step == 0:
            print "Epoch: ", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost)

    print "Optimization Finished!"

    correct_prediction = tf.equal(tf.argmax(hypothesis,1), tf.argmax(Y,1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
    print "Accuracy: ", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels,dropout_rate:1})


# comment for git push test