# -*- coding: utf-8 -*-
from random import randint
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

import input_data

# Reading data and set variables
mnist = input_data.read_data_sets("/tmp/data/",one_hot=True)

x = tf.placeholder("float",[None,784])
y = tf.placeholder("float",[None,10])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

activation = tf.nn.softmax(tf.matmul(x,W)+b)
learning_rate = 0.2

cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(activation),reduction_indices=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

training_epochs = 25
batch_size = 100
display_step = 1

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(mnist.train.num_examples/batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(optimizer,feed_dict={x: batch_xs,y: batch_ys})
        avg_cost += sess.run(cost,feed_dict={x: batch_xs, y: batch_ys})/total_batch

    if epoch % display_step == 0:
        print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost)

print "Optimization Finished!"


r = randint(0, mnist.test.num_examples -1)
print "Label: ", sess.run(tf.argmax(mnist.test.labels[r:r+1],1))
print "Prediction: ", sess.run(tf.argmax(activation,1),{x: mnist.test.images[r:r+1]})

plt.imshow(mnist.test.images[r:r+1].reshape(28,28),cmap='Greys',interpolation='nearest')
plt.show()


print "Optimization Finished!"

correct_prediction = tf.equal(tf.argmax(activation,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
print "Accuracy: ", accuracy.eval({x: mnist.test.images, y: mnist.test.labels},session=sess)
