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


X = tf.Variable(tf.random_uniform([100,784], -1.0, 1.0))
print X.get_shape()
Y = tf.Variable(tf.random_uniform([100,10], -1.0, 1.0))
print Y.get_shape()
W1 = tf.get_variable("W1",shape=[28*28,512],initializer=xavier_init.init(28*28,512))
W2 = tf.get_variable("W2",shape=[512,256],initializer=xavier_init.init(512,256))
W3 = tf.get_variable("W3",shape=[256,256],initializer=xavier_init.init(256,256))
W4 = tf.get_variable("W4",shape=[512,256],initializer=xavier_init.init(512,256))
W5 = tf.get_variable("W5",shape=[512,10],initializer=xavier_init.init(512,10))
print W1.get_shape()
print W2.get_shape()
print W3.get_shape()
print W4.get_shape()
print W5.get_shape()

B1 = tf.Variable(tf.random_normal([512]))
B2 = tf.Variable(tf.random_normal([256]))
B3 = tf.Variable(tf.random_normal([256]))
B4 = tf.Variable(tf.random_normal([256]))
B5 = tf.Variable(tf.random_normal([10]))


dropout_rate = tf.placeholder("float")
_L1 = tf.nn.relu(tf.add(tf.matmul(X,W1),B1))
L1 = tf.nn.dropout(_L1,dropout_rate)
_L2 = tf.nn.relu(tf.add(tf.matmul(L1,W2),B2))
L2 = tf.nn.dropout(_L1,dropout_rate)
#_L3 = tf.nn.relu(tf.add(tf.matmul(L2,W3),B3))
#L3 = tf.nn.dropout(_L1,dropout_rate)
#_L4 = tf.nn.relu(tf.add(tf.matmul(L3,W4),B4))
#L4 = tf.nn.dropout(_L1,dropout_rate)
#hypothesis = tf.add(tf.matmul(L4,W5),B5)

print L1.get_shape()," ",W2.get_shape()
print (tf.nn.relu(tf.add(tf.matmul(L1,W2),B2))).get_shape(),W3.get_shape()
#print tf.nn.relu((tf.matmul(tf.nn.relu(tf.matmul(L1,W2)+B2),W3))+B3).get_shape()

print L2.get_shape()
print tf.matmul(L2,W3)
#print tf.nn.relu(tf.add(tf.matmul(L2,W3),B3)).get_shape()