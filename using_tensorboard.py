# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

# 주의사항 tensorboard --logdir=절대경로 로 주어야 정상적으로 동작한다.
#        scalar summary 가 안나오는 문제가 있는데 사파리환경에서 발생하는 문제임 -> chrome 으로 바꾸고 정상적으로 해결
xy = np.loadtxt('xor_train.txt',unpack=True,dtype='float32')
# reshape를 해주어야함
x_data = np.transpose(xy[0:-1])
y_data = np.reshape(xy[-1],(4,1))

X = tf.placeholder(tf.float32,name='X-input')
Y = tf.placeholder(tf.float32,name='Y-input')

W1 = tf.Variable(tf.random_uniform([2,2],-1.0,1.0),name='weight1')
W2 = tf.Variable(tf.random_uniform([2,1],-1.0,1.0),name='weight2')

b1 = tf.Variable(tf.zeros([2]),name='bias1')
b2 = tf.Variable(tf.zeros([1]),name='bias2')

# our hypothesis
with tf.name_scope('layer2'):
    L2 = tf.sigmoid(tf.matmul(X,W1)+b1)

with tf.name_scope('layer3'):
    hypothesis = tf.sigmoid(tf.matmul(L2,W2)+b2)

# cost function
with tf.name_scope('cost'):
    cost = -tf.reduce_mean(Y*tf.log(hypothesis)+(1-Y)*tf.log(1-hypothesis))

cost_sum = tf.summary.scalar('cost',cost)

with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(cost)

w1_hist = tf.summary.histogram('weight1',W1)
w2_hist = tf.summary.histogram('weight2',W2)

bi_hist = tf.summary.histogram('bias1',b1)
b2_hist = tf.summary.histogram('bias2',b2)

y_hist = tf.summary.histogram('y',Y)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #tensorboard --logdir=./logs/xor_logs
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./logs/xor_logs",sess.graph)

# Fit the line.
    for step in xrange(20001):
        if step % 1000 == 0:
            print "step ",step
        summary, _ = sess.run([merged, train], feed_dict={X:x_data, Y:y_data})
        writer.add_summary(summary,step)
        writer.flush()

# Fit the line.
#for step in xrange(20000):
#    sess.run(train,feed_dict={X:x_data, Y:y_data})
#    if step % 2000 == 0:
#        summary = sess.run(merged,feed_dict={X:x_data,Y:y_data})
#        writer.add_summary(summary,step)


print 'learning over'
import time
