# -*- coding: utf-8 -*-
import tensorflow as tf


x_data = [1, 2, 3]
y_data = [1, 2, 3]

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")


#hypothesis = W * X + b
hypothesis = tf.add(tf.mul(W,X),b)


cost = tf.reduce_mean(tf.square(hypothesis - Y))
# 텐서플로우에 기본적으로 포함되어 있는 함수를 이용해 경사 하강법 최적화를 수행합니다.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
# 비용을 최소화 하는 것이 최종 목표
train_op = optimizer.minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in xrange(2001):
        sess.run([train_op,cost],feed_dict={X: x_data, Y: y_data})
        if step % 20 == 0:
            print step, sess.run(cost,feed_dict={X: x_data,Y: y_data}), sess.run(W), sess.run(b)

    print sess.run(hypothesis,feed_dict={X: 5.})
    print sess.run(hypothesis,feed_dict={X: 1.5})


# why bias is chaged
# why somtimes no converges