# -*- coding: utf-8 -*-
import tensorflow as tf

#tf.constant
hello = tf.constant("hello tensorflow")

a = tf.constant(10)
b = tf.constant(22)

c = a + b

# tf.placeholder 입력값을 받을 변수
# 함수를 사용할 때 파라미터
# 모델에 대해서 함수에 파라미터를 주는 것 처럼 실행 때 변수를 주겠다.

X = tf.placeholder("float",[None,3])

# tf.Variable: 그래프를 계산하면서 최적화 할 변수들입니다. 이 값이 바로 신경망을 좌우하는 값들입니다.
# tf.random_normal: 각 변수들의 초기값을 정규분포 랜덤 값으로 초기화합니다.
# name: 나중에 텐서보드등으로 값의 변화를 추적하거나 살펴보기 쉽게 하기 위해 이름을 붙여줍니다.
W = tf.Variable(tf.random_normal([3, 2]), name='Weights')
b = tf.Variable(tf.random_normal([2, 1]), name='Bias')

train = [[1,2,3], [5,6,7]]
# 2*3 * 3*2 + 2*1

expr = tf.matmul(X,W) + b

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print sess.run(tf.matmul(X,W),feed_dict={X: train})
print sess.run(b)
print sess.run(expr,feed_dict={X: train})

