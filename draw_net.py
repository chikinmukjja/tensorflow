import tensorflow as tf
import numpy as np


# image(299*299) -> deeplearning -> 256
# product name -> RNN1 -> 256
# brand name -> RNN2 -> 256
# seller mall name -> RNN3 -> 256
# Maker name -> RNN4 -> 256
# price -> embed -> 256
# High-level catagory -> embed -> 256
# Mid-level catagory -> embed -> 256


image = tf.placeholder(tf.float32,[None,299*299],name="Image")
product = tf.placeholder(tf.float32,[None,1],name="Product_name")
brand = tf.placeholder(tf.float32,[None,1],name="Brand_name")
seller = tf.placeholder(tf.float32,[None,1],name="Seller_mall_name")
maker = tf.placeholder(tf.float32,[None,1],name = "Maker_name")
price = tf.placeholder(tf.float32,[None,1],name = "Price")
high_cat = tf.placeholder(tf.float32,[None,1],name ="High-level_catagory")
mid_cat = tf.placeholder(tf.float32,[None,1], name ="Mid-level_catagory")


dW = tf.Variable(tf.zeros([299*299,256]),name="dW")
r1W = tf.Variable(tf.zeros([1,256]),name="r1W")
r2W = tf.Variable(tf.zeros([1,256]),name="r2W")
r3W = tf.Variable(tf.zeros([1,256]),name="r3W")
r4W = tf.Variable(tf.zeros([1,256]),name="r4W")
e1W = tf.Variable(tf.zeros([1,256]),name="e1W")
e2W = tf.Variable(tf.zeros([1,256]),name="e2W")
e3W = tf.Variable(tf.zeros([1,256]),name="e3W")

f1W = tf.Variable(tf.zeros([2048,1024]),name="f1W")
f2W = tf.Variable(tf.zeros([1024,512]),name="f2W")

oW = tf.Variable(tf.zeros([512,4600]),name="oW")
with tf.name_scope("Deeplearing"):
    dl = tf.matmul(image,dW)

with tf.name_scope("RNN1"):
    rnn1 = tf.matmul(product,r1W)

with tf.name_scope("RNN2"):
    rnn2 = tf.matmul(brand,r2W)

with tf.name_scope("RNN3"):
    rnn3 = tf.matmul(seller,r3W)

with tf.name_scope("RNN4"):
    rnn4 = tf.matmul(maker,r4W)

with tf.name_scope("Embed1"):
    e1 = tf.matmul(price,e1W)
with tf.name_scope("Embed2"):
    e2 = tf.matmul(high_cat,e2W)
with tf.name_scope("Embed3"):
    e3 = tf.matmul(mid_cat,e3W)

with tf.name_scope("Concat-layer"):
    concat = tf.concat(1,[dl,rnn1,rnn2,rnn3,rnn4,e1,e2,e3])

with tf.name_scope("FC-layer1"):
    fc1 = tf.matmul(concat,f1W)
with tf.name_scope("FC-layer2"):
    fc2 = tf.matmul(fc1,f2W)

with tf.name_scope("Output"):
    out = tf.matmul(fc2,oW)



init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./logs/cat_logs", sess.graph)





