
"""
Rewrite of tensorflow example

author: Minu Jeong
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


TRAIN_COUNT = 1000

# load mnist
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# assign variables
x = tf.placeholder(tf.float32, [None, 784])
weight_matrix = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, weight_matrix) + b)

# Ready for training
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.global_variables_initializer()

# init and train
with tf.Session() as sess:
    print("Initializing global variables..")
    sess.run(init)

    print("Training {} times..".format(TRAIN_COUNT))
    for _ in range(TRAIN_COUNT):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={
            x: batch_xs,
            y_: batch_ys
        })

    print("Calculating accuracy")
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    print("PREDICTION", correct_prediction)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # print trained accuracy
    print("Accuracy: {}".format(
        sess.run(accuracy, feed_dict={
            x: mnist.test.images,
            y_: mnist.test.labels
        }))
    )
