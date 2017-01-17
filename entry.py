
"""
Rewrite of tensorflow example

author: Minu Jeong
"""

import os
import math
import itertools

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from PIL import Image


TRAIN_COUNT = 10000

# load mnist_dataset
mnist_dataset = input_data.read_data_sets("MNIST_data/", one_hot=True)

# assign variables
weight_matrix = tf.Variable(tf.zeros([784, 10]))
bias = tf.Variable(tf.zeros([10]))

train_image_tensor = tf.placeholder(tf.float32, [None, 784])
train_label_tensor = tf.placeholder(tf.float32, [None, 10])
softmax_tensor = tf.nn.softmax(
    tf.matmul(train_image_tensor, weight_matrix) + bias
)

train_step_operation = \
    tf.train.GradientDescentOptimizer(0.5).minimize(
        tf.reduce_mean(
            -tf.reduce_sum(
                train_label_tensor * tf.log(softmax_tensor),
                axis=[1]
            )))

init_operation = tf.global_variables_initializer()

# init and train
with tf.Session() as sess:
    print("Initializing global variables..")
    sess.run(init_operation)

    print("Training {} times..".format(TRAIN_COUNT))
    for _ in range(TRAIN_COUNT):
        batch_xs, batch_ys = mnist_dataset.train.next_batch(100)
        sess.run(train_step_operation, feed_dict={
            train_image_tensor: batch_xs,
            train_label_tensor: batch_ys
        })

    print("Calculating accuracy")
    correct_prediction_tensor = tf.equal(
        tf.argmax(softmax_tensor, 1),
        tf.argmax(train_label_tensor, 1)
    )

    print("PREDICTION", correct_prediction_tensor)
    accuracy_tensor = tf.reduce_mean(
        tf.cast(
            correct_prediction_tensor,
            tf.float32
        )
    )

    # print trained accuracy_tensor
    print("Accuracy: {}".format(
        sess.run(accuracy_tensor, feed_dict={
            train_image_tensor: mnist_dataset.test.images,
            train_label_tensor: mnist_dataset.test.labels
        }))
    )


def save_data_as_images(dirname, data, maxcount=100):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    for i, imgdat in enumerate(data[:maxcount]):
        w = int(math.sqrt(len(imgdat)))
        outimage = Image.new("L", (w, w))
        px = outimage.load()

        for x, y in itertools.product(range(w), repeat=2):
            data_index = y * w + x

            if data_index not in imgdat:
                continue

            value = imgdat[data_index]
            px[x, y] = 256 - int(255 * value)

        outimage.save("{}/img_{}.png".format(dirname, i))

save_data_as_images("images", mnist_dataset.test.images.tolist())
save_data_as_images("labels", mnist_dataset.test.labels.tolist())
