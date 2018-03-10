import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

import config as cfg
from provider.data_provider import DataProvider
from yolo_net.yolo_v1 import Yolo


def main(unused_argv):
    data = DataProvider()
    yolo = Yolo()

    global_step = tf.train.create_global_step()
    learning_rate = tf.train.exponential_decay(
        cfg.LEARNING_RATE,
        global_step,
        cfg.DECAY_STEPS,
        cfg.DECAY_RATE,
        cfg.STAIRCASE
    )
    optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=learning_rate
    )
    train_op = slim.learning.create_train_op(
        yolo.loss,
        optimizer,
        global_step
    )

    gpu_options = tf.GPUOptions()
    config = tf.ConfigProto(gpu_options)
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    for iter in range(1, cfg.MAX_ITER + 1):
        images, labels = data.get_data()
        feed_dict = {
            yolo.images: images,
            yolo.labels: labels
        }

        if iter % cfg.SUMMARY_ITER == 0:
            loss, _ = sess.run([yolo.loss, train_op], feed_dict=feed_dict)
            print("Iter: {}, Loss: {}".format(iter, loss))
        else:
            sess.run(train_op, feed_dict=feed_dict)


if __name__ == '__main__':
    tf.app.run()
