# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

def inference(images, class_num, boxes_per_cell, cell_num,
              keep_probability, phase_train=True, reuse=None):
    batch_norm_params = {
        # Decay for the moving averages.
        'decay': 0.995,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
        # force in-place updates of mean and variance estimates
        'updates_collections': None,
        # Moving averages ends up in the trainable variables collection
        'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES],
    }

    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                        weights_regularizer=slim.l2_regularizer(0.005),
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        return yolo_v1(images, class_num, boxes_per_cell, cell_num,
                       is_training=phase_train, dropout_keep_prob=keep_probability, reuse=reuse)


def yolo_v1(inputs, class_num, boxes_per_cell, cell_num,
                        is_training=True,
                        dropout_keep_prob=0.8,
                        reuse=None,
                        scope='yolo_v1'):

    with tf.variable_scope(scope, 'yolo_v1', [inputs], reuse=reuse):
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
            with slim.arg_scope([slim.conv2d],
                                stride=1, activation_fn=tf.nn.leaky_relu), slim.arg_scope([slim.max_pool2d], padding='SAME'):
                net = tf.pad(
                    inputs, np.array([[0, 0], [3, 3], [3, 3], [0, 0]]),
                    name='pad_1')
                net = slim.conv2d(net, 64, 7, stride=2, padding='VALID', scope='conv_2')
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_3')
                net = slim.conv2d(net, 192, 3, scope='conv_4')
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_5')
                net = slim.conv2d(net, 128, 1, scope='conv_6')
                net = slim.conv2d(net, 256, 3, scope='conv_7')
                net = slim.conv2d(net, 256, 1, scope='conv_8')
                net = slim.conv2d(net, 512, 3, scope='conv_9')
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_10')
                net = slim.conv2d(net, 256, 1, scope='conv_11')
                net = slim.conv2d(net, 512, 3, scope='conv_12')
                net = slim.conv2d(net, 256, 1, scope='conv_13')
                net = slim.conv2d(net, 512, 3, scope='conv_14')
                net = slim.conv2d(net, 256, 1, scope='conv_15')
                net = slim.conv2d(net, 512, 3, scope='conv_16')
                net = slim.conv2d(net, 256, 1, scope='conv_17')
                net = slim.conv2d(net, 512, 3, scope='conv_18')
                net = slim.conv2d(net, 512, 1, scope='conv_19')
                net = slim.conv2d(net, 1024, 3, scope='conv_20')
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_21')
                net = slim.conv2d(net, 512, 1, scope='conv_22')
                net = slim.conv2d(net, 1024, 3, scope='conv_23')
                net = slim.conv2d(net, 512, 1, scope='conv_24')
                net = slim.conv2d(net, 1024, 3, scope='conv_25')
                net = slim.conv2d(net, 1024, 3, scope='conv_26')
                net = tf.pad(
                    net, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]),
                    name='pad_27')
                net = slim.conv2d(
                    net, 1024, 3, stride=2, padding='VALID', scope='conv_28')
                net = slim.conv2d(net, 1024, 3, scope='conv_29')
                net = slim.conv2d(net, 1024, 3, scope='conv_30')

            net = tf.transpose(net, [0, 3, 1, 2], name='trans_31')
            net = slim.flatten(net, scope="flatten_32")

            with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.leaky_relu,  weights_regularizer=slim.l2_regularizer(0.001)):
                # net = slim.fully_connected(net, 256, scope='fc_33')
                # net = slim.dropout(net, scope="dropout_34", keep_prob=dropout_keep_prob)
                net = slim.fully_connected(net, 4096, scope='fc35')
                net = slim.dropout(net, scope="dropout_36", keep_prob=dropout_keep_prob)
            net = slim.fully_connected(net, cell_num * cell_num*(class_num+5*boxes_per_cell), scope="output_37")
    return net
