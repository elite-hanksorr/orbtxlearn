import tensorflow as tf
import numpy as np

CONV1_FEATURES = 8
CONV2_FEATURES = 16
CONV3_FEATURES = 32
FILTER_SIZE = 5

HIDDEN1 = 2048

STATE_SIZE = 256

def conv2d_max(images, W, strides=1, pooling=2):
    conv = tf.nn.conv2d(images, W, [1, strides, strides, 1], padding='SAME')
    return tf.layers.max_pooling2d(conv, pooling, pooling, padding='VALID')

def lstm(batches, x, y, channels, state_len):
    state = tf.placeholder(tf.float32, [state_len], 'state')
    images = tf.placeholder(tf.float32, [batches, y, x, channels], 'images')

    k1 = tf.placeholder(tf.float32, [FILTER_SIZE, FILTER_SIZE, channels, CONV1_FEATURES])
    conv1 = conv2d_max(images, k1)

    k2 = tf.placeholder(tf.float32, [FILTER_SIZE, FILTER_SIZE, CONV1_FEATURES, CONV2_FEATURES])
    conv2 = conv2d_max(conv1, k2)

    k3 = tf.placeholder(tf.float32, [FILTER_SIZE, FILTER_SIZE, CONV2_FEATURES, CONV3_FEATURES])
    conv2 = conv2d_max(conv2, k3, pooling=4)

    w1 = tf.placeholder(tf.float32, [])

    return ({
        'state': state,
        'images': images,
        'k1': k1,
        'k2': k2
    },
    {
        'output': conv2
    },
    {
        'conv1': conv1,
        'conv2': conv2
    })

with tf.Session() as sess:
    inputs, outputs, layers = lstm(1, 480, 480, 3, STATE_SIZE)
    sess.run(inputs['state'], feed_dict={
        inputs['images']:np.zeros((1, 480, 480, 3)),
        inputs['state']:np.zeros(STATE_SIZE),
        inputs['k1']:np.zeros((FILTER_SIZE, FILTER_SIZE, 3, CONV1_FEATURES)),
        inputs['k2']:np.zeros((FILTER_SIZE, FILTER_SIZE, CONV1_FEATURES, CONV2_FEATURES)),
        })
    writer = tf.summary.FileWriter('log', sess.graph)
    writer.close()