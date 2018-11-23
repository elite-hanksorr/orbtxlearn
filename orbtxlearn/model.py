import functools
from collections import defaultdict
import math

import tensorflow as tf
import numpy as np

PRE_LSTM_CONV_FEATURES = [10, 15, 20, 25, 32]
CONV_FILTER_SIZE = 5

PRE_LSTM_FC_NODES = [1024]
STATE_SIZE = 256
POST_LSTM_FC_NODES = [64, 16]

def in_variable_scope(*vscope_args, **vscope_kwargs):
    def wrapper(f):
        @functools.wraps(f)
        def wrapped(*f_args, **f_kwargs):
            with tf.variable_scope(*vscope_args, **vscope_kwargs):
                return f(*f_args, **f_kwargs)
        return wrapped
    return wrapper

def in_name_scope(*nscope_args, **nscope_kwargs):
    def wrapper(f):
        @functools.wraps(f)
        def wrapped(*f_args, **f_kwargs):
            with tf.name_scope(*nscope_args, **nscope_kwargs):
                return f(*f_args, **f_kwargs)
        return wrapped
    return wrapper


@in_variable_scope(None, default_name='5x5conv2d_2x2maxpool')
def conv2d_max(images, depth, strides=1, pooling=2):
    conv = tf.contrib.layers.conv2d(images, depth, CONV_FILTER_SIZE, activation_fn=tf.nn.leaky_relu)
    pooled = tf.layers.max_pooling2d(conv, pooling, pooling, padding='VALID')
    return pooled

@in_variable_scope('lstm', default_name='lstm', reuse=tf.AUTO_REUSE)
def lstm(layer, state_size):
    batches, _ = layer.shape.as_list()
    # layer = tf.reshape(layer, [-1])

    cell = tf.contrib.rnn.LSTMBlockCell(state_size, use_peephole=True)
    c, h = cell.zero_state(batches, tf.float32)

    var_c = tf.get_variable('c', initializer=c)
    var_h = tf.get_variable('h', initializer=h)

    output, (new_c, new_h) = cell(layer, (var_c, var_h))
    with tf.control_dependencies([tf.assign(var_c, new_c), tf.assign(var_h, new_h)]):
        return tf.identity(output)

def make_model(batches, x, y, channels, state_size):
    images = tf.placeholder(tf.float32, [batches, y, x, channels], name='images')
    layer = images
    layers = defaultdict(list)

    for i, n in enumerate(PRE_LSTM_CONV_FEATURES):
        layer = conv2d_max(layer, n)
        print(f'pre_lstm_conv[{i}]: {layer.shape.as_list()}')
        layers['pre_lstm_conv'].append(layer)

    layer = tf.reshape(layer, [1, -1])
    for i, n in enumerate(PRE_LSTM_FC_NODES + [STATE_SIZE]):
        layer = tf.contrib.layers.fully_connected(layer, n, activation_fn=tf.nn.leaky_relu)
        print(f'pre_lstm_fc[{i}]: {layer.shape.as_list()}')
        layers['pre_lstm_fc'].append(layer)

    layer = lstm(layer, STATE_SIZE)
    print(f'state: {layer.shape.as_list()}')
    layers['state'].append(layer)

    for i, n in enumerate(POST_LSTM_FC_NODES):
        layer = tf.contrib.layers.fully_connected(layer, n, activation_fn=tf.nn.leaky_relu)
        layers['post_lstm_fc'].append(layer)
        print(f'post_lstm_fc[{i+1}]: {layer.shape.as_list()}')

    output = tf.nn.softmax(tf.contrib.layers.fully_connected(layer, 2, activation_fn=tf.nn.softmax))
    print(f'output: {output.shape.as_list()}')

    return \
        ({
            'images': images,
        },
        {
            'output': output,
        },
        {
            'pre_lstm_conv': layers['pre_lstm_conv'],
            'pre_lstm_fc': layers['pre_lstm_fc'],
            'state': layers['state'],
            'post_lstm_fc': layers['post_lstm_fc'],
        })

tf.reset_default_graph()
with tf.Session() as sess:
    inputs, outputs, layers = make_model(1, 480, 480, 3, STATE_SIZE)
    sess.run(tf.global_variables_initializer())
    sess.run(outputs['output'], feed_dict={
        inputs['images']: np.zeros((1, 480, 480, 3))
    })
    sess.run(outputs['output'], feed_dict={
        inputs['images']: np.zeros((1, 480, 480, 3))
    })
    writer = tf.summary.FileWriter('log', sess.graph)
    writer.close()
