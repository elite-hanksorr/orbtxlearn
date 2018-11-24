import functools
from collections import defaultdict, deque
import math
import time
from typing import Generator, List, Tuple, Dict, Union

import tensorflow as tf
import numpy as np

PRE_LSTM_CONV_FEATURES = [10, 15, 20, 25, 32]
CONV_FILTER_SIZE = 5

PRE_LSTM_FC_NODES = [1024]
STATE_SIZE = 256
POST_LSTM_FC_NODES = [64, 16]

def in_variable_scope(*vscope_args, **vscope_kwargs):
    '''Decorate a function with this to wrap the body in tf.variable_scope'''

    def wrapper(f):
        @functools.wraps(f)
        def wrapped(*f_args, **f_kwargs):
            with tf.variable_scope(*vscope_args, **vscope_kwargs):
                return f(*f_args, **f_kwargs)
        return wrapped
    return wrapper

def in_name_scope(*nscope_args, **nscope_kwargs):
    '''Decorate a function with this to wrap the body in tf.name_scope'''

    def wrapper(f):
        @functools.wraps(f)
        def wrapped(*f_args, **f_kwargs):
            with tf.name_scope(*nscope_args, **nscope_kwargs):
                return f(*f_args, **f_kwargs)
        return wrapped
    return wrapper


@in_variable_scope(None, default_name='conv2d_2x2maxpool')
def conv2d_max(images, depth, filter_size=CONV_FILTER_SIZE, strides=1, pooling=2):
    '''Convolution followed by maxpool.

    :param images: Tensor of shape [batch_size, height, width, channels]
    :param depth: Number of feature maps/output channels
    :param filter_size: Convolution filter size (default 5)
    :param strides: Convolution stride size (default 1)
    :param pooling: Maxpool cluster size (default 2)'''

    conv = tf.contrib.layers.conv2d(images, depth, filter_size, activation_fn=tf.nn.leaky_relu)
    pooled = tf.layers.max_pooling2d(conv, pooling, pooling, padding='VALID')
    return pooled

@in_variable_scope('lstm', default_name='lstm', reuse=tf.AUTO_REUSE)
def lstm(layer):
    '''LSTM cell using the efficient tf.contrib.rnn.LSTMBlockFusedCell

    :param layer: A tensor. The size of the LSTM is the size of this tensor'''

    layer = tf.reshape(layer, [1, -1])
    n = layer.shape.as_list()[1]

    cell = tf.contrib.rnn.LSTMBlockCell(1, use_peephole=True)
    c, h = cell.zero_state(1, tf.float32)

    var_c = tf.get_variable('c', initializer=c)
    var_h = tf.get_variable('h', initializer=h)

    output, (new_c, new_h) = cell(layer, (var_c, var_h))
    with tf.control_dependencies([tf.assign(var_c, new_c), tf.assign(var_h, new_h)]):
        return tf.identity(output)

def make_model(batches: int, height: int, width: int, channels: int) \
    -> Tuple[Dict[str, List[tf.Variable]], Dict[str, List[tf.Variable]], Dict[str, List[tf.Variable]]]:

    images = tf.placeholder(tf.float32, [batches, height, width, channels], name='images')
    layer = images
    layers = defaultdict(list)

    for i, n in enumerate(PRE_LSTM_CONV_FEATURES):
        layer = conv2d_max(layer, n)
        # print(f'pre_lstm_conv[{i}]: {layer.shape.as_list()}')
        layers['pre_lstm_conv'].append(layer)

    layer = tf.reshape(layer, [1, -1])
    for i, n in enumerate(PRE_LSTM_FC_NODES + [STATE_SIZE]):
        layer = tf.contrib.layers.fully_connected(layer, n, activation_fn=tf.nn.leaky_relu)
        # print(f'pre_lstm_fc[{i}]: {layer.shape.as_list()}')
        layers['pre_lstm_fc'].append(layer)

    layer = lstm(layer)
    # print(f'state: {layer.shape.as_list()}')
    layers['state'].append(layer)

    layer = tf.reshape(layer, [1, -1])

    for i, n in enumerate(POST_LSTM_FC_NODES):
        layer = tf.contrib.layers.fully_connected(layer, n, activation_fn=tf.nn.leaky_relu)
        layers['post_lstm_fc'].append(layer)
        # print(f'post_lstm_fc[{i+1}]: {layer.shape.as_list()}')

    output = tf.nn.softmax(tf.contrib.layers.fully_connected(layer, 2, activation_fn=tf.nn.softmax))
    output = tf.reshape(output, [-1])
    # print(f'output: {output.shape.as_list()}')

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

def test_model(size: int = 360) -> Union[float, int]:
    tf.reset_default_graph()
    with tf.Session() as sess:
        inputs, outputs, layers = make_model(1, size, size, 3)
        sess.run(tf.global_variables_initializer())

        start = time.time()
        for i in range(10):
            o = sess.run(outputs['output'], feed_dict={
                inputs['images']: np.zeros((1, size, size, 3))
            })
            # print(o)

        delta = time.time() - start
        print(f'10 iterations of {size}x{size}: {delta:.2f}s')

        writer = tf.summary.FileWriter('log', sess.graph)
        writer.close()

        return delta

class Agent():
    def __init__(height: int, width: int, channels: int):
        '''Makes a game agent using make_model().'''

        tf.reset_default_graph()
        self._sess = tf.Session()

        self._inputs, self._outputs, self._layers = make_model(1, height, width, channels)
        self._sess.run(tf.global_variables_initializer())

    def __call__(self, image: np.ndarray) -> bool:
        '''Evaluate the model with a new image.

        :param image: An np.ndarray of shape (height, width, channels)
        :returns: A bool representing whether we should send OrbtXL a keydown (True) or keyup (False)'''

        keydown, keyup = self._sess.run(outputs['output'], feed_dict={
            inputs['images']: image.reshape((1, height, width, channels))
        })

        return keydown >= 0.5

    def close(self):
        '''Close the tf.Session'''

        self._sess.close()


def find_reasonable_image_size() -> None:
    '''Test various image sizes, and choose the largest image size that takes less
    than 100ms to process through our model.'''

    SIZES = [12, 24, 36, 48, 60, 72, 84, 96]
    magnitude = 1
    last_reasonable = None
    while True:
        for n in SIZES:
            n *= magnitude
            if n < 2**len(PRE_LSTM_CONV_FEATURES):
                continue
            delta = test_model(n)

            if delta <= 1.0:
                last_reasonable = n
            elif last_reasonable is not None:
                print(f'Use {last_reasonable}x{last_reasonable}')
                return
            else:
                print(f'No image size large enough for {len(PRE_LST_CONV_FEATURES)} convolutions ran faster than 100ms')
                return

        magnitude *= 10
