import functools
import collections
import logging
import math
import time
from typing import Generator, List, Tuple, Dict, Union, NamedTuple

import tensorflow as tf
import numpy as np

from . import config

__all__ = ['CATEGORIES', '']

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


CATEGORIES = {
    'keydown': 0,
    'keyup': 1
}

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

@in_name_scope('conv2d')
def conv2d(images, filter_size, strides, depth):
    '''Convolutional layer.

    :param images: Tensor of shape [batch_size, height, width, in_channels]
    :param filter_size: Convolution filter size
    :param strides: Convolution stride size
    :param depth: Number of output channels'''

    _, _, _, channels = images.shape.as_list()

    if isinstance(strides, (list, tuple, np.ndarray)):
        strides = [1, strides[0], strides[1], 1]
    else:
        strides = [1, strides, strides, 1]

    filter = tf.Variable(tf.initializers.glorot_normal()([filter_size, filter_size, channels, depth]))
    conv = tf.nn.conv2d(images, filter, strides, padding='SAME')
    return tf.nn.leaky_relu(conv)

@in_variable_scope('lstm', default_name='lstm', reuse=tf.AUTO_REUSE)
def lstm(layer):
    '''LSTM cell using the efficient tf.contrib.rnn.LSTMBlockFusedCell

    :param layer: A tensor. The size of the LSTM is the size of this tensor'''

    layer = tf.reshape(layer, [1, -1])
    n = layer.shape.as_list()[1]

    cell = tf.contrib.rnn.LSTMBlockCell(n, use_peephole=True)
    zero = cell.zero_state(1, tf.float32)
    state_vars = [tf.get_variable(f'lstm_state{i}', initializer=state) for i,state in enumerate(zero)]

    output, new_states = cell(layer, state_vars)
    with tf.control_dependencies([tf.assign(state_var, new_state) for state_var,new_state in zip(state_vars, new_states)]):
        return tf.identity(output)

def make_model(batches: int, height: int, width: int, channels: int) \
    -> Tuple[Dict[str, List[tf.Variable]], Dict[str, List[tf.Variable]], Dict[str, List[tf.Variable]]]:

    images = tf.placeholder(tf.float32, [batches, height, width, channels], name='images')
    layer = images / 256
    layers: Dict[str, List] = collections.defaultdict(list)

    for i, (filter_size, stride, depth) in enumerate(config.params.pre_lstm_conv_layers):
        layer = conv2d(layer, filter_size, stride, depth)
        # print(f'pre_lstm_conv[{i}]: {layer.shape.as_list()}')
        layers['pre_lstm_conv'].append(layer)
        tf.summary.image(f'conv{i}', tf.transpose(layer, [3, 1, 2, 0]), layer.shape.as_list()[3])  # Swap batch_size and channels

    layer = tf.reshape(layer, [batches, -1])
    for i, n in enumerate(config.params.pre_lstm_fc_nodes + [config.params.state_size]):
        layer = tf.layers.dense(layer, n, kernel_initializer=tf.initializers.glorot_normal(), activation=tf.nn.leaky_relu)
        # print(f'pre_lstm_fc[{i}]: {layer.shape.as_list()}')
        layers['pre_lstm_fc'].append(layer)

    #layer = lstm(layer)
    # print(f'state: {layer.shape.as_list()}')
    layers['state'].append(layer)

    layer = tf.reshape(layer, [batches, -1])

    for i, n in enumerate(config.params.post_lstm_fc_nodes):
        layer = tf.layers.dense(layer, n, kernel_initializer=tf.initializers.glorot_normal(), activation=tf.nn.leaky_relu)
        layers['post_lstm_fc'].append(layer)
        # print(f'post_lstm_fc[{i+1}]: {layer.shape.as_list()}')

    logits = tf.identity(tf.layers.dense(layer, 2, kernel_initializer=tf.initializers.glorot_normal(), activation=None), name='logits')
    softmax = tf.nn.softmax(logits)
    action = tf.squeeze(tf.random.multinomial(logits, 1), name='action')

    return \
        ({
            'images': images,
        },
        {
            'logits': logits,
            'softmax': softmax,
            'action': action,
        },
        {
            'pre_lstm_conv': layers['pre_lstm_conv'],
            'pre_lstm_fc': layers['pre_lstm_fc'],
            'state': layers['state'],
            'post_lstm_fc': layers['post_lstm_fc'],
        })

@in_name_scope('train')
def make_optimizer(batches: int, logits_layer):
    actions = tf.placeholder(tf.int32, shape=[batches], name='actions_placeholder')
    rewards = tf.placeholder(tf.float32, shape=[batches], name='rewards_placeholder')

    one_hot = tf.one_hot(tf.reshape(actions, [batches]), 2)

    cross_entropies = tf.losses.softmax_cross_entropy(onehot_labels=one_hot, logits=logits_layer)
    loss = tf.reduce_sum(rewards * cross_entropies)
    adam = tf.train.AdamOptimizer(config.training.learning_rate)
    optimizer = adam.minimize(loss, global_step=tf.train.get_or_create_global_step())

    tf.summary.histogram('cross_entropies', cross_entropies)
    tf.summary.histogram('rewards', rewards)
    tf.summary.scalar('loss', loss)

    return \
        ({
            'actions': actions,
            'rewards': rewards
        },
        {
            'loss': loss,
            'optimizer': optimizer
        },
        {
        })

def test_model(size: int = 480) -> Union[float, int]:
    tf.reset_default_graph()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        inputs, outputs, layers = make_model(1, size, size, 3)

        sess.run(tf.global_variables_initializer())

        o = sess.run(outputs['keydown'], feed_dict={
            inputs['images']: np.zeros((1, size, size, 3))
        })
        log.debug(o)

        return o

def find_reasonable_image_size() -> None:
    '''Test various image sizes, and choose the largest image size that takes less
    than 100ms to process through our model.'''

    SIZES = [12, 24, 36, 48, 60, 72, 84, 96]

    # Force tensorflow to do optimizations now, to not skew test results
    test_model(60)

    magnitude = 1
    last_reasonable = None
    okay = True
    while okay:
        okay = False
        for n in SIZES:
            n *= magnitude
            try:
                delta = test_model(n)
                if delta > 1.0:
                    break
            # If we're using maxpooling, an image that's too small may raise ValueError
            except ValueError:
                continue
            # If the model is too large for memory, then just stop here
            except tf.errors.ResourceExhaustedError:
                break
            else:
                last_reasonable = n
        else:
            okay = True
            magnitude *= 10

    if last_reasonable is not None:
        print(f'Okay to use up to {last_reasonable}x{last_reasonable}')
    else:
        print(f'No image size large enough for {len(config.params.pre_lstm_conv_layers)} convolutions and small enough to fit in memory ran faster than 100ms')
