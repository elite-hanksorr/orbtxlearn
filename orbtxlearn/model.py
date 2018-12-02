import functools
import collections
import logging
import math
import time
from typing import Generator, List, Tuple, Dict, Union, NamedTuple, Any, Optional, Callable

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

INITIALIZER = tf.initializers.glorot_normal()

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

@in_name_scope('fully_connected')
def fully_connected(layer, size: int, activation_fn: Optional[Callable] = tf.nn.relu):
    '''
    Fully connected layer

    :param layer: Input tensor of shape [batch_size, input_units]
    :param size: Number of output units
    :param activation_fn: Optional activation function
    '''

    kernel = tf.Variable(INITIALIZER([layer.shape.as_list()[1], size]), name='kernel')
    bias = tf.Variable(INITIALIZER([size]), name='bias')

    tf.summary.histogram('kernel', kernel, family='parameters')
    tf.summary.histogram('bias', bias, family='parameters')

    layer = layer @ kernel + bias
    if activation_fn is not None:
        layer = activation_fn(layer)

    return layer


@in_name_scope('conv2d')
def conv2d(images, filter_size: int, strides: Union[int, List], depth: int, padding: str):
    '''
    Convolutional layer.

    :param images: Tensor of shape [batch_size, height, width, in_channels]
    :param filter_size: Convolution filter size
    :param strides: Convolution stride size
    :param depth: Number of output channels
    :param padding: Which padding to use. 'SAME' or 'VALID'
    '''

    channels = images.shape.as_list()[3]

    if isinstance(strides, (list, tuple, np.ndarray)):
        strides = [1, strides[0], strides[1], 1]
    else:
        strides = [1, strides, strides, 1]

    filter = tf.Variable(INITIALIZER([filter_size, filter_size, channels, depth]), name='filter')
    biases = tf.Variable(INITIALIZER([depth]), name='biases')
    conv = tf.nn.conv2d(images, filter, strides, padding=padding) + biases
    relu = tf.nn.relu(conv)

    filter_images = tf.reshape(tf.transpose(filter, [2, 3, 0, 1]), [-1, filter_size, filter_size, 1])
    tf.summary.image('filter_image', filter_images, max_outputs=channels*depth, family='parameters')
    tf.summary.histogram('filter', filter, family='parameters')
    tf.summary.histogram('biases', biases, family='parameters')

    return relu

@in_variable_scope('lstm', default_name='lstm', reuse=tf.AUTO_REUSE)
def lstm(layer) -> Any:
    '''LSTM cell using the efficient tf.contrib.rnn.LSTMBlockFusedCell

    :param layer: A tensor. The size of the LSTM is the size of this tensor'''

    layer = tf.expand_dims(layer, 0)
    n = layer.shape.as_list()[1]

    cell = tf.contrib.rnn.LSTMBlockCell(n, use_peephole=True)
    zero = cell.zero_state(1, tf.float32)
    state_vars = [tf.get_variable(f'lstm_state{i}', initializer=state) for i,state in enumerate(zero)]

    output, new_states = cell(layer, state_vars)
    with tf.control_dependencies([tf.assign(state_var, new_state) for state_var,new_state in zip(state_vars, new_states)]):
        return tf.identity(output)

@in_variable_scope('model')
def make_model(height: int, width: int, channels: int) \
    -> Tuple[Dict[str, List[tf.Variable]], Dict[str, List[tf.Variable]], Dict[str, List[tf.Variable]]]:
    print('Building model...')

    images = tf.placeholder(tf.float32, [None, height, width, channels], name='images')
    layer = images / 256
    layers: Dict[str, List] = collections.defaultdict(list)

    for i, (filter_size, stride, depth, padding) in enumerate(config.params.pre_lstm_conv_layers):
        layer = conv2d(layer, filter_size, stride, depth, padding)
        # print(f'pre_lstm_conv[{i}]: {layer.shape.as_list()}')
        layers['pre_lstm_conv'].append(layer)

    layer = tf.reshape(layer, [-1, np.prod(layer.shape.as_list()[1:])])
    for i, n in enumerate(config.params.pre_lstm_fc_nodes + [config.params.state_size]):
        layer = fully_connected(layer, n, activation_fn=tf.nn.relu)
        # print(f'pre_lstm_fc[{i}]: {layer.shape.as_list()}')
        layers['pre_lstm_fc'].append(layer)

    #layer = lstm(layer)
    #layer = tf.reshape(layer, [batches, -1])
    # print(f'state: {layer.shape.as_list()}')
    #layers['state'].append(layer)
    #tf.summary.histogram(f'state{i}', layer, family='layers)

    for i, n in enumerate(config.params.post_lstm_fc_nodes):
        layer = fully_connected(layer, 2, activation_fn=tf.nn.relu)
        # print(f'post_lstm_fc[{i+1}]: {layer.shape.as_list()}')
        layers['post_lstm_fc'].append(layer)

    # logits = tf.identity(tf.layers.dense(layer, 2, kernel_initializer=tf.initializers.glorot_normal(), activation=None), name='logits')
    logits = tf.identity(fully_connected(layer, 2, activation_fn=None), name='logits')
    softmax = tf.nn.softmax(logits)
    action = tf.squeeze(tf.random.multinomial(logits, 1), name='action')

    with tf.variable_scope('summary'):
        tf.summary.image('image', tf.expand_dims(images[0], 0), max_outputs=1, family='inputs')

        with tf.variable_scope('conv'):
            for i, conv in enumerate(layers['pre_lstm_conv']):
                transposed = tf.transpose(conv[0:1], [3, 1, 2, 0])  # batch <- channels, channels <- 1
                tf.summary.image(f'conv{i}', transposed, transposed.shape.as_list()[0], family='layers')

        for layer_type in ['pre_lstm_fc', 'post_lstm_fc']:
            for i, l in enumerate(layers[layer_type]):
                tf.summary.histogram(f'{layer_type}_{i}', tf.identity(l), family='layers')

        tf.summary.scalar('keydown_logit', logits[0,0], family='outputs')
        tf.summary.scalar('keyup_logit', logits[0,1], family='outputs')
        tf.summary.scalar('keydown_softmax', softmax[0,0], family='outputs')
        tf.summary.scalar('action', action[0], family='outputs')

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
            #'pre_lstm_fc': layers['pre_lstm_fc'],
            #'state': layers['state'],
            'post_lstm_fc': layers['post_lstm_fc'],
        })

@in_name_scope('train')
def make_optimizer(logits_layer):
    print('Building optimizer...')

    actions = tf.placeholder(tf.int32, shape=[None], name='actions_placeholder')
    rewards = tf.placeholder(tf.float32, shape=[None], name='rewards_placeholder')

    one_hot = tf.one_hot(actions, 2)
    assigned_rewards = one_hot * tf.expand_dims(rewards, -1) + (1-one_hot) * config.params.rewards['nothing']
    loss = tf.reduce_mean((assigned_rewards - logits_layer)**2)

    # cross_entropies = tf.losses.softmax_cross_entropy(onehot_labels=one_hot, logits=logits_layer)
    # loss = tf.reduce_sum(rewards * cross_entropies)
    optimizer = tf.train.RMSPropOptimizer(config.training.learning_rate) \
        .minimize(loss, global_step=tf.train.get_or_create_global_step())

    with tf.name_scope('summary'):
        summary = tf.summary.merge([
            # tf.summary.histogram('cross_entropies', cross_entropies, family='train'),
            tf.summary.histogram('rewards', rewards[0], family='train'),
            tf.summary.scalar('loss', tf.identity(loss), family='train'),
        ])

    return \
        ({
            'actions': actions,
            'rewards': rewards
        },
        {
            'loss': loss,
            'optimizer': optimizer,
            'summary': summary,
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
