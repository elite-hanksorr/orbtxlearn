import tensorflow as tf
import numpy as np

CONV1_FEATURES = 8
CONV2_FEATURES = 16
CONV3_FEATURES = 32
FILTER_SIZE = 5

HIDDEN1 = 2048
STATE_SIZE = 256
HIDDEN3 = 128
HIDDEN4 = 64

def conv2d_max(images, W, strides=1, pooling=2):
    conv = tf.nn.conv2d(images, W, [1, strides, strides, 1], padding='SAME')
    return tf.layers.max_pooling2d(conv, pooling, pooling, padding='VALID')

def lstm(batches, x, y, channels, state_len):
    state = tf.get_variable('state', [1, state_len])
    images = tf.placeholder(tf.float32, [batches, y, x, channels], 'images')

    k1 = tf.placeholder(tf.float32, [FILTER_SIZE, FILTER_SIZE, channels, CONV1_FEATURES], 'k1')
    conv1 = conv2d_max(images, k1)
    print(conv1.get_shape())

    k2 = tf.placeholder(tf.float32, [FILTER_SIZE, FILTER_SIZE, CONV1_FEATURES, CONV2_FEATURES], 'k2')
    conv2 = conv2d_max(conv1, k2)
    print(conv2.get_shape())

    k3 = tf.placeholder(tf.float32, [FILTER_SIZE, FILTER_SIZE, CONV2_FEATURES, CONV3_FEATURES], 'k3')
    conv3 = conv2d_max(conv2, k3, pooling=4)
    print(conv3.get_shape())

    w1 = tf.placeholder(tf.float32, [CONV3_FEATURES * (x//16) * (y//16), HIDDEN1], 'w1')
    b1 = tf.placeholder(tf.float32, [HIDDEN1], 'b1')
    layer1 = tf.nn.relu(tf.reshape(conv3, [1, -1]) @ w1 + b1)

    w2 = tf.placeholder(tf.float32, [HIDDEN1, STATE_SIZE], 'w2')
    b2 = tf.placeholder(tf.float32, [STATE_SIZE], 'b2')
    layer2 = tf.nn.relu(layer1 @ w2 + b2)

    forget = tf.placeholder(tf.float32, [STATE_SIZE], 'forget')
    assignment = tf.assign(state, (1-forget)*state + forget*layer2)

    w3 = tf.placeholder(tf.float32, [STATE_SIZE, HIDDEN3], 'w3')
    b3 = tf.placeholder(tf.float32, [HIDDEN3], 'b3')
    with tf.control_dependencies([assignment]):
        layer4 = tf.nn.relu(state @ w3 + b3)
        
    w4 = tf.placeholder(tf.float32, [HIDDEN3, HIDDEN4], 'w4')
    b4 = tf.placeholder(tf.float32, [HIDDEN4], 'b4')
    layer5 = tf.nn.relu(layer4 @ w4 + b4)
        
    w5 = tf.placeholder(tf.float32, [HIDDEN4, 2], 'w5')
    b5 = tf.placeholder(tf.float32, [2], 'b5')
    output = tf.nn.relu(layer5 @ w5 + b5)

    return ({
        'state': state,
        'images': images,
        'k1': k1,
        'k2': k2,
        'k3': k3,
        'w1': w1,
        'b1': b1,
        'w2': w2,
        'b2': b2,
        'w3': w3,
        'b3': b3,
        'w4': w4,
        'b4': b4,
        'w5': w5,
        'b5': b5,
    },
    {
        'output': output
    },
    {
        'conv1': conv1,
        'conv2': conv2,
        'conv3': conv3,
        'layer1': layer1,
        'layer2': layer2,
        'state': assignment,
        'layer4': layer4,
        'layer5': layer5
    })

with tf.Session() as sess:
    inputs, outputs, layers = lstm(1, 480, 480, 3, STATE_SIZE)
    sess.run(inputs['state'], feed_dict={
        inputs['images']:np.zeros((1, 480, 480, 3)),
        inputs['state']:np.zeros((1, STATE_SIZE)),
        inputs['k1']:np.zeros((FILTER_SIZE, FILTER_SIZE, 3, CONV1_FEATURES)),
        inputs['k2']:np.zeros((FILTER_SIZE, FILTER_SIZE, CONV1_FEATURES, CONV2_FEATURES)),
        inputs['k3']:np.zeros((FILTER_SIZE, FILTER_SIZE, CONV2_FEATURES, CONV3_FEATURES)),
        inputs['w1']:np.zeros((CONV3_FEATURES * (480//16) * (480//16), HIDDEN1)),
        inputs['b1']:np.zeros((HIDDEN1,)),
        inputs['w2']:np.zeros((HIDDEN1, STATE_SIZE)),
        inputs['b2']:np.zeros((STATE_SIZE,)),
        inputs['w3']:np.zeros((STATE_SIZE, HIDDEN3)),
        inputs['b3']:np.zeros((HIDDEN3,)),
        inputs['w4']:np.zeros((HIDDEN3, HIDDEN4)),
        inputs['b4']:np.zeros((HIDDEN4,)),
        inputs['w5']:np.zeros((HIDDEN4, 2)),
        inputs['b5']:np.zeros((2,)),
        })
    writer = tf.summary.FileWriter('log', sess.graph)
    writer.close()