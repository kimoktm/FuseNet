# ============================================================== #
#                            Layers                              #
#                                                                #
#                                                                #
# Higher level operations for quickly building layers             #
# ============================================================== #

import tensorflow as tf

from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.layers.python.layers import layers as tf_layers


def conv(inputs, kernel_size, num_outputs, name,
        stride_size = [1, 1], padding = 'SAME', activation_fn = tf.nn.relu):
    with tf.variable_scope(name):
        num_filters_in = inputs.get_shape()[-1].value
        kernel_shape   = [kernel_size[0], kernel_size[1], num_filters_in, num_outputs]
        stride_shape   = [1, stride_size[0], stride_size[1], 1]

        weights = tf.get_variable('weights', kernel_shape, tf.float32, xavier_initializer())
        biases  = tf.get_variable('biases', [num_outputs], tf.float32, tf.constant_initializer(0.0))
        conv    = tf.nn.conv2d(inputs, weights, stride_shape, padding = padding)
        outputs = tf.nn.bias_add(conv, biases)

        if activation_fn is not None:
            outputs = activation_fn(outputs)

        return outputs



def conv_btn(inputs, kernel_size, num_outputs, name,
        is_training = True, stride_size = [1, 1], padding = 'SAME', activation_fn = tf.nn.relu):
    with tf.variable_scope(name):
        num_filters_in = inputs.get_shape()[-1].value
        kernel_shape   = [kernel_size[0], kernel_size[1], num_filters_in, num_outputs]
        stride_shape   = [1, stride_size[0], stride_size[1], 1]

        weights = tf.get_variable('weights', kernel_shape, tf.float32, xavier_initializer())
        biases  = tf.get_variable('biases', [num_outputs], tf.float32, tf.constant_initializer(0.0))
        conv    = tf.nn.conv2d(inputs, weights, stride_shape, padding = padding)
        outputs = tf.nn.bias_add(conv, biases)
        outputs = tf_layers.batch_norm(outputs, is_training = is_training)

        if activation_fn is not None:
            outputs = activation_fn(outputs)

        return outputs



def deconv(inputs, kernel_size, num_filters_in, num_outputs, name,
        stride_size = [1, 1], padding = 'SAME', activation_fn = tf.nn.relu):
    with tf.variable_scope(name):
        kernel_shape = [kernel_size[0], kernel_size[1], num_outputs, num_filters_in]
        stride_shape = [1, stride_size[0], stride_size[1], 1]
        input_shape  = tf.shape(inputs)
        output_shape = tf.pack([input_shape[0], input_shape[1], input_shape[2], num_outputs])

        weights = tf.get_variable('weights', kernel_shape, tf.float32, xavier_initializer())
        biases  = tf.get_variable('biases', [num_outputs], tf.float32, tf.constant_initializer(0.0))
        conv_trans = tf.nn.conv2d_transpose(inputs, weights, output_shape, stride_shape, padding = padding)
        outputs    = tf.nn.bias_add(conv_trans, biases)

        if activation_fn is not None:
            outputs = activation_fn(outputs)

        return outputs



def deconv_btn(inputs, kernel_size, num_filters_in, num_outputs, name,
        is_training = True, stride_size = [1, 1], padding = 'SAME', activation_fn = tf.nn.relu):
    with tf.variable_scope(name):
        kernel_shape = [kernel_size[0], kernel_size[1], num_outputs, num_filters_in]
        stride_shape = [1, stride_size[0], stride_size[1], 1]
        input_shape  = tf.shape(inputs)
        output_shape = tf.pack([input_shape[0], input_shape[1], input_shape[2], num_outputs])

        weights = tf.get_variable('weights', kernel_shape, tf.float32, xavier_initializer())
        biases  = tf.get_variable('biases', [num_outputs], tf.float32, tf.constant_initializer(0.0))
        conv_trans = tf.nn.conv2d_transpose(inputs, weights, output_shape, stride_shape, padding = padding)
        outputs    = tf.nn.bias_add(conv_trans, biases)
        outputs    = tf_layers.batch_norm(outputs, is_training = is_training)

        if activation_fn is not None:
            outputs = activation_fn(outputs)

        return outputs



def fully_connected(inputs, num_outputs, name, activation_fn = tf.nn.relu):
    with tf.variable_scope(name):
        num_filters_in = inputs.get_shape()[-1].value

        weights = tf.get_variable('weights', [num_filters_in, num_outputs], tf.float32, xavier_initializer())
        biases  = tf.get_variable('biases', [num_outputs], tf.float32, tf.constant_initializer(0.0))
        outputs = tf.matmul(inputs, weights)
        outputs = tf.nn.bias_add(outputs, biases)

        if activation_fn is not None:
            outputs = activation_fn(outputs)

        return outputs



def add(inputs1, inputs2, name, activation_fn = None):
    with tf.variable_scope(name):
        outputs = tf.add(inputs1, inputs2)

        if activation_fn is not None:
            outputs = activation_fn(outputs)

        return outputs



def maxpool(inputs, kernel_size, name, padding = 'SAME'):
    kernel_shape = [1, kernel_size[0], kernel_size[1], 1]
    
    outputs = tf.nn.max_pool(inputs, ksize = kernel_shape,
            strides = kernel_shape, padding = padding, name = name)

    return outputs



def argmax_pool(inputs, kernel_size, name, padding = 'SAME'):
   # GPU only
    kernel_shape = [1, kernel_size[0], kernel_size[1], 1]
    
    outputs, argmax = tf.nn.max_pool_with_argmax(inputs, ksize = kernel_shape,
                    strides = kernel_shape, padding = padding, name = name)

    return outputs, argmax



def dropout(inputs, keep_prob, name):
    return tf.nn.dropout(inputs, keep_prob = keep_prob, name = name)



def batch_norm(inputs, is_training, name, decay = 0.9997, epsilon = 0.001, activation_fn = None):
    return tf_layers.batch_norm(inputs, name = name, decay = decay,
                            is_training = is_training,
                            epsilon = epsilon, activation_fn = activation_fn)



# https://github.com/tensorflow/tensorflow/issues/2169#issuecomment-253110753
def _unravel_argmax(argmax, shape):
    output_list = []
    output_list.append(argmax // (shape[2] * shape[3]))
    output_list.append(argmax % (shape[2] * shape[3]) // shape[3])

    return tf.pack(output_list)



def unpool_2x2(inputs, raveled_argmax):
    inputs_shape = tf.shape(inputs)
    top_shape = [inputs_shape[0], inputs_shape[1]*2, inputs_shape[2]*2, inputs_shape[3]]

    batch_size = top_shape[0]
    height = top_shape[1]
    width = top_shape[2]
    channels = top_shape[3]

    argmax_shape = tf.to_int64([batch_size, height, width, channels])
    raveled_argmax = _unravel_argmax(raveled_argmax, argmax_shape)

    t1 = tf.to_int64(tf.range(channels))
    t1 = tf.tile(t1, [batch_size * (width//2) * (height//2)])
    t1 = tf.reshape(t1, [-1, channels])
    t1 = tf.transpose(t1, perm = [1, 0])
    t1 = tf.reshape(t1, [channels, batch_size, height//2, width//2, 1])
    t1 = tf.transpose(t1, perm=[1, 0, 2, 3, 4])

    t2 = tf.to_int64(tf.range(batch_size))
    t2 = tf.tile(t2, [channels * (width//2) * (height//2)])
    t2 = tf.reshape(t2, [-1, batch_size])
    t2 = tf.transpose(t2, perm = [1, 0])
    t2 = tf.reshape(t2, [batch_size, channels, height//2, width//2, 1])

    t3 = tf.transpose(raveled_argmax, perm = [1, 4, 2, 3, 0])

    t = tf.concat(4, [t2, t3, t1])
    indices = tf.reshape(t, [(height//2) * (width//2) * channels * batch_size, 4])

    x1 = tf.transpose(inputs, perm = [0, 3, 1, 2])
    values = tf.reshape(x1, [-1])

    delta = tf.SparseTensor(indices, values, tf.to_int64(top_shape))

    return tf.sparse_tensor_to_dense(tf.sparse_reorder(delta))
