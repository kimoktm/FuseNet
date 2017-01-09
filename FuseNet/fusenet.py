# ============================================================== #
#                           FuseNet                              #
#                                                                #
#                                                                #
# FuseNet tensorflow implementation WIP                          #
# ============================================================== #

import tensorflow as tf

import utils.layers as layers

# class fusenet:

def build(color_inputs, depth_inputs, num_annots, num_classes, is_training = True, dropout_keep_prob = 0.5):

    # Encoder Section
    # Block 1
    color_conv1_1 = layers.conv_btn(color_inputs,  [3, 3], 64, 'conv1_1', is_training = is_training)
    color_conv1_2 = layers.conv_btn(color_conv1_1, [3, 3], 64, 'conv1_2', is_training = is_training)
    depth_conv1_1 = layers.conv_btn(depth_inputs,  [3, 3], 64, 'convd1_1', is_training = is_training)
    depth_conv1_2 = layers.conv_btn(depth_conv1_1, [3, 3], 64, 'convd1_2', is_training = is_training)
    conv1_fuse    = layers.add(color_conv1_2, depth_conv1_2, 'conv1_fuse')
    color_pool1, color_pool1_arg = layers.argmax_pool(conv1_fuse, [2, 2], 'pool1')
    depth_pool1   = layers.maxpool(depth_conv1_2, [2, 2], 'poold1')

    # Block 2
    color_conv2_1 = layers.conv_btn(color_pool1,   [3, 3], 128, 'conv2_1', is_training = is_training)
    color_conv2_2 = layers.conv_btn(color_conv2_1, [3, 3], 128, 'conv2_2', is_training = is_training)
    depth_conv2_1 = layers.conv_btn(depth_pool1,   [3, 3], 128, 'convd2_1', is_training = is_training)
    depth_conv2_2 = layers.conv_btn(depth_conv2_1, [3, 3], 128, 'convd2_2', is_training = is_training)
    conv2_fuse    = layers.add(color_conv2_2, depth_conv2_2, 'conv2_fuse')
    color_pool2, color_pool2_arg = layers.argmax_pool(conv2_fuse, [2, 2], 'pool2')
    depth_pool2   = layers.maxpool(depth_conv2_2, [2, 2], 'poold2')

    # Block 3
    color_conv3_1 = layers.conv_btn(color_pool2,   [3, 3], 256, 'conv3_1', is_training = is_training)
    color_conv3_2 = layers.conv_btn(color_conv3_1, [3, 3], 256, 'conv3_2', is_training = is_training)
    color_conv3_3 = layers.conv_btn(color_conv3_2, [3, 3], 256, 'conv3_3', is_training = is_training)
    depth_conv3_1 = layers.conv_btn(depth_pool2,   [3, 3], 256, 'convd3_1', is_training = is_training)
    depth_conv3_2 = layers.conv_btn(depth_conv3_1, [3, 3], 256, 'convd3_2', is_training = is_training)
    depth_conv3_3 = layers.conv_btn(depth_conv3_2, [3, 3], 256, 'convd3_3', is_training = is_training)
    conv3_fuse    = layers.add(color_conv3_3, depth_conv3_3, 'conv3_fuse')
    color_pool3, color_pool3_arg = layers.argmax_pool(conv3_fuse, [2, 2], 'pool3')
    color_drop3   = layers.dropout(color_pool3, dropout_keep_prob, 'drop3')
    depth_pool3   = layers.maxpool(depth_conv3_3, [2, 2], 'poold3')
    depth_drop3   = layers.dropout(depth_pool3, dropout_keep_prob, 'dropd3')

    # Block 4
    color_conv4_1 = layers.conv_btn(color_drop3,   [3, 3], 512, 'conv4_1', is_training = is_training)
    color_conv4_2 = layers.conv_btn(color_conv4_1, [3, 3], 512, 'conv4_2', is_training = is_training)
    color_conv4_3 = layers.conv_btn(color_conv4_2, [3, 3], 512, 'conv4_3', is_training = is_training)
    depth_conv4_1 = layers.conv_btn(depth_drop3,   [3, 3], 512, 'convd4_1', is_training = is_training)
    depth_conv4_2 = layers.conv_btn(depth_conv4_1, [3, 3], 512, 'convd4_2', is_training = is_training)
    depth_conv4_3 = layers.conv_btn(depth_conv4_2, [3, 3], 512, 'convd4_3', is_training = is_training)
    conv4_fuse    = layers.add(color_conv4_3, depth_conv4_3, 'conv4_fuse')
    color_pool4, color_pool4_arg = layers.argmax_pool(conv4_fuse, [2, 2], 'pool4')
    color_drop4   = layers.dropout(color_pool4, dropout_keep_prob, 'drop4')
    depth_pool4   = layers.maxpool(depth_conv4_3, [2, 2], 'poold4')
    depth_drop4   = layers.dropout(depth_pool4, dropout_keep_prob, 'dropd4')

    # Block 5
    color_conv5_1 = layers.conv_btn(color_drop4,   [3, 3], 512, 'conv5_1', is_training = is_training)
    color_conv5_2 = layers.conv_btn(color_conv5_1, [3, 3], 512, 'conv5_2', is_training = is_training)
    color_conv5_3 = layers.conv_btn(color_conv5_2, [3, 3], 512, 'conv5_3', is_training = is_training)
    depth_conv5_1 = layers.conv_btn(depth_drop4,   [3, 3], 512, 'convd5_1', is_training = is_training)
    depth_conv5_2 = layers.conv_btn(depth_conv5_1, [3, 3], 512, 'convd5_2', is_training = is_training)
    depth_conv5_3 = layers.conv_btn(depth_conv5_2, [3, 3], 512, 'convd5_3', is_training = is_training)
    conv5_fuse    = layers.add(color_conv5_3, depth_conv5_3, 'conv5_fuse')
    color_pool5, color_pool5_arg = layers.argmax_pool(conv5_fuse, [2, 2], 'pool5')
    color_drop5   = layers.dropout(color_pool5, dropout_keep_prob, 'drop5')

    # Decoder Section
    # Block 1
    unpool5   = layers.unpool_2x2(color_drop5, color_pool5_arg)
    deconv5_3 = layers.deconv_btn(unpool5,   [3, 3], 512, 512, 'deconv5_3', is_training = is_training)
    deconv5_2 = layers.deconv_btn(deconv5_3, [3, 3], 512, 512, 'deconv5_2', is_training = is_training)
    deconv5_1 = layers.deconv_btn(deconv5_2, [3, 3], 512, 512, 'deconv5_1', is_training = is_training)
    decdrop5  = layers.dropout(deconv5_1, dropout_keep_prob, 'decdrop5')

    # Block 2
    unpool4   = layers.unpool_2x2(decdrop5, color_pool4_arg)
    deconv4_3 = layers.deconv_btn(unpool4,   [3, 3], 512, 512, 'deconv4_3', is_training = is_training)
    deconv4_2 = layers.deconv_btn(deconv4_3, [3, 3], 512, 512, 'deconv4_2', is_training = is_training)
    deconv4_1 = layers.deconv_btn(deconv4_2, [3, 3], 512, 256, 'deconv4_1', is_training = is_training)
    decdrop4  = layers.dropout(deconv4_1, dropout_keep_prob, 'decdrop4')

    # Block 3
    unpool3   = layers.unpool_2x2(decdrop4, color_pool3_arg)
    deconv3_3 = layers.deconv_btn(unpool3,   [3, 3], 256, 256, 'deconv3_3', is_training = is_training)
    deconv3_2 = layers.deconv_btn(deconv3_3, [3, 3], 256, 256, 'deconv3_2', is_training = is_training)
    deconv3_1 = layers.deconv_btn(deconv3_2, [3, 3], 256, 128, 'deconv3_1', is_training = is_training)
    decdrop3  = layers.dropout(deconv3_1, dropout_keep_prob, 'decdrop3')

    # Block 4
    unpool2   = layers.unpool_2x2(decdrop3, color_pool2_arg)
    deconv2_2 = layers.deconv_btn(unpool2,   [3, 3], 128, 128, 'deconv2_2', is_training = is_training)
    deconv2_1 = layers.deconv_btn(deconv2_2, [3, 3], 128,  64, 'deconv2_1', is_training = is_training)
    decdrop2  = layers.dropout(deconv2_1, dropout_keep_prob, 'decdrop2')

    # Block 5
    unpool1   = layers.unpool_2x2(decdrop2, color_pool1_arg)
    deconv1_2 = layers.deconv_btn(unpool1, [3, 3], 64, 64, 'deconv1_2', is_training = is_training)
    score     = layers.conv(deconv1_2, [3, 3], num_annots, 'score')
    logits    = tf.reshape(score, (-1, num_annots))

    return logits


def loss(logits, labels):

    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels = labels, logits = logits, name = 'cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name = 'cross_entropy')

    return cross_entropy_mean


def training(loss, learning_rate):
    
    optimizer   = tf.train.GradientDescentOptimizer(learning_rate)
    global_step = tf.Variable(0, name = 'global_step', trainable = False)
    train_op    = optimizer.minimize(loss, global_step = global_step)

    return train_op
