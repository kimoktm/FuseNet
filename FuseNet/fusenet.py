# ============================================================== #
#                           FuseNet                              #
#                                                                #
#                                                                #
# FuseNet tensorflow implementation WIP                          #
# ============================================================== #

import tensorflow as tf

import utils.layers as layers


def build(color_inputs, depth_inputs, num_annots, is_training):
    """
    Build fusenet network:
    ----------
    Args:
        color_inputs: Tensor, [batch_size, height, width, 3]
        depth_inputs: Tensor, [batch_size, height, width, 1]
        num_annots: Integer, number of segmentation (annotation) labels
        is_training: Boolean, in training mode or not (for dropout & bn)
    Returns:
        annot_logits: Tensor, predicted annotated image flattened 
                              [batch_size * height * width,  num_annots]
    """

    dropout_keep_prob = tf.select(is_training, 0.5, 1.0)

    # Encoder Section
    # Block 1
    color_conv1_1 = layers.conv_btn(color_inputs,  [3, 3], 64, 'conv1_1', is_training = is_training)
    color_conv1_2 = layers.conv_btn(color_conv1_1, [3, 3], 64, 'conv1_2', is_training = is_training)
    depth_conv1_1 = layers.conv_btn(depth_inputs,  [3, 3], 64, 'd_conv1_1', is_training = is_training)
    depth_conv1_2 = layers.conv_btn(depth_conv1_1, [3, 3], 64, 'd_conv1_2', is_training = is_training)
    conv1_fuse    = layers.add(color_conv1_2, depth_conv1_2, 'conv1_fuse')
    color_pool1   = layers.maxpool(conv1_fuse, [2, 2], 'pool1')
    depth_pool1   = layers.maxpool(depth_conv1_2, [2, 2], 'd_pool1')

    # Block 2
    color_conv2_1 = layers.conv_btn(color_pool1,   [3, 3], 128, 'conv2_1', is_training = is_training)
    color_conv2_2 = layers.conv_btn(color_conv2_1, [3, 3], 128, 'conv2_2', is_training = is_training)
    depth_conv2_1 = layers.conv_btn(depth_pool1,   [3, 3], 128, 'd_conv2_1', is_training = is_training)
    depth_conv2_2 = layers.conv_btn(depth_conv2_1, [3, 3], 128, 'd_conv2_2', is_training = is_training)
    conv2_fuse    = layers.add(color_conv2_2, depth_conv2_2, 'conv2_fuse')
    color_pool2   = layers.maxpool(conv2_fuse, [2, 2], 'pool2')
    depth_pool2   = layers.maxpool(depth_conv2_2, [2, 2], 'd_pool2')

    # Block 3
    color_conv3_1 = layers.conv_btn(color_pool2,   [3, 3], 256, 'conv3_1', is_training = is_training)
    color_conv3_2 = layers.conv_btn(color_conv3_1, [3, 3], 256, 'conv3_2', is_training = is_training)
    color_conv3_3 = layers.conv_btn(color_conv3_2, [3, 3], 256, 'conv3_3', is_training = is_training)
    depth_conv3_1 = layers.conv_btn(depth_pool2,   [3, 3], 256, 'd_conv3_1', is_training = is_training)
    depth_conv3_2 = layers.conv_btn(depth_conv3_1, [3, 3], 256, 'd_conv3_2', is_training = is_training)
    depth_conv3_3 = layers.conv_btn(depth_conv3_2, [3, 3], 256, 'd_conv3_3', is_training = is_training)
    conv3_fuse    = layers.add(color_conv3_3, depth_conv3_3, 'conv3_fuse')
    color_pool3   = layers.maxpool(conv3_fuse, [2, 2], 'pool3')
    color_drop3   = layers.dropout(color_pool3, dropout_keep_prob, 'drop3')
    depth_pool3   = layers.maxpool(depth_conv3_3, [2, 2], 'd_pool3')
    depth_drop3   = layers.dropout(depth_pool3, dropout_keep_prob, 'd_drop3')

    # Block 4
    color_conv4_1 = layers.conv_btn(color_drop3,   [3, 3], 512, 'conv4_1', is_training = is_training)
    color_conv4_2 = layers.conv_btn(color_conv4_1, [3, 3], 512, 'conv4_2', is_training = is_training)
    color_conv4_3 = layers.conv_btn(color_conv4_2, [3, 3], 512, 'conv4_3', is_training = is_training)
    depth_conv4_1 = layers.conv_btn(depth_drop3,   [3, 3], 512, 'd_conv4_1', is_training = is_training)
    depth_conv4_2 = layers.conv_btn(depth_conv4_1, [3, 3], 512, 'd_conv4_2', is_training = is_training)
    depth_conv4_3 = layers.conv_btn(depth_conv4_2, [3, 3], 512, 'd_conv4_3', is_training = is_training)
    conv4_fuse    = layers.add(color_conv4_3, depth_conv4_3, 'conv4_fuse')
    color_pool4   = layers.maxpool(conv4_fuse, [2, 2], 'pool4')
    color_drop4   = layers.dropout(color_pool4, dropout_keep_prob, 'drop4')
    depth_pool4   = layers.maxpool(depth_conv4_3, [2, 2], 'd_pool4')
    depth_drop4   = layers.dropout(depth_pool4, dropout_keep_prob, 'd_drop4')

    # Block 5
    color_conv5_1 = layers.conv_btn(color_drop4,   [3, 3], 512, 'conv5_1', is_training = is_training)
    color_conv5_2 = layers.conv_btn(color_conv5_1, [3, 3], 512, 'conv5_2', is_training = is_training)
    color_conv5_3 = layers.conv_btn(color_conv5_2, [3, 3], 512, 'conv5_3', is_training = is_training)
    depth_conv5_1 = layers.conv_btn(depth_drop4,   [3, 3], 512, 'd_conv5_1', is_training = is_training)
    depth_conv5_2 = layers.conv_btn(depth_conv5_1, [3, 3], 512, 'd_conv5_2', is_training = is_training)
    depth_conv5_3 = layers.conv_btn(depth_conv5_2, [3, 3], 512, 'd_conv5_3', is_training = is_training)
    conv5_fuse    = layers.add(color_conv5_3, depth_conv5_3, 'conv5_fuse')
    color_pool5   = layers.maxpool(conv5_fuse, [2, 2], 'pool5')
    color_drop5   = layers.dropout(color_pool5, dropout_keep_prob, 'drop5')

    # Decoder Section
    # Block 1
    upsample5 = layers.deconv_upsample(color_drop5, 2, 'upsample5')
    deconv5_3 = layers.deconv_btn(upsample5, [3, 3], 512, 512, 'deconv5_3', is_training = is_training)
    deconv5_2 = layers.deconv_btn(deconv5_3, [3, 3], 512, 512, 'deconv5_2', is_training = is_training)
    deconv5_1 = layers.deconv_btn(deconv5_2, [3, 3], 512, 512, 'deconv5_1', is_training = is_training)
    decdrop5  = layers.dropout(deconv5_1, dropout_keep_prob, 'decdrop5')

    # Block 2
    upsample4 = layers.deconv_upsample(decdrop5, 2, 'upsample4')
    deconv4_3 = layers.deconv_btn(upsample4, [3, 3], 512, 512, 'deconv4_3', is_training = is_training)
    deconv4_2 = layers.deconv_btn(deconv4_3, [3, 3], 512, 512, 'deconv4_2', is_training = is_training)
    deconv4_1 = layers.deconv_btn(deconv4_2, [3, 3], 512, 256, 'deconv4_1', is_training = is_training)
    decdrop4  = layers.dropout(deconv4_1, dropout_keep_prob, 'decdrop4')

    # Block 3
    upsample3 = layers.deconv_upsample(decdrop4, 2, 'upsample3')
    deconv3_3 = layers.deconv_btn(upsample3, [3, 3], 256, 256, 'deconv3_3', is_training = is_training)
    deconv3_2 = layers.deconv_btn(deconv3_3, [3, 3], 256, 256, 'deconv3_2', is_training = is_training)
    deconv3_1 = layers.deconv_btn(deconv3_2, [3, 3], 256, 128, 'deconv3_1', is_training = is_training)
    decdrop3  = layers.dropout(deconv3_1, dropout_keep_prob, 'decdrop3')

    # Block 4
    upsample2 = layers.deconv_upsample(decdrop3, 2, 'upsample2')
    deconv2_2 = layers.deconv_btn(upsample2, [3, 3], 128, 128, 'deconv2_2', is_training = is_training)
    deconv2_1 = layers.deconv_btn(deconv2_2, [3, 3], 128,  64, 'deconv2_1', is_training = is_training)
    decdrop2  = layers.dropout(deconv2_1, dropout_keep_prob, 'decdrop2')

    # Block 5
    upsample1    = layers.deconv_upsample(decdrop2, 2, 'upsample1')
    deconv1_2    = layers.deconv_btn(upsample1, [3, 3], 64, 64, 'deconv1_2', is_training = is_training)
    annot_score  = layers.conv(deconv1_2, [3, 3], num_annots, 'score')
    annot_logits = tf.reshape(annot_score, (-1, num_annots))

    return annot_logits


def segmentation_loss(logits, labels, class_weights = None):
    """
    Segmentation loss:
    ----------
    Args:
        logits: Tensor, predicted    [batch_size * height * width,  num_annots]
        labels: Tensor, ground truth [batch_size * height * width, num_annots]
        class_weights: Tensor, weighting of class for loss [num_annots, 1] or None

    Returns:
        segment_loss: Segmentation loss
    """

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                        labels = labels, logits = logits, name = 'segment_cross_entropy_per_example')

    if class_weights is not None:
        weights = tf.matmul(labels, class_weights, a_is_sparse=True)
        weights = tf.reshape(weights, [-1])
        cross_entropy = cross_entropy * weights

    segment_loss  = tf.reduce_mean(cross_entropy, name = 'segment_cross_entropy')

    tf.summary.scalar("loss/segmentation", segment_loss)

    return segment_loss


def l2_loss():
    """
    L2 loss:
    -------
    Returns:
        l2_loss: L2 loss for all weights
    """
    
    weights = [var for var in tf.trainable_variables() if var.name.endswith('weights:0')]
    l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in weights])

    tf.summary.scalar("loss/weights", l2_loss)

    return l2_loss



def loss(annot_logits, annots, weight_decay_factor, class_weights = None):
    """
    Total loss:
    ----------
    Args:
        annot_logits: Tensor, predicted    [batch_size * height * width,  num_annots]
        annots: Tensor, ground truth [batch_size, height, width, 1]
        weight_decay_factor: float, factor with which weights are decayed
        class_weights: Tensor, weighting of class for loss [num_annots, 1] or None

    Returns:
        total_loss: Segmentation + WeightDecayFactor * L2 loss
    """

    segment_loss = segmentation_loss(annot_logits, annots, class_weights)
    total_loss   = segment_loss + weight_decay_factor * l2_loss()

    tf.summary.scalar("loss/total", total_loss)

    return total_loss


def segmentation_accuracy(logits, labels):
    """
    Segmentation accuracy:
    ----------
    Args:
        logits: Tensor, predicted    [batch_size * height * width,  num_annots]
        labels: Tensor, ground truth [batch_size, height, width, 1]

    Returns:
        segmentation_accuracy: Segmentation accuracy
    """

    labels = tf.to_int64(labels)
    labels = tf.reshape(labels, [-1, 1])
    predicted_annots = tf.reshape(tf.argmax(logits, axis=1), [-1, 1])
    correct_predictions = tf.equal(predicted_annots, labels)
    segmentation_accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    tf.summary.scalar("accuarcy/segmentation", segmentation_accuracy)

    return segmentation_accuracy


def segmentation_metrics(logits, labels):
    """
    Segmentation metrics:
    ----------
    Args:
        logits: Tensor, predicted    [batch_size * height * width,  num_annots]
        labels: Tensor, ground truth [batch_size, height, width, 1]

    Returns:
        true positives, false positives, true negatives, false negatives
    """

    num_annots = logits.get_shape()[1]
    labels = tf.reshape(labels, [-1])
    labels = tf.cast(labels, tf.int64)
    labels = tf.one_hot(labels, depth=num_annots)

    predicted_annots = tf.argmax(logits, axis=1)
    predicted_annots = tf.one_hot(predicted_annots, depth=num_annots)

    true_positives = tf.logical_and(tf.equal(labels, 1), tf.equal(predicted_annots, 1))
    false_positives = tf.logical_and(tf.equal(labels, 0), tf.equal(predicted_annots, 1))
    true_negatives = tf.logical_and(tf.equal(labels, 0), tf.equal(predicted_annots, 0))
    false_negatives = tf.logical_and(tf.equal(labels, 1), tf.equal(predicted_annots, 0))
    
    true_positives_count = tf.reduce_sum(tf.cast(true_positives, tf.int64), axis=0)
    false_positives_count = tf.reduce_sum(tf.cast(false_positives, tf.int64), axis=0)
    true_negatives_count = tf.reduce_sum(tf.cast(true_negatives, tf.int64), axis=0)
    false_negatives_count = tf.reduce_sum(tf.cast(false_negatives, tf.int64), axis=0)
    
    return true_positives_count, false_positives_count, true_negatives_count, false_negatives_count

    
def accuracy(annot_logits, annots):
    """
    Accuracy:
    --------
    Args:
        annot_logits: Tensor, predicted    [batch_size * height * width,  num_annots]
        annots: Tensor, ground truth [batch_size, height, width, 1]
    
    Returns:
        segmentation_acc: Segmentation accuracy
    """

    return segmentation_accuracy(annot_logits, annots)
        

def train(loss, learning_rate, learning_rate_decay_steps, learning_rate_decay_rate, global_step):
    """
    Train opetation:
    ----------
    Args:
        loss: loss to use for training
        learning_rate: Float, learning rate
        learning_rate_decay_steps: Int, amount of steps after which to reduce the learning rate
        learning_rate_decay_rate: Float, decay rate for learning rate

    Returns:
        train_op: Training operation
    """
    
    decayed_learning_rate = tf.train.exponential_decay(learning_rate, global_step, 
                            learning_rate_decay_steps, learning_rate_decay_rate, staircase = True)

    # execute update_ops to update batch_norm weights
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        optimizer   = tf.train.AdamOptimizer(decayed_learning_rate)
        train_op    = optimizer.minimize(loss, global_step = global_step)

    tf.summary.scalar("learning_rate", decayed_learning_rate)

    return train_op


def predictions(annot_logits, batch_size, image_size):
    """
    Prediction operation:
    ----------------
    Args:
        annot_logits: Tensor, predicted    [batch_size * height * width, num_annots]
        batch_size: Int, batch size
        image_size: Int, image width/height
    
    Returns:
        predicted_images: Tensor, predicted images   [batch_size, image_size, image_size]
    """

    predicted_images = tf.reshape(tf.argmax(annot_logits, axis = 1), [batch_size, image_size, image_size])

    return predicted_images

