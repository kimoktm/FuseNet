# ============================================================== #
#                           FuseNet                              #
#                                                                #
#                                                                #
# FuseNet tensorflow implementation WIP                          #
# ============================================================== #

import tensorflow as tf

import utils.layers as layers


def build(color_inputs, depth_inputs, num_annots, num_classes, is_training = True):
    """
    Build fusenet network:
    ----------
    Args:
        color_inputs: Tensor, [batch_size, height, width, 3]
        depth_inputs: Tensor, [batch_size, height, width, 1]
        num_annots: Integer, number of segmentation (annotation) labels
        num_classes: Integer, number of classification labels
        is_training: Boolean, in training mode or not (for dropout & bn)

    Returns:
        annot_logits: Tensor, predicted annotated image flattened 
                              [batch_size * height * width,  num_annots]
        class_logits: Tensor, predicted classes [batch_size , num_classes]
    """

    dropout_keep_prob = 0.5 if is_training else 1.0

    # Encoder Section
    # Block 1
    color_conv1_1 = layers.conv_btn(color_inputs,  [3, 3], 64, 'conv1_1', is_training = is_training)
    color_conv1_2 = layers.conv_btn(color_conv1_1, [3, 3], 64, 'conv1_2', is_training = is_training)
    depth_conv1_1 = layers.conv_btn(depth_inputs,  [3, 3], 64, 'd_conv1_1', is_training = is_training)
    depth_conv1_2 = layers.conv_btn(depth_conv1_1, [3, 3], 64, 'd_conv1_2', is_training = is_training)
    conv1_fuse    = layers.add(color_conv1_2, depth_conv1_2, 'conv1_fuse')
    color_pool1, color_pool1_arg = layers.maxpool_arg(conv1_fuse, [2, 2], 'pool1')
    depth_pool1   = layers.maxpool(depth_conv1_2, [2, 2], 'd_pool1')

    # Block 2
    color_conv2_1 = layers.conv_btn(color_pool1,   [3, 3], 128, 'conv2_1', is_training = is_training)
    color_conv2_2 = layers.conv_btn(color_conv2_1, [3, 3], 128, 'conv2_2', is_training = is_training)
    depth_conv2_1 = layers.conv_btn(depth_pool1,   [3, 3], 128, 'd_conv2_1', is_training = is_training)
    depth_conv2_2 = layers.conv_btn(depth_conv2_1, [3, 3], 128, 'd_conv2_2', is_training = is_training)
    conv2_fuse    = layers.add(color_conv2_2, depth_conv2_2, 'conv2_fuse')
    color_pool2, color_pool2_arg = layers.maxpool_arg(conv2_fuse, [2, 2], 'pool2')
    depth_pool2   = layers.maxpool(depth_conv2_2, [2, 2], 'd_pool2')

    # Block 3
    color_conv3_1 = layers.conv_btn(color_pool2,   [3, 3], 256, 'conv3_1', is_training = is_training)
    color_conv3_2 = layers.conv_btn(color_conv3_1, [3, 3], 256, 'conv3_2', is_training = is_training)
    color_conv3_3 = layers.conv_btn(color_conv3_2, [3, 3], 256, 'conv3_3', is_training = is_training)
    depth_conv3_1 = layers.conv_btn(depth_pool2,   [3, 3], 256, 'd_conv3_1', is_training = is_training)
    depth_conv3_2 = layers.conv_btn(depth_conv3_1, [3, 3], 256, 'd_conv3_2', is_training = is_training)
    depth_conv3_3 = layers.conv_btn(depth_conv3_2, [3, 3], 256, 'd_conv3_3', is_training = is_training)
    conv3_fuse    = layers.add(color_conv3_3, depth_conv3_3, 'conv3_fuse')
    color_pool3, color_pool3_arg = layers.maxpool_arg(conv3_fuse, [2, 2], 'pool3')
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
    color_pool4, color_pool4_arg = layers.maxpool_arg(conv4_fuse, [2, 2], 'pool4')
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
    color_pool5, color_pool5_arg = layers.maxpool_arg(conv5_fuse, [2, 2], 'pool5')
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
    unpool1      = layers.unpool_2x2(decdrop2, color_pool1_arg)
    deconv1_2    = layers.deconv_btn(unpool1, [3, 3], 64, 64, 'deconv1_2', is_training = is_training)
    annot_score  = layers.conv(deconv1_2, [3, 3], num_annots, 'score')
    annot_logits = tf.reshape(annot_score, (-1, num_annots))

    # Classification
    flattend     = layers.flatten(color_drop5, 'flatten')
    fconnected6  = layers.fully_connected(flattend, 4096, 'fc6')
    f6_drop      = layers.dropout(fconnected6, dropout_keep_prob, 'fc6drop')
    fconnected7  = layers.fully_connected(f6_drop, 4096, 'fc7')
    f7_drop      = layers.dropout(fconnected7, dropout_keep_prob, 'fc7drop')
    class_logits = layers.fully_connected(f7_drop, num_classes, 'fc8', activation_fn = None)

    return annot_logits, class_logits


def segmentation_loss(logits, labels):
    """
    Segmentation loss:
    ----------
    Args:
        logits: Tensor, predicted    [batch_size * height * width,  num_annots]
        labels: Tensor, ground truth [batch_size, height, width, 1]

    Returns:
        segment_loss: Segmentation loss
    """

    labels = tf.reshape(labels, [-1])
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels = labels, logits = logits, name = 'segment_cross_entropy_per_example')
    segment_loss  = tf.reduce_mean(cross_entropy, name = 'segment_cross_entropy')

    return segment_loss


def classification_loss(logits, labels):
    """
    Classification loss:
    ----------
    Args:
        logits: Tensor, predicted    [batch_size,  num_classes]
        labels: Tensor, ground truth [batch_size, 1]

    Returns:
        class_loss: Classification loss
    """

    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels = labels, logits = logits, name = 'class_cross_entropy_per_example')
    class_loss = tf.reduce_mean(cross_entropy, name = 'class_cross_entropy')

    return class_loss

def l2_loss():
    """
    L2 loss:
    -------
    Returns:
        l2_loss: L2 loss for all weights
    """
    
    weights = [var for var in tf.trainable_variables() if var.name.endswith('weights:0')]
    l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in weights])
    return l2_loss
    
def loss(annot_logits, annots, class_logits, classes, weight_decay_factor):
    """
    Total loss:
    ----------
    Args:
        annot_logits: Tensor, predicted    [batch_size * height * width,  num_annots]
        annots: Tensor, ground truth [batch_size, height, width, 1]
        class_logits: Tensor, predicted    [batch_size,  num_classes]
        classes: Tensor, ground truth [batch_size, 1]
        weight_decay_factor: float, factor with which weights are decayed

    Returns:
        total_loss: Segmentation + Classification losses + WeightDecayFactor * L2 loss
    """

    segment_loss = segmentation_loss(annot_logits, annots)
    class_loss   = classification_loss(class_logits, classes)
    total_loss   = segment_loss + class_loss + weight_decay_factor * l2_loss()

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
    predicted_annots = tf.argmax(logits, axis=1)
    correct_predictions = tf.equal(predicted_annots, labels)
    segmentation_accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    return segmentation_accuracy


def classification_accuracy(logits, labels):
    """
    Classification accuracy:
    ----------
    Args:
        logits: Tensor, predicted    [batch_size,  num_classes]
        labels: Tensor, ground truth [batch_size, 1]

    Returns:
        class_accuracy: Classification accuracy
    """
    
    predicted_classes = tf.argmax(logits, axis=1)
    correct_predictions = tf.equal(predicted_classes, labels)
    class_accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    return class_accuracy 

    
def accuracy(annot_logits, annots, class_logits, classes):
    """
    Accuracy:
    --------
    Args:
        annot_logits: Tensor, predicted    [batch_size * height * width,  num_annots]
        annots: Tensor, ground truth [batch_size, height, width, 1]
        class_logits: Tensor, predicted    [batch_size,  num_classes]
        classes: Tensor, ground truth [batch_size, 1]
    
    Returns:
        total_accuracy: Segmentation + Classification accuracies
        segmentation_acc: Segmentation accuracy
        classification_acc: Classification accuracy
    """
    
    segmentation_acc = segmentation_accuracy(annot_logits, annots)
    classification_acc = classification_accuracy(class_logits, classes)
    total_accuracy = (segmentation_acc + classification_acc)/2

    return total_accuracy, segmentation_acc, classification_acc
        

def train(loss, learning_rate, learning_rate_decay_steps, learning_rate_decay_rate):
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
    
    global_step = tf.Variable(0, name = 'global_step', trainable = False)
    decayed_learning_rate = tf.train.exponential_decay(learning_rate, global_step, 
                            learning_rate_decay_steps, learning_rate_decay_rate, staircase = True)
    optimizer   = tf.train.AdamOptimizer(decayed_learning_rate)
    train_op    = optimizer.minimize(loss, global_step = global_step)

    return train_op
