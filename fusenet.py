import tensorflow as tf

class FuseNet:

    def run(self, input, dropout_keep_prob):
        # Layer1
        rgb_convolution1_1 = self.convolution_layer(input, output_channels=64, height=3, width=3, name='rgb_conv1_1')
        rgb_convolution1_2 = self.convolution_layer(rgb_convolution1_1, output_channels=64, height=3, width=3, name='rgb_conv1_2')

        d_convolution1_1 = self.convolution_layer(input, output_channels=64, height=3, width=3, name='d_conv1_1')
        d_convolution1_2 = self.convolution_layer(d_convolution1_1, output_channels=64, height=3, width=3, name='d_conv1_2')

        fuse1 = self.fuse_layer(rgb_convolution1_2, d_convolution1_2, name='fuse1')

        rgb_pool1 = self.maxpool_layer(fuse1, height=2, width=2, name='rbg_pool1')
        d_pool1 = self.maxpool_layer(d_convolution1_2, height=2, width=2, name='d_pool1')

        # Layer2
        rgb_convolution2_1 = self.convolution_layer(rgb_pool1, output_channels=128, height=3, width=3, name='rgb_conv2_1')
        rgb_convolution2_2 = self.convolution_layer(rgb_convolution2_1, output_channels=128, height=3, width=3, name='rgb_conv2_2')

        d_convolution2_1 = self.convolution_layer(d_pool1, output_channels=128, height=3, width=3, name='d_conv2_1')
        d_convolution2_2 = self.convolution_layer(d_convolution2_1, output_channels=128, height=3, width=3, name='d_conv2_2')

        fuse2 = self.fuse_layer(rgb_convolution2_2, d_convolution2_2, name='fuse2')

        rgb_pool2 = self.maxpool_layer(fuse2, height=2, width=2, name='rbg_pool2')
        d_pool2 = self.maxpool_layer(d_convolution2_2, height=2, width=2, name='d_pool2')

        # Layer3
        rgb_convolution3_1 = self.convolution_layer(rgb_pool2, output_channels=256, height=3, width=3, name='rgb_conv3_1')
        rgb_convolution3_2 = self.convolution_layer(rgb_convolution3_1, output_channels=256, height=3, width=3, name='rgb_conv3_2')
        rgb_convolution3_3 = self.convolution_layer(rgb_convolution3_2, output_channels=256, height=3, width=3, name='rgb_conv3_3')

        d_convolution3_1 = self.convolution_layer(d_pool2, output_channels=256, height=3, width=3, name='d_conv3_1')
        d_convolution3_2 = self.convolution_layer(d_convolution3_1, output_channels=256, height=3, width=3, name='d_conv3_2')
        d_convolution3_3 = self.convolution_layer(d_convolution3_2, output_channels=256, height=3, width=3, name='d_conv3_3')

        fuse3 = self.fuse_layer(rgb_convolution3_3, d_convolution3_3, name='fuse3')

        rgb_pool3 = self.maxpool_layer(fuse3, height=2, width=2, name='rbg_pool3')
        d_pool3 = self.maxpool_layer(d_convolution3_3, height=2, width=2, name='d_pool3')

        rgb_pool3_dropout = tf.nn.dropout(rgb_pool3, keep_prob=dropout_keep_prob, name='rgb_pool3_dropout')
        d_pool3_dropout = tf.nn.dropout(d_pool3, keep_prob=dropout_keep_prob, name='d_pool3_dropout')

        # Layer4
        rgb_convolution4_1 = self.convolution_layer(rgb_pool3_dropout, output_channels=512, height=3, width=3, name='rgb_conv4_1')
        rgb_convolution4_2 = self.convolution_layer(rgb_convolution4_1, output_channels=512, height=3, width=3, name='rgb_conv4_2')
        rgb_convolution4_3 = self.convolution_layer(rgb_convolution4_2, output_channels=512, height=3, width=3, name='rgb_conv4_3')

        d_convolution4_1 = self.convolution_layer(d_pool3_dropout, output_channels=512, height=3, width=3, name='d_conv4_1')
        d_convolution4_2 = self.convolution_layer(d_convolution4_1, output_channels=512, height=3, width=3, name='d_conv4_2')
        d_convolution4_3 = self.convolution_layer(d_convolution4_2, output_channels=512, height=3, width=3, name='d_conv4_3')

        fuse4 = self.fuse_layer(rgb_convolution4_3, d_convolution4_3, name='fuse4')

        rgb_pool4 = self.maxpool_layer(fuse4, height=2, width=2, name='rbg_pool4')
        d_pool4 = self.maxpool_layer(d_convolution4_3, height=2, width=2, name='d_pool4')

        rgb_pool4_dropout = tf.nn.dropout(rgb_pool4, keep_prob=dropout_keep_prob, name='rgb_pool4_dropout')
        d_pool4_dropout = tf.nn.dropout(d_pool4, keep_prob=dropout_keep_prob, name='d_pool4_dropout')

        # Layer5
        rgb_convolution5_1 = self.convolution_layer(rgb_pool4_dropout, output_channels=512, height=3, width=3, name='rgb_conv5_1')
        rgb_convolution5_2 = self.convolution_layer(rgb_convolution5_1, output_channels=512, height=3, width=3, name='rgb_conv5_2')
        rgb_convolution5_3 = self.convolution_layer(rgb_convolution5_2, output_channels=512, height=3, width=3, name='rgb_conv5_3')

        d_convolution5_1 = self.convolution_layer(d_pool4_dropout, output_channels=512, height=3, width=3, name='d_conv5_1')
        d_convolution5_2 = self.convolution_layer(d_convolution5_1, output_channels=512, height=3, width=3, name='d_conv5_2')
        d_convolution5_3 = self.convolution_layer(d_convolution5_2, output_channels=512, height=3, width=3, name='d_conv5_3')

        fuse5 = self.fuse_layer(rgb_convolution5_3, d_convolution5_3, name='fuse5')

        rgb_pool5 = self.maxpool_layer(fuse5, height=2, width=2, name='rbg_pool5')
        d_pool5 = self.maxpool_layer(d_convolution5_3, height=2, width=2, name='d_pool5')

        rgb_pool5_dropout = tf.nn.dropout(rgb_pool5, keep_prob=dropout_keep_prob, name='rgb_pool5_dropout')
        d_pool5_dropout = tf.nn.dropout(d_pool5, keep_prob=dropout_keep_prob, name='d_pool5_dropout')

        return rgb_pool5_dropout


    def maxpool_layer(self, input, height, width, name):
        return tf.nn.max_pool(input,
                              ksize=[1, height, width, 1],
                              strides=[1, height, width, 1],
                              padding='SAME',
                              name=name)

    def convolution_layer(self, input, output_channels, height, width, name):
        with tf.name_scope(name) as scope:
            input_channels = input.get_shape()[-1].value
            filter = tf.Variable(tf.truncated_normal([height, width, input_channels, output_channels],
                                                     stddev=0.001),
                                 name='weights')
            biases = tf.Variable(tf.truncated_normal([output_channels],
                                                     stddev=0.001),
                                 trainable=True,
                                 name='biases')

            convolution = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')
            z = tf.nn.bias_add(convolution, biases)
            #TODO: Batch normalization
            return tf.nn.relu(z, name=scope)

    def fuse_layer(self, input1, input2, name):
        return tf.add(input1, input2, name=name)

    def fully_connected_layer(self, input, output_number, name):
        with tf.name_scope(name) as scope:
            input_number = input.get_shape()[-1].value
            weights = tf.Variable(tf.truncated_normal([input_number, output_number],
                                                      stddev=0.001),
                                  name='weights')
            bias = tf.Variable(tf.truncated_normal([output_number],
                                                   stddev=0.001),
                               name='bias')

            z = tf.matmul(input, weights) + bias
            return tf.nn.relu(z, name=scope)

    ##### Unpooling
    # https://github.com/tensorflow/tensorflow/issues/2169
    def unpool_layer(self, input, name):
        """N-dimensional version of the unpooling operation from
     https://www.robots.ox.ac.uk/~vgg/rg/papers/Dosovitskiy_Learning_to_Generate_2015_CVPR_paper.pdf

        :param tensor: A Tensor of shape [b, d0, d1, ..., dn, ch]
        :return: A Tensor of shape [b, 2*d0, 2*d1, ..., 2*dn, ch]
        """
        with tf.name_scope(name) as scope:
            sh = input.get_shape().as_list()
            dim = len(sh[1:-1])
            out = (tf.reshape(input, [-1] + sh[-dim:]))
            for i in range(dim, 0, -1):
                out = tf.concat(i, [out, tf.zeros_like(out)])
            out_size = [-1] + [s * 2 for s in sh[1:-1]] + [sh[-1]]
            out = tf.reshape(out, out_size, name=scope)
        return out
