# ============================================================== #
#                             Tools                              #
#                                                                #
#                                                                #
# Tools to download and intialize graph with vgg weights         #
# ============================================================== #

from __future__ import print_function

import numpy as np
import tensorflow as tf


def load_vgg_weights(vgg_path, graph, sess):
    """
    Load vgg weights from the given numpy file and assign 
    weights to their correspondants in the given graph. 
    Layers should have the same vgg names inorder to work
    (conv1_1, conv1_2..). depth variables will also be 
    assigned (d_conv1_1..)
    ----------
    Args:
        vgg_path: path to vgg weights numpy file
        graph: tensorflow graph to assign weights to
        sess: current tensorflow session
    """

    if vgg_path is None:
        print('[ERROR   ]\tVgg not found. skipping vgg wights initialization')

    data_dict = np.load(vgg_path, encoding = 'latin1').item()
    variables = graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    for v in variables:
        variable_name  = v.name[0:-2].split('/')
        layer_name     = variable_name[0]
        vgg_layer_name = layer_name[2:] if layer_name.startswith('d_') else layer_name

        if vgg_layer_name in data_dict:
            if v.name.endswith('weights:0'):

                # convert from vgg bgr to fusenet rgb
                if variable_name[0] == 'conv1_1':
                    print('[PROGRESS]\tAssigning %s by reshaping bgr to rgb' % v.name[0:-2])
                    bgr_weights = data_dict[vgg_layer_name][0]
                    rgb_weights = np.stack((bgr_weights[:,:,2,:], bgr_weights[:,:,1,:], bgr_weights[:,:,0,:]), axis = 2)
                    sess.run(v.assign(rgb_weights))

                # average rgb weights to one channel weights (x, y, 3, z) -> (x, y, 1, z)
                elif variable_name[0] == 'd_conv1_1':
                    print('[PROGRESS]\tAssigning %s by averaging rgb to one channel' % v.name[0:-2])
                    avg_weights = np.mean(data_dict[vgg_layer_name][0], axis = 2, keepdims = True)
                    sess.run(v.assign(avg_weights))

                # assign weights if tensors have the same shape
                elif np.array_equal(v.get_shape(), data_dict[vgg_layer_name][0].shape):
                    print('[PROGRESS]\tAssigning %s' % v.name[0:-2])
                    sess.run(v.assign(data_dict[vgg_layer_name][0]))

            elif v.name.endswith('bias:0'):

                # assign bias weights if they have the same shape
                if np.array_equal(v.get_shape(), data_dict[vgg_layer_name][1].shape):
                    print('[PROGRESS]\tAssigning %s' % v.name[0:-2])
                    sess.run(v.assign(data_dict[vgg_layer_name][1]))
            else:
                print('[WARNING ]\tSkipping %s as its not found in VGG' % v.name[0:-2])

    print('[INFO    ]\tVGG weights loading complete')


